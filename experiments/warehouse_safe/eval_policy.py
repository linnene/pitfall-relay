"""Evaluate fixed Warehouse-A actors and record local execution/latent data.

The env is stepped directly (no training, no online update, no epsilon
exploration). Each agent chooses from its own observation and GRU hidden state
through the unchanged actor. Actor forward hooks are read-only.

All per-step arrays and event logs are saved; event truth is stored separately
from the detector inputs.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


def sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
    tmp.replace(path)


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_eval_env(config):
    import gym
    import macro_marl.my_env  # noqa: F401
    run = config["run"]
    params = {
        "grid_dim": [6, 6],
        "big_box_reward": 300.0,
        "small_box_reward": 20.0,
        "penalty": -10.0,
        "n_agent": run["n_agent"],
        "terminate_step": run["env_terminate_step"],
        "TB_move_speed": run["tb_m_speed"],
        "TB_move_noisy": 0.0,
        "TB_move_cost": 0.0,
        "fetch_pass_obj_tc": 4,
        "fetch_look_for_obj_tc": 6,
        "fetch_manip_noisy": 0.0,
        "delay_delivery_penalty": run["d_pen"],
        "obs_human_wait": False,
        "human_speed_per_step": [run["h0_speed_ps"], run["h1_speed_ps"]],
    }
    return gym.make(config["env_id"], **params)


def build_controller(env, config, device="cpu"):
    from macro_marl.cores.pg_based.mac_iaicc.controller import MAC
    run = config["run"]
    return MAC(
        env,
        obs_last_action=False,
        a_mlp_layer_size=[32, 32],
        a_rnn_layer_size=run["a_rnn_layer_size"],
        c_mlp_layer_size=[32, 32],
        c_rnn_layer_size=run["c_rnn_layer_size"],
        device=device,
    )


def load_actor_state(checkpoint_dir, controller):
    import torch
    checkpoint_dir = Path(checkpoint_dir)
    loaded = []
    for idx, agent in enumerate(controller.agents):
        candidates = sorted(checkpoint_dir.glob(f"*_agent_{idx}1.tar"))
        if not candidates:
            raise FileNotFoundError(f"no actor checkpoint for agent {idx} in {checkpoint_dir}")
        payload = torch.load(candidates[0], map_location="cpu")
        agent.actor_net.load_state_dict(payload["actor_net_state_dict"])
        agent.actor_net.eval()
        loaded.append({"agent": idx, "path": str(candidates[0]),
                       "sha256": sha256_file(candidates[0])})
    return loaded


class HookRecorder:
    def __init__(self, controller, enabled=True):
        self.enabled = enabled
        self.handles = []
        self.current = {}
        if enabled:
            import torch
            for idx, agent in enumerate(controller.agents):
                def make_hook(agent_idx):
                    def hook(module, inputs, output):
                        x = inputs[0]
                        h_in = inputs[1] if len(inputs) > 1 else None
                        logits, h_out = output
                        def flat(t):
                            if t is None:
                                return np.zeros(agent.actor_net.gru.hidden_size, dtype=np.float32)
                            return t.detach().cpu().reshape(-1).numpy().astype(np.float32)
                        self.current[agent_idx] = {
                            "obs": x.detach().cpu().reshape(-1).numpy().astype(np.float32),
                            "h_in": flat(h_in),
                            "h_out": flat(h_out),
                            "logits": logits.detach().cpu().reshape(-1).numpy().astype(np.float32),
                        }
                    return hook
                self.handles.append(agent.actor_net.register_forward_hook(make_hook(idx)))

    def clear(self):
        self.current = {}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def snapshot_agents(env):
    env = env.unwrapped
    agents = env.agents
    turtles = agents[:2]
    fetch = agents[2]
    return {
        "fetch_action": int(fetch.cur_action.idx) if fetch.cur_action is not None else -1,
        "fetch_done": bool(fetch.cur_action_done),
        "fetch_serving": bool(fetch.serving),
        "fetch_serving_failed": bool(fetch.serving_failed),
        "fetch_time_left": float(fetch.cur_action_time_left),
        "ready_objs": np.asarray(fetch.ready_objs, dtype=np.float32).copy(),
        "found_objs": list(fetch.found_objs),
        "baskets": [np.asarray(t.objs_in_basket, dtype=np.float32).copy() for t in turtles],
        "turtle_actions": [int(t.cur_action.idx) if t.cur_action is not None else -1 for t in turtles],
        "turtle_done": [bool(t.cur_action_done) for t in turtles],
        "turtle_time_left": [float(t.cur_action_time_left) for t in turtles],
        "turtle_bwp": [int(t.cur_BWP.idx) if t.cur_BWP is not None else -1 for t in turtles],
        "finish_count": len(env.finish_serve_humans),
        "humans": [
            {
                "cur_step": int(h.cur_step),
                "time_left": float(h.cur_step_time_left),
                "next_obj": int(h.next_request_obj_idx),
                "obtained": bool(h.next_requested_obj_obtained),
                "finished": bool(h.whole_task_finished),
            }
            for h in env.humans
        ],
    }


def install_step_wrappers(env, recorder):
    env = env.unwrapped
    for idx, turtle_obj in enumerate(env.agents[:2]):
        original = turtle_obj.step
        def make_wrapper(agent_idx, turtle, original_step):
            def wrapper(action, humans):
                before_basket = np.asarray(turtle.objs_in_basket, dtype=np.float32).copy()
                before_action = int(turtle.cur_action.idx) if turtle.cur_action is not None else -1
                before_time = float(turtle.cur_action_time_left)
                reward = original_step(action, humans)
                after_basket = np.asarray(turtle.objs_in_basket, dtype=np.float32).copy()
                gained = float(np.sum(after_basket - before_basket))
                if (before_action == 3 and bool(turtle.cur_action_done)
                        and float(turtle.cur_action_time_left) <= -turtle.get_tool_wait + 1e-9
                        and gained <= 0.0):
                    recorder.append_event({
                        "type": "tool_wait_timeout",
                        "agent": agent_idx,
                        "fetch_action": int(env.agents[2].cur_action.idx) if env.agents[2].cur_action is not None else -1,
                        "turtle_action": before_action,
                        "time_left": float(turtle.cur_action_time_left),
                        "basket": after_basket.tolist(),
                    })
                return reward
            return wrapper
        turtle_obj.step = make_wrapper(idx, turtle_obj, original)

    fetch = env.agents[2]
    original_fetch = fetch.step
    def fetch_wrapper(action, agents):
        before_failed = bool(fetch.serving_failed)
        before_action = int(fetch.cur_action.idx) if fetch.cur_action is not None else -1
        before_time = float(fetch.cur_action_time_left)
        before_baskets = [np.asarray(t.objs_in_basket, dtype=np.float32).copy() for t in agents[:2]]
        before_ready = np.asarray(fetch.ready_objs, dtype=np.float32).copy()
        reward = original_fetch(action, agents)
        after_baskets = [np.asarray(t.objs_in_basket, dtype=np.float32).copy() for t in agents[:2]]
        after_failed = bool(fetch.serving_failed)
        after_action = int(fetch.cur_action.idx) if fetch.cur_action is not None else -1
        completed = before_action in (1, 2) and before_time <= 0.0
        if completed and before_failed:
            recorder.append_event({
                "type": "handoff_failure",
                "agent": 2,
                "target": before_action - 1,
                "fetch_action": before_action,
                "baskets_before": [b.tolist() for b in before_baskets],
                "baskets_after": [b.tolist() for b in after_baskets],
                "ready_before": before_ready.tolist(),
            })
        elif completed and not before_failed:
            target = before_action - 1
            gain = float(np.sum(after_baskets[target] - before_baskets[target]))
            if gain > 0:
                recorder.append_event({
                    "type": "handoff_success",
                    "agent": 2,
                    "target": target,
                    "fetch_action": before_action,
                    "baskets_before": [b.tolist() for b in before_baskets],
                    "baskets_after": [b.tolist() for b in after_baskets],
                })
            else:
                recorder.append_event({
                    "type": "handoff_failure",
                    "agent": 2,
                    "target": target,
                    "fetch_action": before_action,
                    "baskets_before": [b.tolist() for b in before_baskets],
                    "baskets_after": [b.tolist() for b in after_baskets],
                })
        if before_failed is False and after_failed is True:
            recorder.append_event({"type": "serving_failed_start", "agent": 2,
                                   "fetch_action": after_action})
        return reward
    fetch.step = fetch_wrapper
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hooks", choices=["on", "off"], default="on")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--flex-sha", default="unknown")
    args = parser.parse_args()

    upstream_root = Path(args.upstream_root).resolve()
    sys.path.insert(0, str(upstream_root / "src"))
    import torch
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1) if hasattr(torch, "set_num_interop_threads") else None

    out_dir = Path(args.out_dir).resolve()
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episode_summaries = []
    all_events = []

    env = build_eval_env(config)
    controller = build_controller(env, config, args.device)
    loaded = load_actor_state(args.checkpoint_dir, controller)
    hooks = HookRecorder(controller, enabled=(args.hooks == "on"))
    max_steps = int(config["run"]["env_terminate_step"])

    for episode_idx in range(args.episodes):
        episode_seed = int(args.eval_seed) + episode_idx
        seed_all(episode_seed)
        obs = env.reset()
        obs = [torch.from_numpy(o).float().view(1, -1) for o in obs]
        h_states = [None] * env.unwrapped.n_agent
        valids = [1.0] * env.unwrapped.n_agent
        avail = env.get_avail_actions()
        avail_t = [torch.FloatTensor(a).view(1, -1) for a in avail]
        actions_hist = []
        actual_hist = []
        mac_done_hist = []
        valid_hist = []
        reward_hist = []
        done_hist = []
        obs_hist = []
        h_in_hist = []
        h_out_hist = []
        logits_hist = []
        state_hist = []
        event_local = []

        recorder = type("R", (), {})()
        recorder.events = []
        recorder.step = 0
        recorder.append_event = lambda e, r=recorder: r.events.append(dict(e, step=r.step))
        install_step_wrappers(env, recorder)
        done = False
        ep_return = 0.0
        discounted = 0.0
        for step in range(max_steps):
            recorder.step = step
            hooks.clear()
            # controller.select_action expects list of [1, obs_dim]
            obs_list = [o.view(1, -1) for o in obs]
            valid_mask = [bool(v) for v in valids]
            actions, new_h = controller.select_action(
                obs_list, h_states, valid_mask,
                avail_t, eps=0.0, test_mode=True,
            )
            hook_snapshot = dict(hooks.current)
            actions_hist.append([int(a) for a in actions])
            valid_hist.append([int(v) for v in valid_mask])
            before = snapshot_agents(env)
            next_obs, rewards, done, info = env.step(actions)
            after = snapshot_agents(env)
            actual = [int(x) for x in info["cur_mac"]]
            mac_done = [int(x) for x in info["mac_done"]]
            actual_hist.append(actual)
            mac_done_hist.append(mac_done)
            reward_hist.append(float(np.sum(rewards)))
            done_hist.append(int(done))
            ep_return += float(np.sum(rewards))
            discounted += float(np.sum(rewards)) * (config["run"]["gamma"] ** step)
            state_hist.append({"before": before, "after": after})
            obs_hist.append([np.asarray(o, dtype=np.float32).reshape(-1) for o in next_obs])
            h_in = np.zeros((env.unwrapped.n_agent, config["run"]["a_rnn_layer_size"]), dtype=np.float32)
            h_out = np.zeros_like(h_in)
            logits = np.zeros((env.unwrapped.n_agent, 8), dtype=np.float32)
            for agent_idx, rec in hook_snapshot.items():
                if rec is None:
                    continue
                h_in[agent_idx, :rec["h_in"].size] = rec["h_in"]
                h_out[agent_idx, :rec["h_out"].size] = rec["h_out"]
                logits[agent_idx, :rec["logits"].size] = rec["logits"]
            h_in_hist.append(h_in)
            h_out_hist.append(h_out)
            logits_hist.append(logits)
            # h_states after this decision: select_action already returned them.
            h_states = new_h
            obs = [torch.from_numpy(o).float().view(1, -1) for o in next_obs]
            valids = [1.0 if m else 0.0 for m in mac_done]
            avail = env.get_avail_actions()
            avail_t = [torch.FloatTensor(a).view(1, -1) for a in avail]
            if done:
                break

        steps = len(actions_hist)
        # Determine next decision after each event: valid actor call for any/relevant agent.
        valid_arr = np.array(valid_hist, dtype=np.int8)
        events = []
        for event in recorder.events:
            event = dict(event)
            event_step = int(event["step"])
            if event["type"] in ("handoff_failure", "handoff_success"):
                relevant_agents = [int(event.get("target", 0))]
            elif event["type"] == "tool_wait_timeout":
                relevant_agents = [2]
            else:
                relevant_agents = list(range(env.unwrapped.n_agent))
            next_any = None
            next_relevant = None
            first_agent_decision = None
            for future in range(event_step + 1, steps):
                if next_any is None and np.any(valid_arr[future] > 0):
                    next_any = future
                if valid_arr[future, int(event.get("agent", 0))] > 0 and first_agent_decision is None:
                    first_agent_decision = future
                if any(valid_arr[future, a] > 0 for a in relevant_agents):
                    next_relevant = future
                    break
            event["next_any_decision_step"] = next_any
            event["next_relevant_decision_step"] = next_relevant
            event["first_agent_decision_step"] = first_agent_decision
            event["event_episode"] = episode_idx
            event["event_seed"] = episode_seed
            events.append(event)
            all_events.append({"episode": episode_idx, "seed": episode_seed, **event})

        obs_max = max(o.size for frame in obs_hist for o in frame)
        obs_arr = np.zeros((steps, env.unwrapped.n_agent, obs_max), dtype=np.float32)
        for t, frame in enumerate(obs_hist):
            for a, o in enumerate(frame):
                obs_arr[t, a, :o.size] = o
        max_act = max(env.unwrapped.n_action)
        logits_pad = np.zeros((steps, env.unwrapped.n_agent, max_act), dtype=np.float32)
        for t, l in enumerate(logits_hist):
            for a in range(env.unwrapped.n_agent):
                n = env.unwrapped.n_action[a]
                logits_pad[t, a, :n] = l[a, :n]
        arrays = {
            "selected_action": np.array(actions_hist, dtype=np.int16),
            "actual_action": np.array(actual_hist, dtype=np.int16),
            "mac_done": np.array(mac_done_hist, dtype=np.int8),
            "valid": np.array(valid_hist, dtype=np.int8),
            "reward_sum": np.array(reward_hist, dtype=np.float32),
            "done": np.array(done_hist, dtype=np.int8),
            "obs": obs_arr,
            "h_in": np.array(h_in_hist, dtype=np.float32),
            "h_out": np.array(h_out_hist, dtype=np.float32),
            "logits": logits_pad,
            "fetch_action": np.array([s["after"]["fetch_action"] for s in state_hist], dtype=np.int16),
            "fetch_serving_failed": np.array([s["after"]["fetch_serving_failed"] for s in state_hist], dtype=np.int8),
            "fetch_ready_objs": np.array([s["after"]["ready_objs"] for s in state_hist], dtype=np.float32),
            "baskets": np.array([s["after"]["baskets"] for s in state_hist], dtype=np.float32),
            "turtle_actions": np.array([s["after"]["turtle_actions"] for s in state_hist], dtype=np.int16),
            "turtle_time_left": np.array([s["after"]["turtle_time_left"] for s in state_hist], dtype=np.float32),
            "human_step": np.array([[h["cur_step"] for h in s["after"]["humans"]] for s in state_hist], dtype=np.int16),
            "human_time_left": np.array([[h["time_left"] for h in s["after"]["humans"]] for s in state_hist], dtype=np.float32),
            "human_obtained": np.array([[h["obtained"] for h in s["after"]["humans"]] for s in state_hist], dtype=np.int8),
            "human_finished": np.array([[h["finished"] for h in s["after"]["humans"]] for s in state_hist], dtype=np.int8),
            "finish_count": np.array([s["after"]["finish_count"] for s in state_hist], dtype=np.int16),
        }
        stem = episodes_dir / f"episode_{episode_idx:05d}"
        np.savez_compressed(str(stem) + ".npz", **arrays)
        human_finish_steps = []
        for h in range(env.unwrapped.n_human):
            hits = np.nonzero(arrays["human_finished"][:, h] > 0)[0]
            human_finish_steps.append(int(hits[0]) + 1 if len(hits) else None)
        summary = {
            "episode": episode_idx,
            "seed": episode_seed,
            "success": bool(done),
            "steps": steps,
            "human_finish_steps": human_finish_steps,
            "undiscounted_return": ep_return,
            "discounted_return": discounted,
            "events": events,
            "event_counts": {name: sum(1 for e in events if e["type"] == name)
                             for name in ("handoff_failure", "handoff_success", "tool_wait_timeout")},
            "final_human_finished": [bool(h.whole_task_finished) for h in env.unwrapped.humans],
            "final_finish_count": len(env.unwrapped.finish_serve_humans),
            "label": args.label,
            "checkpoint_dir": str(args.checkpoint_dir),
            "hooks": args.hooks,
            "training_seed": args.seed,
            "eval_seed": args.eval_seed,
            "flex_sha": args.flex_sha,
            "arrays": str(stem) + ".npz",
            "array_sha256": sha256_file(str(stem) + ".npz"),
        }
        write_json(stem.with_suffix(".json"), summary)
        episode_summaries.append(summary)
        # restore wrappers for next episode (env.reset creates new agents; reinstall per episode)
        env = build_eval_env(config)

    hooks.close()
    events_path = out_dir / "events.jsonl"
    with open(events_path, "w", encoding="utf-8") as handle:
        for event in all_events:
            handle.write(json.dumps(json_safe(event), sort_keys=True) + "\n")
    summary = {
        "label": args.label,
        "checkpoint_dir": str(args.checkpoint_dir),
        "loaded_actor": loaded,
        "training_seed": args.seed,
        "eval_seed": args.eval_seed,
        "episodes_requested": args.episodes,
        "episodes_completed": len(episode_summaries),
        "successes": sum(1 for e in episode_summaries if e["success"]),
        "success_rate": (sum(1 for e in episode_summaries if e["success"]) / len(episode_summaries)
                         if episode_summaries else None),
        "mean_steps": float(np.mean([e["steps"] for e in episode_summaries])) if episode_summaries else None,
        "mean_return": float(np.mean([e["undiscounted_return"] for e in episode_summaries])) if episode_summaries else None,
        "event_counts": {name: sum(e["event_counts"][name] for e in episode_summaries)
                         for name in ("handoff_failure", "handoff_success", "tool_wait_timeout")},
        "episodes": episode_summaries,
        "hooks": args.hooks,
        "flex_sha": args.flex_sha,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(json_safe({k: summary[k] for k in ("successes", "episodes_completed", "success_rate",
                                                       "mean_steps", "mean_return", "event_counts")}), indent=2))
    env.close()


if __name__ == "__main__":
    main()
