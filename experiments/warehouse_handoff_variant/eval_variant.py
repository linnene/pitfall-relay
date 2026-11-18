"""Evaluate a frozen Warehouse-A Actor under the handoff-interface stall variant."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "warehouse_safe"))
import eval_policy as base  # noqa: E402
from variant import (  # noqa: E402
    StallController, link_windows, sample_stall_intervals,
    t1_decision_record, window_events,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--variant-config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hooks", choices=["on", "off"], default="on")
    parser.add_argument("--stall-control", action="store_true",
                        help="same episode seeds/intervals but stall disabled")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--flex-sha", default="unknown")
    args = parser.parse_args()

    variant_config = json.loads(Path(args.variant_config).read_text())
    base_config = json.loads(Path(variant_config["base_config"]).read_text())
    stall_cfg = variant_config["stall"]
    upstream_root = Path(args.upstream_root).resolve()
    sys.path.insert(0, str(upstream_root / "src"))
    import torch
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    out_dir = Path(args.out_dir).resolve()
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episode_summaries = []
    all_variant_events = []
    all_stall_failures = []
    all_linked = []

    env = base.build_eval_env(base_config)
    controller = base.build_controller(env, base_config, args.device)
    loaded = base.load_actor_state(args.checkpoint_dir, controller)
    hooks = base.HookRecorder(controller, enabled=(args.hooks == "on"))
    max_steps = int(base_config["run"]["env_terminate_step"])
    n_agent = env.unwrapped.n_agent

    for episode_idx in range(args.episodes):
        episode_seed = int(args.eval_seed) + episode_idx
        intervals = sample_stall_intervals(
            episode_seed, max_steps=max_steps,
            n_intervals=int(stall_cfg.get("intervals_per_episode", 2)),
            min_length=int(stall_cfg.get("min_length", 6)),
            max_length=int(stall_cfg.get("max_length", 14)),
            salt=int(stall_cfg.get("salt", 20261118)),
        )
        stall = StallController(intervals, enabled=not args.stall_control)
        base.seed_all(episode_seed)
        obs = env.reset()
        obs = [torch.from_numpy(o).float().view(1, -1) for o in obs]
        h_states = [None] * n_agent
        valids = [1.0] * n_agent
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
        stall_hist = []
        state_hist = []
        t1_decisions = []

        shared = {}
        recorder = type("R", (), {})()
        recorder.events = []
        recorder.step = 0
        recorder.append_event = lambda e, r=recorder: r.events.append(dict(e, step=r.step))
        base.install_step_wrappers(env, recorder)
        stall.install(env.unwrapped.agents[2], recorder, shared)

        done = False
        ep_return = 0.0
        discounted = 0.0
        for step in range(max_steps):
            recorder.step = step
            hooks.clear()
            obs_list = [o.view(1, -1) for o in obs]
            valid_mask = [bool(v) for v in valids]
            actions, new_h = controller.select_action(
                obs_list, h_states, valid_mask, avail_t, eps=0.0, test_mode=True,
            )
            hook_snapshot = dict(hooks.current)
            actions_hist.append([int(a) for a in actions])
            valid_hist.append([int(v) for v in valid_mask])
            if valid_mask[1]:
                t1_decisions.append(t1_decision_record(
                    env, step, actions[1], stall, obs[1].detach().cpu().numpy()))
            before = base.snapshot_agents(env)
            next_obs, rewards, done, info = env.step(actions)
            after = base.snapshot_agents(env)
            actual = [int(x) for x in info["cur_mac"]]
            mac_done = [int(x) for x in info["mac_done"]]
            actual_hist.append(actual)
            mac_done_hist.append(mac_done)
            reward_hist.append(float(np.sum(rewards)))
            done_hist.append(int(done))
            ep_return += float(np.sum(rewards))
            discounted += float(np.sum(rewards)) * (base_config["run"]["gamma"] ** step)
            stall_hist.append(stall.active(step))
            state_hist.append({"before": before, "after": after})
            obs_hist.append([np.asarray(o, dtype=np.float32).reshape(-1) for o in next_obs])
            h_in = np.zeros((n_agent, base_config["run"]["a_rnn_layer_size"]), dtype=np.float32)
            h_out = np.zeros_like(h_in)
            logits = np.zeros((n_agent, 8), dtype=np.float32)
            for agent_idx, rec in hook_snapshot.items():
                if rec is None:
                    continue
                h_in[agent_idx, :rec["h_in"].size] = rec["h_in"]
                h_out[agent_idx, :rec["h_out"].size] = rec["h_out"]
                logits[agent_idx, :rec["logits"].size] = rec["logits"]
            h_in_hist.append(h_in)
            h_out_hist.append(h_out)
            logits_hist.append(logits)
            h_states = new_h
            obs = [torch.from_numpy(o).float().view(1, -1) for o in next_obs]
            valids = [1.0 if m else 0.0 for m in mac_done]
            avail = env.get_avail_actions()
            avail_t = [torch.FloatTensor(a).view(1, -1) for a in avail]
            if done:
                break

        steps = len(actions_hist)
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
                relevant_agents = list(range(n_agent))
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

        linked = link_windows(stall.triggered, t1_decisions)
        v_events = window_events(linked, episode_idx, episode_seed)
        all_variant_events.extend(v_events)
        all_stall_failures.extend(stall.triggered)
        all_linked.extend(linked)

        obs_max = max(o.size for frame in obs_hist for o in frame)
        obs_arr = np.zeros((steps, n_agent, obs_max), dtype=np.float32)
        for t, frame in enumerate(obs_hist):
            for a, o in enumerate(frame):
                obs_arr[t, a, :o.size] = o
        max_act = max(env.unwrapped.n_action)
        logits_pad = np.zeros((steps, n_agent, max_act), dtype=np.float32)
        for t, l in enumerate(logits_hist):
            for a in range(n_agent):
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
            "human_served": np.array([[1 if h in s["after"]["finish_serve_humans"] else 0
                                       for h in range(env.unwrapped.n_human)]
                                      for s in state_hist], dtype=np.int8),
            "finish_count": np.array([s["after"]["finish_count"] for s in state_hist], dtype=np.int16),
            "stall_active": np.array([1 if x is not None else 0 for x in stall_hist], dtype=np.int8),
        }
        stem = episodes_dir / f"episode_{episode_idx:05d}"
        np.savez_compressed(str(stem) + ".npz", **arrays)
        human_finish_steps = []
        for h in range(env.unwrapped.n_human):
            hits = np.nonzero(arrays["human_served"][:, h] > 0)[0]
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
            "probe_events": v_events,
            "variant_events": v_events,
            "stall_intervals": [list(x) for x in intervals],
            "stall_enabled": not args.stall_control,
            "stall_failures": list(stall.triggered),
            "window_links": linked,
            "t1_decisions": t1_decisions,
            "event_counts": {name: sum(1 for e in events if e["type"] == name)
                             for name in ("handoff_attempt_start", "handoff_failure", "handoff_success",
                                          "tool_wait_start", "tool_wait_timeout")},
            "stall_failure_count": len(stall.triggered),
            "window_count": sum(1 for e in v_events if e["window"]),
            "final_human_finished": [bool(h.whole_task_finished) for h in env.unwrapped.humans],
            "final_finish_count": len(env.unwrapped.finish_serve_humans),
            "label": args.label,
            "checkpoint_dir": str(args.checkpoint_dir),
            "hooks": args.hooks,
            "training_seed": args.seed,
            "eval_seed": args.eval_seed,
            "stall_control": bool(args.stall_control),
            "flex_sha": args.flex_sha,
            "arrays": str(stem) + ".npz",
            "array_sha256": base.sha256_file(str(stem) + ".npz"),
        }
        base.write_json(stem.with_suffix(".json"), summary)
        episode_summaries.append(summary)
        env = base.build_eval_env(base_config)

    hooks.close()
    with open(out_dir / "variant_events.jsonl", "w", encoding="utf-8") as handle:
        for event in all_variant_events:
            handle.write(json.dumps(base.json_safe(event), sort_keys=True) + "\n")
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
        "event_counts": {name: sum(e["event_counts"].get(name, 0) for e in episode_summaries)
                         for name in ("handoff_attempt_start", "handoff_failure", "handoff_success",
                                      "tool_wait_start", "tool_wait_timeout")},
        "stall_failure_count": len(all_stall_failures),
        "window_count": sum(1 for e in all_variant_events if e["window"]),
        "window_coverage": (sum(1 for e in all_variant_events if e["window"])
                            / len(all_stall_failures)) if all_stall_failures else None,
        "probe_event_count": len(all_variant_events),
        "stall_control": bool(args.stall_control),
        "episodes": episode_summaries,
        "probe_events": all_variant_events,
        "variant_events": all_variant_events,
        "stall_failures": all_stall_failures,
        "window_links": all_linked,
        "hooks": args.hooks,
        "flex_sha": args.flex_sha,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    base.write_json(out_dir / "summary.json", summary)
    print(json.dumps(base.json_safe({
        "episodes_completed": summary["episodes_completed"],
        "successes": summary["successes"],
        "success_rate": summary["success_rate"],
        "mean_steps": summary["mean_steps"],
        "stall_failure_count": summary["stall_failure_count"],
        "window_count": summary["window_count"],
        "window_coverage": summary["window_coverage"],
    }), indent=2))
    env.close()


if __name__ == "__main__":
    main()
