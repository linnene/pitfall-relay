#!/usr/bin/env python3
"""Small-scale VMAS navigation collection and local-rule evaluation.

Purpose (pre-registered in the final report): check whether the *original*
VMAS navigation scenario yields collision-penalty events, near misses, and
normal avoidance trajectories on CPU, whether the post-action observation and
the per-agent ``agent_collisions`` info align, and how far a simple local rule
(lidar + velocity only) can recover the label.

Constraints: CPU only, <= 16384 environment steps in total (num_envs * steps,
trial included), <= 15 min job wall clock, no training, no changes to upstream
scenarios, no package installs.

Raw arrays are written next to this script's --out-dir on the experiment host.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import vmas

PAIR_SWAPS = [(0, 1), (2, 3)]
AVOID_AGENTS = {2, 3}
LIDAR_OFFSET = 6  # obs = [pos(2), vel(2), goal_rel(2), lidar(12)]
N_RAYS = 12
LIDAR_MAX_RANGE = 0.35
AGENT_RADIUS = 0.1


def build_env(seed: int, num_envs: int, n_agents: int):
    return vmas.make_env(
        scenario="navigation",
        num_envs=num_envs,
        device="cpu",
        continuous_actions=True,
        max_steps=None,
        seed=seed,
        n_agents=n_agents,
        collisions=True,
        agent_collision_penalty=-1.0,
    )


def swap_goals(env, crossing=False):
    """Swap goals inside each pair so the two agents head for each other.

    Runtime intervention only; upstream scenario code is untouched. With
    ``crossing=True`` each goal is set to the partner's spawn position, which
    guarantees a head-on crossing instead of relying on where the random goal
    sampler happened to place the partner's goal.
    """
    agents = env.scenario.world.agents
    for i, j in PAIR_SWAPS:
        if crossing:
            target_i = agents[j].state.pos.clone()  # i heads to j's spawn
            target_j = agents[i].state.pos.clone()  # j heads to i's spawn
        else:
            target_i = agents[j].goal.state.pos.clone()
            target_j = agents[i].goal.state.pos.clone()
        agents[i].goal.set_pos(target_i, None)
        agents[j].goal.set_pos(target_j, None)
    for agent in agents:
        agent.pos_shaping = (
            torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=-1)
            * env.scenario.pos_shaping_factor
        )


def policy(env, target_speed=0.18, gain=8.0, repel_gain=2.0, repel_range=0.45):
    """Scripted controller: goal seeking, plus repulsion for AVOID_AGENTS.

    Uses ground-truth positions (privileged on purpose: this is the data
    generator, not the detector).
    """
    agents = env.scenario.world.agents
    pos = [a.state.pos for a in agents]
    vel = [a.state.vel for a in agents]
    acts = []
    for i, agent in enumerate(agents):
        delta = agent.goal.state.pos - pos[i]
        norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v_des = delta / norm * target_speed
        u = (v_des - vel[i]) * gain
        if i in AVOID_AGENTS:
            for j in range(len(agents)):
                if j == i:
                    continue
                rel = pos[i] - pos[j]
                dist = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                push = (repel_range - dist).clamp_min(0.0) * repel_gain
                u = u + rel / dist * push
        acts.append(u.clamp(-1.0, 1.0))
    return acts


def collect(seed, num_envs, n_agents, episodes, episode_len, out_path, crossing=False):
    env = build_env(seed, num_envs, n_agents)
    records = {k: [] for k in (
        "obs", "action", "reward", "collision", "pos", "vel", "rot", "goal", "gap", "done"
    )}
    steps_used = 0
    t0 = time.time()
    for episode in range(episodes):
        env.reset(seed=seed * 1000 + episode)
        swap_goals(env, crossing=crossing)
        for _ in range(episode_len):
            acts = policy(env)
            stacked = torch.stack([a.detach() for a in acts], dim=1)  # [B, A, 2]
            obs, rews, dones, infos = env.step(acts)
            steps_used += num_envs

            agents = env.scenario.world.agents
            pos = torch.stack([a.state.pos for a in agents], dim=1)
            vel = torch.stack([a.state.vel for a in agents], dim=1)
            rot = torch.stack([a.state.rot for a in agents], dim=1)
            goal = torch.stack([a.goal.state.pos for a in agents], dim=1)
            # center distance minus two radii = surface gap (negative = overlap)
            dist = torch.cdist(pos, pos)
            gap = dist - 2 * AGENT_RADIUS
            eye = torch.eye(n_agents, dtype=torch.bool).unsqueeze(0)
            gap = gap.masked_fill(eye, float("inf"))
            min_gap = gap.min(dim=-1).values

            records["obs"].append(torch.stack(obs, dim=1).numpy())
            records["action"].append(stacked.numpy())
            records["reward"].append(torch.stack(rews, dim=1).numpy())
            records["collision"].append(
                torch.stack([infos[i]["agent_collisions"] for i in range(n_agents)], dim=1).numpy()
            )
            records["pos"].append(pos.numpy())
            records["vel"].append(vel.numpy())
            records["rot"].append(rot.numpy())
            records["goal"].append(goal.numpy())
            records["gap"].append(min_gap.numpy())
            records["done"].append(dones.numpy())
    elapsed = time.time() - t0
    meta = {
        "seed": seed,
        "num_envs": num_envs,
        "n_agents": n_agents,
        "episodes": episodes,
        "episode_len": episode_len,
        "env_steps": steps_used,
        "wall_clock_s": round(elapsed, 2),
        "vmas_version": vmas.__version__,
        "torch_version": torch.__version__,
    }
    np.savez_compressed(
        out_path, **{k: np.stack(v).astype(np.float32) for k, v in records.items()}, meta=json.dumps(meta)
    )
    return meta


def load(path):
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files if k != "meta"}, json.loads(str(data["meta"]))


def labels(obs, collision, gap):
    """Per step/agent labels from privileged information only."""
    coll = collision < 0
    near = (~coll) & (gap > 0.005) & (gap <= 0.05)
    normal = gap > 0.15
    return {"collision": coll, "near": near, "normal": normal}


def rule_fire(obs, threshold, require_approach):
    """Local rule. Uses only the agent's own post-action observation.

    obs = [pos(2), vel(2), goal_rel(2), lidar(12)]; lidar channels are
    0.35 - surface_distance, so a larger value means a nearby agent. Ray j
    points at world angle 2*pi*j/12 (agent rotation assumed ~0, checked in the
    trial). The approach variant uses the agent's own velocity from obs.
    """
    lidar = obs[..., LIDAR_OFFSET:LIDAR_OFFSET + N_RAYS]
    closeness = lidar.max(axis=-1)
    fire = closeness >= threshold
    if require_approach:
        idx = lidar.argmax(axis=-1)
        angle = idx * (2 * np.pi / N_RAYS)
        direction = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        speed = obs[..., 2:4]
        fire = fire & ((direction * speed).sum(-1) > 0)
    return fire


def event_spans(flags):
    """Merge contiguous True steps into (env, agent, start, end) spans.

    Arrays are laid out [time, num_envs, n_agents]; the time axis is 0.
    """
    T, B, A = flags.shape
    spans = []
    for e in range(B):
        for a in range(A):
            t = 0
            while t < T:
                if not flags[t, e, a]:
                    t += 1
                    continue
                start = t
                while t + 1 < T and flags[t + 1, e, a]:
                    t += 1
                spans.append((e, a, start, t))
                t += 1
    return spans


def event_metrics(flags, fire):
    spans = event_spans(flags)
    if not spans:
        return {"events": 0, "detected": 0, "recall": None, "latency": []}
    detected = 0
    latencies = []
    for e, a, s, en in spans:
        hit = np.flatnonzero(fire[s:en + 1, e, a])
        if len(hit):
            detected += 1
            latencies.append(int(hit[0]))  # 0 = same step as first penalty
    return {
        "events": len(spans),
        "detected": detected,
        "recall": detected / len(spans),
        "latency": latencies,
    }


def step_metrics(fire, lab):
    coll_steps = lab["collision"]
    out = {}
    tp = (fire & coll_steps).sum()
    fp = (fire & ~coll_steps).sum()
    out["step_precision"] = float(tp / max(tp + fp, 1))
    out["step_recall"] = float(tp / max(coll_steps.sum(), 1))
    out["near_false_rate"] = float((fire & lab["near"]).sum() / max(lab["near"].sum(), 1))
    out["normal_false_rate"] = float((fire & lab["normal"]).sum() / max(lab["normal"].sum(), 1))
    out["fires"] = int(fire.sum())
    out["steps"] = int(fire.size)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seeds", default="101,202")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--episode-len", type=int, default=128)
    p.add_argument("--trial", action="store_true")
    p.add_argument("--eval-only", action="store_true",
                   help="skip collection; evaluate existing npz files (no new env steps)")
    p.add_argument("--crossing-goals", action="store_true",
                   help="set each agent's goal to its partner's spawn position")
    p.add_argument("--budget", type=int, default=16384)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    episodes = 1 if args.trial else args.episodes
    episode_len = 48 if args.trial else args.episode_len
    planned = len(seeds) * args.num_envs * episodes * episode_len
    if not args.eval_only:
        assert planned <= args.budget, f"planned {planned} > budget {args.budget}"

    metas = []
    if not args.eval_only:
        for seed in seeds:
            path = os.path.join(args.out_dir, f"nav_seed{seed}.npz")
            meta = collect(seed, args.num_envs, args.n_agents, episodes, episode_len, path,
                           crossing=args.crossing_goals)
            meta["path"] = path
            metas.append(meta)
            print(json.dumps({"collection": meta}, ensure_ascii=False))

    if args.trial:
        return

    # Load per seed; dev = seed 101 envs 0-3, held-out = seed 101 envs 4-7
    # and every env of seed 202.
    data = {}
    for seed in seeds:
        path = os.path.join(args.out_dir, f"nav_seed{seed}.npz")
        data[seed], _ = load(path)

    dev = {k: v[:, 0:4] for k, v in data[seeds[0]].items() if k != "done"}
    heldout_traj = {k: v[:, 4:8] for k, v in data[seeds[0]].items() if k != "done"}
    heldout_layout = {k: v for k, v in data[seeds[1]].items() if k != "done"}

    def lab_for(d):
        return labels(d["obs"], d["collision"], d["gap"])

    dev_lab = lab_for(dev)
    print(json.dumps({"dev_counts": {
        "collision_steps": int(dev_lab["collision"].sum()),
        "near_steps": int(dev_lab["near"].sum()),
        "normal_steps": int(dev_lab["normal"].sum()),
        "agent_steps": int(dev_lab["collision"].size),
    }}, ensure_ascii=False))

    grid = np.round(np.arange(0.05, 0.301, 0.025), 3)
    results = []
    for theta in grid:
        fire = rule_fire(dev["obs"], float(theta), require_approach=False)
        em = event_metrics(dev_lab["collision"], fire)
        sm = step_metrics(fire, dev_lab)
        score = (em["recall"] or 0) - sm["normal_false_rate"] - 0.5 * sm["near_false_rate"]
        results.append({"theta": float(theta), "event_recall": em["recall"],
                        "events": em["events"], "normal_false": sm["normal_false_rate"],
                        "near_false": sm["near_false_rate"], "score": round(score, 4)})
    best = max(results, key=lambda r: r["score"])
    print(json.dumps({"threshold_scan": sorted(results, key=lambda r: -r["score"])[:5],
                      "chosen_theta": best["theta"]}, ensure_ascii=False))

    theta = best["theta"]
    report = {"chosen_theta": theta, "dev_scan_top": sorted(results, key=lambda r: -r["score"])[:5],
              "splits": {}, "counts": {}}
    for name, d in [("dev", dev), ("heldout_traj", heldout_traj), ("heldout_layout", heldout_layout)]:
        lb = lab_for(d)
        report["counts"][name] = {
            "collision_steps": int(lb["collision"].sum()),
            "near_steps": int(lb["near"].sum()),
            "normal_steps": int(lb["normal"].sum()),
            "agent_steps": int(lb["collision"].size),
        }
        for rname, req in [("R1_lidar", False), ("R2_lidar_approach", True)]:
            fire = rule_fire(d["obs"], theta, require_approach=req)
            em = event_metrics(lb["collision"], fire)
            sm = step_metrics(fire, lb)
            report["splits"].setdefault(name, {})[rname] = {"event": em, "step": sm}

    # One step-aligned example: prefer an event that starts after the initial
    # spawn phase, since spawn-adjacent contacts are less informative.
    example = None
    for name, d in [("dev", dev), ("heldout_traj", heldout_traj), ("heldout_layout", heldout_layout)]:
        lb = lab_for(d)
        spans = event_spans(lb["collision"])
        if not spans:
            continue
        mid = [sp for sp in spans if sp[2] > 10]
        e, a, s, en = (mid[0] if mid else max(spans, key=lambda sp: sp[3] - sp[2]))
        lo, hi = max(0, s - 2), min(d["obs"].shape[0], en + 3)
        rows = []
        for t in range(lo, hi):
            rows.append({
                "t": t,
                "action": [round(float(x), 3) for x in d["action"][t, e, a]],
                "lidar_max_closeness": round(float(d["obs"][t, e, a, LIDAR_OFFSET:LIDAR_OFFSET + N_RAYS].max()), 3),
                "agent_collisions": round(float(d["collision"][t, e, a]), 3),
                "min_gap": round(float(d["gap"][t, e, a]), 4),
                "label": "collision" if lb["collision"][t, e, a] else ("near" if lb["near"][t, e, a] else "normal"),
            })
        example = {"split": name, "env": int(e), "agent": int(a), "event_span": [int(s), int(en)], "rows": rows}
        break
    report["alignment_example"] = example
    report["pairs"] = {"aggressive": [0, 1], "avoidant": [2, 3]}

    print(json.dumps({"report": report}, ensure_ascii=False))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps({"json_out": args.json_out}))


if __name__ == "__main__":
    main()
