#!/usr/bin/env python3
"""Multi-failure collection on stock VMAS 1.5.2 scenarios (CPU only).

Scenarios and failure mechanisms:
  balance   object drop        ground_rew  (-10, team-shared)
  give_way  contact            agent_collision_rew (teammate), obstacle_collision_rew (passage/obstacle)
  football  task outcome       sparse_reward (-100 team when conceding)

Everything is stock upstream behaviour: only documented make_env kwargs are used.
No training, no world model, no upstream edit. Raw trajectories stay on the host.

Array layout is [time, num_envs, n_agents, ...]; time is axis 0.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import vmas

# ---------------------------------------------------------------- shared utils


def make(scenario, num_envs, seed, **kw):
    return vmas.make_env(
        scenario=scenario,
        num_envs=num_envs,
        device="cpu",
        continuous_actions=True,
        seed=seed,
        **kw,
    )


def episode_ids(dones):
    """Per-step episode id from auto/manual resets. dones: [T, B] bool."""
    T, B = dones.shape
    eid = np.zeros((T, B), dtype=np.int32)
    counter = np.zeros(B, dtype=np.int32)
    prev_done = np.zeros(B, dtype=bool)
    for t in range(T):
        counter = counter + prev_done
        eid[t] = counter
        prev_done = dones[t]
    return eid


def spans(flags):
    """Contiguous True runs per column of a 2-D [T, N] matrix."""
    out = []
    T, N = flags.shape
    for i in range(N):
        t = 0
        while t < T:
            if not flags[t, i]:
                t += 1
                continue
            s = t
            while t + 1 < T and flags[t + 1, i]:
                t += 1
            out.append((i, s, t))
            t += 1
    return out


def rule_metrics(flags, fire, near, normal):
    """flags/fire/near/normal: [T, ...]; columns are agents or team-slots."""
    flags = flags.reshape(flags.shape[0], -1)
    fire = fire.reshape(fire.shape[0], -1)
    near = near.reshape(near.shape[0], -1)
    normal = normal.reshape(normal.shape[0], -1)
    sp = spans(flags)
    det = 0
    lat = []
    for i, s, e in sp:
        hit = np.flatnonzero(fire[s:e + 1, i])
        if len(hit):
            det += 1
            lat.append(int(hit[0]))
    tp = int((fire & flags).sum())
    fp = int((fire & ~flags).sum())
    return {
        "events": len(sp),
        "detected": det,
        "recall": (det / len(sp)) if sp else None,
        "latency": lat,
        "step_precision": tp / max(tp + fp, 1),
        "step_recall": tp / max(int(flags.sum()), 1),
        "near_false": float((fire & near).sum()) / max(int(near.sum()), 1),
        "normal_false": float((fire & normal).sum()) / max(int(normal.sum()), 1),
        "fires": int(fire.sum()),
    }


def fire_rule(values, theta, direction):
    return values >= theta if direction == ">=" else values <= theta


def scan_threshold(values, flags, near, normal, grid, direction):
    best = None
    rows = []
    for th in grid:
        fire = fire_rule(values, th, direction)
        m = rule_metrics(flags, fire, near, normal)
        score = (m["recall"] or 0) - m["normal_false"] - 0.5 * m["near_false"]
        row = {"theta": float(th), "score": round(score, 4), **{k: m[k] for k in
               ("events", "recall", "near_false", "normal_false")}}
        rows.append(row)
        if best is None or score > best[0]:
            best = (score, th)
    rows.sort(key=lambda r: -r["score"])
    return best[1], rows[:5]


def save(path, arrays, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


# ------------------------------------------------------------------- balance


def collect_balance(seed, num_envs, episodes, ep_len, path):
    env = make("balance", num_envs, seed)
    world = env.scenario.world
    agents = world.agents
    pkg, line, floor = env.scenario.package, env.scenario.line, env.scenario.floor
    rec = {k: [] for k in ("obs", "action", "reward", "ground_rew", "pos_rew",
                           "done", "pkg_floor_gap", "line_floor_gap", "mode")}
    t0 = time.time()
    for ep in range(episodes):
        mode = ["hold", "weak", "drop"][ep % 3]
        env.reset(seed=seed * 1000 + ep)
        for _ in range(ep_len):
            if mode == "drop":
                acts = [torch.zeros(num_envs, 2) for _ in agents]
            elif mode == "weak":
                acts = [torch.tensor([0.0, 0.2]).repeat(num_envs, 1) for _ in agents]
            else:
                acts = [torch.tensor([0.0, 1.0]).repeat(num_envs, 1) for _ in agents]
            obs, rews, dones, infos = env.step(acts)
            rec["obs"].append(torch.stack(obs, 1).numpy())
            rec["action"].append(torch.stack(acts, 1).numpy())
            rec["reward"].append(torch.stack(rews, 1).numpy())
            rec["ground_rew"].append(torch.stack([infos[i]["ground_rew"] for i in range(len(agents))], 1).numpy())
            rec["pos_rew"].append(torch.stack([infos[i]["pos_rew"] for i in range(len(agents))], 1).numpy())
            rec["done"].append(dones.numpy())
            rec["pkg_floor_gap"].append(env.scenario.world.get_distance(pkg, floor).numpy())
            rec["line_floor_gap"].append(env.scenario.world.get_distance(line, floor).numpy())
            rec["mode"].append(np.full(num_envs, {"hold": 0, "weak": 1, "drop": 2}[mode], dtype=np.int8))
    meta = {"scenario": "balance", "seed": seed, "num_envs": num_envs,
            "episodes": episodes, "episode_len": ep_len,
            "env_steps": num_envs * episodes * ep_len,
            "wall_clock_s": round(time.time() - t0, 2)}
    save(path, {k: np.stack(v) for k, v in rec.items()}, meta)
    return meta


# ------------------------------------------------------------------ give_way


def collect_give_way(seed, num_envs, episodes, ep_len, path):
    env = make(
        "give_way", num_envs, seed,
        observe_rel_pos=True,
        agent_collision_penalty=-1.0,
        passage_collision_penalty=-2.0,
        obstacle_collision_penalty=-1.0,
    )
    world = env.scenario.world
    agents = world.agents
    rec = {k: [] for k in ("obs", "action", "reward", "agent_collision", "obstacle_collision",
                           "pos_rew", "final_rew", "done", "mode", "goal_dist",
                           "agent_gap", "wall_gap")}
    t0 = time.time()
    for ep in range(episodes):
        mode = ["bump", "straight", "yield", "wall"][ep % 4]
        env.reset(seed=seed * 1000 + ep)
        for step in range(ep_len):
            acts = []
            for i, ag in enumerate(agents):
                other = agents[1 - i]
                if mode == "bump":
                    goal = other.state.pos
                elif mode == "yield" and i == 0 and step < 24:
                    goal = ag.state.pos
                elif mode == "wall" and i == 0:
                    goal = ag.state.pos + torch.tensor([0.0, 1.0]).repeat(num_envs, 1)
                else:
                    goal = ag.goal.state.pos
                vec = goal - ag.state.pos
                norm = vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                acts.append((vec / norm * 0.5).clamp(-0.5, 0.5))
            obs, rews, dones, infos = env.step(acts)
            rec["obs"].append(torch.stack(obs, 1).numpy())
            rec["action"].append(torch.stack(acts, 1).numpy())
            rec["reward"].append(torch.stack(rews, 1).numpy())
            rec["agent_collision"].append(torch.stack([infos[i]["agent_collision_rew"] for i in range(len(agents))], 1).numpy())
            rec["obstacle_collision"].append(torch.stack([infos[i]["obstacle_collision_rew"] for i in range(len(agents))], 1).numpy())
            rec["pos_rew"].append(torch.stack([infos[i]["pos_rew"] for i in range(len(agents))], 1).numpy())
            rec["final_rew"].append(torch.stack([infos[i]["final_rew"] for i in range(len(agents))], 1).numpy())
            rec["done"].append(dones.numpy())
            rec["mode"].append(np.full(num_envs, ["bump", "straight", "yield", "wall"].index(mode), dtype=np.int8))
            p0, p1 = agents[0].state.pos, agents[1].state.pos
            rec["agent_gap"].append((p0 - p1).norm(dim=-1).numpy())
            wall = torch.stack(
                [world.get_distance(ag, lm) for ag in agents for lm in world.landmarks]
            ).min(dim=0).values
            rec["wall_gap"].append(wall.numpy())
            rec["goal_dist"].append(torch.stack([(ag.goal.state.pos - ag.state.pos).norm(dim=-1) for ag in agents], 1).numpy())
    meta = {"scenario": "give_way", "seed": seed, "num_envs": num_envs,
            "episodes": episodes, "episode_len": ep_len,
            "env_steps": num_envs * episodes * ep_len,
            "penalties": {"agent": -1.0, "passage": -2.0, "obstacle": -1.0},
            "observe_rel_pos": True,
            "wall_clock_s": round(time.time() - t0, 2)}
    save(path, {k: np.stack(v) for k, v in rec.items()}, meta)
    return meta


# ------------------------------------------------------------------ football


def collect_football(seed, num_envs, episodes, ep_len, path):
    env = make("football", num_envs, seed, n_blue_agents=3, n_red_agents=3)
    # env.agents are the three blue agents; red agents are scripted inside the
    # scenario and the ball is a movable Agent entity, so neither takes actions.
    agents = env.agents
    ball = env.scenario.ball
    rec = {k: [] for k in ("obs", "action", "reward", "sparse", "dense_ball_goal",
                           "dense_agent_ball", "done", "mode", "ball_pos", "ball_vel")}
    t0 = time.time()
    for ep in range(episodes):
        mode = ["passive", "chase", "defend"][ep % 3]
        env.reset(seed=seed * 1000 + ep)
        for _ in range(ep_len):
            acts = []
            for ag in agents:
                if mode == "passive":
                    acts.append(torch.zeros(num_envs, 2))
                elif mode == "chase":
                    vec = ball.state.pos - ag.state.pos
                    acts.append(vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                else:  # defend: stand between ball and own goal (x = -pitch/2)
                    own = torch.tensor([-env.scenario.pitch_length / 2, 0.0]).repeat(num_envs, 1)
                    vec = 0.5 * (ball.state.pos + own) - ag.state.pos
                    acts.append(vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            obs, rews, dones, infos = env.step(acts)
            rec["obs"].append(torch.stack(obs, 1).numpy())
            rec["action"].append(torch.stack(acts, 1).numpy())
            rec["reward"].append(torch.stack(rews, 1).numpy())
            rec["sparse"].append(torch.stack([infos[i]["sparse_reward"] for i in range(len(agents))], 1).numpy())
            rec["dense_ball_goal"].append(torch.stack([infos[i]["ball_goal_pos_rew"] for i in range(len(agents))], 1).numpy())
            rec["dense_agent_ball"].append(torch.stack([infos[i]["all_agent_ball_pos_rew"] for i in range(len(agents))], 1).numpy())
            rec["done"].append(dones.numpy())
            rec["mode"].append(np.full(num_envs, {"passive": 0, "chase": 1, "defend": 2}[mode], dtype=np.int8))
            rec["ball_pos"].append(ball.state.pos.numpy())
            rec["ball_vel"].append(ball.state.vel.numpy())
    meta = {"scenario": "football", "seed": seed, "num_envs": num_envs,
            "episodes": episodes, "episode_len": ep_len,
            "env_steps": num_envs * episodes * ep_len,
            "ai_red_agents": True, "n_blue": 3, "n_red": 3,
            "wall_clock_s": round(time.time() - t0, 2)}
    save(path, {k: np.stack(v) for k, v in rec.items()}, meta)
    return meta


# ------------------------------------------------------------------ analysis


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files if k != "meta"}, json.loads(str(d["meta"]))


def analyze_balance(root, seeds):
    out = {"scenario": "balance", "splits": {}, "threshold_scan": []}
    data = {s: load(f"{root}/balance_seed{s}.npz")[0] for s in seeds}
    dev = {k: v[:, 0:4] for k, v in data[seeds[0]].items()}
    ht = {k: v[:, 4:8] for k, v in data[seeds[0]].items()}
    hl = data[seeds[1]]
    # Team-level labels: drop = ground_rew < 0 for any agent (shared, count once).
    def lab(d):
        drop = (d["ground_rew"] < 0).any(-1)          # [T, B], team-shared
        gap = np.minimum(d["pkg_floor_gap"], d["line_floor_gap"])
        near = (~drop) & (gap < 0.03)
        normal = gap > 0.08
        return drop, near, normal
    def values(d, kind):
        if kind == "line_rot":
            return np.abs(d["obs"][..., 15]).max(-1)          # |rot| per env
        return np.maximum(0.0, -d["obs"][..., 11]).max(-1)    # falling package speed

    grid = np.round(np.arange(0.02, 0.401, 0.02), 3)
    d_lab = lab(dev)
    rules = {}
    for kind in ("line_rot", "pkg_fall_speed"):
        th, sc = scan_threshold(values(dev, kind), *d_lab, grid, ">=")
        rules[kind] = {"theta": float(th), "scan": sc}
    out["rules"] = rules
    out["theta"] = rules["line_rot"]["theta"]
    for name, d in (("dev", dev), ("heldout_traj", ht), ("heldout_layout", hl)):
        drop, near, normal = lab(d)
        counts = {
            "drop_steps": int(drop.sum()),
            "near_steps": int(near.sum()),
            "normal_steps": int(normal.sum()),
            "success_steps": int((d["done"].astype(bool) & ~drop).sum()),
            "team_steps": int(drop.size),
        }
        out["splits"][name] = {"counts": counts}
        for kind in rules:
            fire = fire_rule(values(d, kind), rules[kind]["theta"], ">=")
            out["splits"][name][f"R_{kind}"] = rule_metrics(drop, fire, near, normal)
        # Example: first drop step with 2 steps of context.
        idx = np.argwhere(drop)
        if len(idx) and "example" not in out:
            t, e = idx[0]
            lo, hi = max(0, t - 2), min(drop.shape[0], t + 3)
            out["example"] = {"split": name, "env": int(e), "rows": [{
                "t": int(tt),
                "line_rot": round(float(d["obs"][tt, e, 0, 15]), 3),
                "pkg_vel_y": round(float(d["obs"][tt, e, 0, 11]), 3),
                "pkg_floor_gap": round(float(d["pkg_floor_gap"][tt, e]), 4),
                "ground_rew": round(float(d["ground_rew"][tt, e, 0]), 2),
                "label": "drop" if drop[tt, e] else ("near" if near[tt, e] else "normal"),
            } for tt in range(lo, hi)]}
    return out


def analyze_give_way(root, seeds):
    out = {"scenario": "give_way", "splits": {}, "threshold_scan": []}
    data = {s: load(f"{root}/give_way_seed{s}.npz")[0] for s in seeds}
    half = data[seeds[0]]["obs"].shape[1] // 2
    dev = {k: v[:, :half] for k, v in data[seeds[0]].items()}
    ht = {k: v[:, half:] for k, v in data[seeds[0]].items()}
    hl = data[seeds[1]]

    def lab(d):
        mate = d["agent_collision"] < 0          # teammate contact
        obst = d["obstacle_collision"] < 0      # passage or other landmark contact
        gap = d["agent_gap"][..., None]
        near = (~mate) & (~obst) & (gap < 0.15)
        normal = (~mate) & (~obst) & (gap > 0.30)
        return mate, obst, near, normal

    def dist(d):
        return np.linalg.norm(d["obs"][..., 4:6], axis=-1)  # [T, B, A]

    d_lab = lab(dev)
    theta, scan = scan_threshold(dist(dev), d_lab[0], d_lab[2], d_lab[3],
                                 np.round(np.arange(0.02, 0.301, 0.02), 3), "<=")
    out["threshold_scan"] = scan
    out["theta"] = float(theta)
    for name, d in (("dev", dev), ("heldout_traj", ht), ("heldout_layout", hl)):
        mate, obst, near, normal = lab(d)
        fire = fire_rule(dist(d), theta, "<=")
        out["splits"][name] = {
            "counts": {"mate_contact_steps": int(mate.sum()), "obstacle_contact_steps": int(obst.sum()),
                       "near_steps": int(near.sum()), "normal_steps": int(normal.sum()),
                       "success_steps": int((d["final_rew"] > 0).sum()),
                       "agent_steps": int(mate.size)},
            "R1_rel_distance": rule_metrics(mate, fire, near, normal),
            "obstacle_events": rule_metrics(obst, fire, near, normal),
        }
        idx = np.argwhere(mate)
        if len(idx) and "example" not in out:
            t, e, a = idx[0]
            lo, hi = max(0, t - 2), min(mate.shape[0], t + 3)
            out["example"] = {"split": name, "env": int(e), "agent": int(a), "rows": [{
                "t": int(tt),
                "rel_distance": round(float(dist(d)[tt, e, a]), 3),
                "action": [round(float(x), 2) for x in d["action"][tt, e, a]],
                "agent_collision": round(float(d["agent_collision"][tt, e, a]), 2),
                "obstacle_collision": round(float(d["obstacle_collision"][tt, e, a]), 2),
                "label": "mate_contact" if mate[tt, e, a] else ("obstacle" if obst[tt, e, a] else ("near" if near[tt, e, a] else "normal")),
            } for tt in range(lo, hi)]}
    return out


def analyze_football(root, seeds):
    out = {"scenario": "football", "splits": {}, "threshold_scan": []}
    data = {s: load(f"{root}/football_seed{s}.npz")[0] for s in seeds}
    dev = {k: v[:, 0:4] for k, v in data[seeds[0]].items()}
    ht = {k: v[:, 4:8] for k, v in data[seeds[0]].items()}
    hl = data[seeds[1]]

    def lab(d):
        conceded = (d["sparse"] < 0).any(-1)        # [T, B], team-shared
        scored = (d["sparse"] > 0).any(-1)
        ball_x = d["ball_pos"][..., 0]
        near = (~conceded) & (~scored) & (ball_x < -1.5 + 0.5)
        normal = (~conceded) & (~scored) & (ball_x > -1.5 + 1.2)
        return conceded, scored, near, normal

    def risk(d):
        # distance of the ball from blue's own goal line at x = -1.5
        return d["ball_pos"][..., 0] + 1.5

    c, s, n, nm = lab(dev)
    theta, scan = scan_threshold(risk(dev), c, n, nm,
                                 np.round(np.arange(0.1, 1.01, 0.1), 2), "<=")
    out["threshold_scan"] = scan
    out["theta"] = float(theta)
    for name, d in (("dev", dev), ("heldout_traj", ht), ("heldout_layout", hl)):
        c, s, n, nm = lab(d)
        fire = fire_rule(risk(d), theta, "<=")
        out["splits"][name] = {
            "counts": {"conceded_steps": int(c.sum()), "scored_steps": int(s.sum()),
                       "near_steps": int(n.sum()), "normal_steps": int(nm.sum()),
                       "env_steps": int(c.size)},
            "R1_ball_to_own_goal": rule_metrics(c, fire, n, nm),
        }
        idx = np.argwhere(c)
        if len(idx) and "example" not in out:
            t, e = idx[0]
            lo, hi = max(0, t - 2), min(c.shape[0], t + 3)
            out["example"] = {"split": name, "env": int(e), "rows": [{
                "t": int(tt),
                "ball_x": round(float(d["ball_pos"][tt, e, 0]), 3),
                "ball_vx": round(float(d["ball_vel"][tt, e, 0]), 3),
                "sparse_reward": round(float(d["sparse"][tt, e, 0]), 1),
                "dense_ball_goal": round(float(d["dense_ball_goal"][tt, e, 0]), 3),
                "label": "conceded" if c[tt, e] else ("near" if n[tt, e] else "normal" if nm[tt, e] else "unclassified"),
            } for tt in range(lo, hi)]}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--seeds", default="301,302")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--episode-len", type=int, default=128)
    p.add_argument("--scenario", choices=["balance", "give_way", "football", "all"], default="all")
    p.add_argument("--trial", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--episode-len-map", default="")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    ep_len_map = {"balance": args.episode_len, "give_way": args.episode_len, "football": args.episode_len}
    for item in args.episode_len_map.split(","):
        if item:
            k, v = item.split("=")
            ep_len_map[k] = int(v)
    episodes = 3 if args.trial else args.episodes
    if args.trial:
        ep_len_map = {k: 48 for k in ep_len_map}

    collectors = {"balance": collect_balance, "give_way": collect_give_way, "football": collect_football}
    todo = list(collectors) if args.scenario == "all" else [args.scenario]

    metas = []
    if not args.eval_only:
        for name in todo:
            for seed in seeds:
                path = f"{args.root}/{name}_seed{seed}.npz"
                meta = collectors[name](seed, args.num_envs, episodes, ep_len_map[name], path)
                meta["path"] = path
                metas.append(meta)
                print(json.dumps({"collection": meta}, ensure_ascii=False))
    if args.trial:
        return

    analyzers = {"balance": analyze_balance, "give_way": analyze_give_way, "football": analyze_football}
    report = {name: analyzers[name](args.root, seeds) for name in todo}
    print(json.dumps({"report": report}, ensure_ascii=False, default=float))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=float)


if __name__ == "__main__":
    main()
