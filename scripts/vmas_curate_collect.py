#!/usr/bin/env python3
"""Targeted success-collection for the curation task (CPU, stock VMAS 1.5.2).

balance push_goal : agents push the line/package toward the goal while holding
                    it up, so the package can actually be delivered.
give_way escort   : green goes first; blue waits until green is at its goal,
                    then goes. Goal distances are read from scenario state only
                    to script the data-collecting behaviour.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import vmas


def save(path, arrays, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


def collect_balance(seed, num_envs, episodes, ep_len, path):
    env = vmas.make_env("balance", num_envs=num_envs, device="cpu",
                        continuous_actions=True, seed=seed)
    sc = env.scenario
    rec = {k: [] for k in ("obs", "action", "reward", "ground_rew", "pos_rew",
                           "done", "pkg_floor_gap", "line_floor_gap")}
    t0 = time.time()
    for ep in range(episodes):
        env.reset(seed=seed * 1000 + ep)
        for _ in range(ep_len):
            acts = []
            for ag in sc.world.agents:
                obs = sc.observation(ag)  # [B, 16]
                dx = obs[:, 8]
                ux = torch.clamp(0.5 * torch.sign(dx), -0.5, 0.5)
                ux = torch.where(dx.abs() < 0.02, torch.zeros_like(ux), ux)
                acts.append(torch.stack([ux, torch.ones_like(ux)], dim=1))
            obs, rews, dones, infos = env.step(acts)
            n = len(sc.world.agents)
            rec["obs"].append(torch.stack(obs, 1).numpy())
            rec["action"].append(torch.stack(acts, 1).numpy())
            rec["reward"].append(torch.stack(rews, 1).numpy())
            rec["ground_rew"].append(torch.stack([infos[i]["ground_rew"] for i in range(n)], 1).numpy())
            rec["pos_rew"].append(torch.stack([infos[i]["pos_rew"] for i in range(n)], 1).numpy())
            rec["done"].append(dones.numpy())
            rec["pkg_floor_gap"].append(sc.world.get_distance(sc.package, sc.floor).numpy())
            rec["line_floor_gap"].append(sc.world.get_distance(sc.line, sc.floor).numpy())
    meta = {"scenario": "balance", "mode": "push_goal", "seed": seed,
            "num_envs": num_envs, "episodes": episodes, "episode_len": ep_len,
            "env_steps": num_envs * episodes * ep_len,
            "wall_clock_s": round(time.time() - t0, 2)}
    save(path, {k: np.stack(v) for k, v in rec.items()}, meta)
    return meta


def collect_give_way(seed, num_envs, episodes, ep_len, path):
    env = vmas.make_env("give_way", num_envs=num_envs, device="cpu",
                        continuous_actions=True, seed=seed,
                        observe_rel_pos=True,
                        agent_collision_penalty=-1.0,
                        passage_collision_penalty=-2.0,
                        obstacle_collision_penalty=-1.0)
    sc = env.scenario
    agents = sc.world.agents
    rec = {k: [] for k in ("obs", "action", "reward", "agent_collision", "obstacle_collision",
                           "pos_rew", "final_rew", "done", "goal_dist",
                           "agent_gap", "wall_gap", "green_to_goal")}
    t0 = time.time()
    for ep in range(episodes):
        env.reset(seed=seed * 1000 + ep)
        for step in range(ep_len):
            green_goal_dist = (agents[1].goal.state.pos - agents[1].state.pos).norm(dim=-1)
            # time-based release: green needs ~150 steps to cross; blue starts
            # only after green has had time to clear the corridor.
            blue_release = torch.full((num_envs,), step >= 170, dtype=torch.bool)
            acts = []
            for i, ag in enumerate(agents):
                vec = ag.goal.state.pos - ag.state.pos
                cmd = (vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 0.5).clamp(-0.5, 0.5)
                if i == 0:
                    # waypoint through the corridor centre, then the goal
                    wp = torch.zeros(num_envs, 2)
                    to_wp = wp - ag.state.pos
                    wp_dist = to_wp.norm(dim=-1, keepdim=True)
                    target = torch.where(wp_dist > 0.30, wp, ag.goal.state.pos)
                    vec = target - ag.state.pos
                    cmd = (vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 0.5).clamp(-0.5, 0.5)
                    cmd = torch.where(blue_release.unsqueeze(-1), cmd, torch.zeros_like(cmd))
                acts.append(cmd)
            obs, rews, dones, infos = env.step(acts)
            rec["obs"].append(torch.stack(obs, 1).numpy())
            rec["action"].append(torch.stack(acts, 1).numpy())
            rec["reward"].append(torch.stack(rews, 1).numpy())
            rec["agent_collision"].append(torch.stack([infos[i]["agent_collision_rew"] for i in range(2)], 1).numpy())
            rec["obstacle_collision"].append(torch.stack([infos[i]["obstacle_collision_rew"] for i in range(2)], 1).numpy())
            rec["pos_rew"].append(torch.stack([infos[i]["pos_rew"] for i in range(2)], 1).numpy())
            rec["final_rew"].append(torch.stack([infos[i]["final_rew"] for i in range(2)], 1).numpy())
            rec["done"].append(dones.numpy())
            rec["goal_dist"].append(torch.stack([(ag.goal.state.pos - ag.state.pos).norm(dim=-1) for ag in agents], 1).numpy())
            rec["agent_gap"].append((agents[0].state.pos - agents[1].state.pos).norm(dim=-1).numpy())
            rec["wall_gap"].append(torch.stack([sc.world.get_distance(ag, lm) for ag in agents for lm in sc.world.landmarks]).min(0).values.numpy())
            rec["green_to_goal"].append(green_goal_dist.numpy())
    meta = {"scenario": "give_way", "mode": "escort", "seed": seed,
            "num_envs": num_envs, "episodes": episodes, "episode_len": ep_len,
            "env_steps": num_envs * episodes * ep_len,
            "penalties": {"agent": -1.0, "passage": -2.0, "obstacle": -1.0},
            "wall_clock_s": round(time.time() - t0, 2)}
    save(path, {k: np.stack(v) for k, v in rec.items()}, meta)
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--scenario", choices=["balance", "give_way"], required=True)
    p.add_argument("--seeds", default="301,302")
    p.add_argument("--num-envs", type=int, default=6)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--episode-len", type=int, default=256)
    p.add_argument("--trial", action="store_true")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    episodes = 1 if args.trial else args.episodes
    ep_len = args.episode_len
    fn = collect_balance if args.scenario == "balance" else collect_give_way
    for seed in seeds:
        meta = fn(seed, args.num_envs, episodes, ep_len,
                  f"{args.root}/{args.scenario}_seed{seed}.npz")
        print(json.dumps({"collection": meta}))


if __name__ == "__main__":
    main()
