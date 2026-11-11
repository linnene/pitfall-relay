#!/usr/bin/env python3
"""True paired collection: identical reset + identical action prefix, then fork.

Mechanisms
  nav_cross : crossing contact. fail = collision penalty; success = all agents
              reach goals without any penalty.
  give_way  : teammate contact. fail = agent_collision_rew<0; success = both
              reach goals without contact (final_rew>0).
  football  : concede vs clearance. fail = sparse_reward<0; success = ball was
              inside the near-goal zone and left it (x>-0.5) without a goal.

Branches are run sequentially with the same reset seed and identical actions up
to `fork`. Pre-fork states are compared after the run; any mismatch is recorded.
"""
import argparse
import json
import os

import numpy as np
import torch
import vmas

PRE_STEPS = 0  # episodes are cut at the first done
SIGN = 1.0    # lateral detour direction for the navigation success branch


def save(path, arrays, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


# ----------------------------------------------------------------- navigation
NAV_SWAPS = [(0, 1), (2, 3)]


def nav_setup(env):
    agents = env.scenario.world.agents
    for i, j in NAV_SWAPS:
        gi = agents[i].goal.state.pos.clone()
        gj = agents[j].goal.state.pos.clone()
        agents[i].goal.set_pos(gj, None)
        agents[j].goal.set_pos(gi, None)
    for ag in agents:
        ag.pos_shaping = torch.linalg.vector_norm(
            ag.state.pos - ag.goal.state.pos, dim=-1) * env.scenario.pos_shaping_factor


def nav_policy(env, t, fork, branch):
    agents = env.scenario.world.agents
    acts = []
    for i, ag in enumerate(agents):
        delta = ag.goal.state.pos - ag.state.pos
        n = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v_des = delta / n * 0.18
        u = (v_des - ag.state.vel) * 8.0
        if branch == "success" and t >= fork and i == 0:
            delta = ag.goal.state.pos - ag.state.pos
            perp = torch.stack([-delta[:, 1], delta[:, 0]], dim=-1)
            perp = perp / perp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            # deterministic sign: steer away from the nearest other agent
            others = torch.stack([a.state.pos for j, a in enumerate(agents) if j != i], 1)
            d = (others - ag.state.pos.unsqueeze(1)).norm(dim=-1)
            nearest = others.gather(1, d.argmin(1).view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            side = ((nearest - ag.state.pos) * perp).sum(-1, keepdim=True)
            s = torch.where(side >= 0, -1.0, 1.0)
            u = u + s * perp * 0.5
        acts.append(u.clamp(-1.0, 1.0))
    return acts


# ------------------------------------------------------------------- give_way


def gw_setup(env):
    return None


def gw_policy(env, t, fork, branch):
    agents = env.scenario.world.agents
    acts = []
    for i, ag in enumerate(agents):
        target = ag.goal.state.pos
        if branch == "success" and t >= fork and i == 0:
            # wait at the corridor entrance while the other agent clears
            target = ag.state.pos.clone()
            if t >= fork + 60:
                target = ag.goal.state.pos
        vec = target - ag.state.pos
        acts.append((vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 0.5).clamp(-0.5, 0.5))
    return acts


# -------------------------------------------------------------------- football


def fb_setup(env):
    return None


def fb_policy(env, t, fork, branch):
    sc = env.scenario
    agents = sc.world.agents[: len(env.agents)]
    ball = sc.ball.state.pos
    own = torch.tensor([-sc.pitch_length / 2, 0.0])
    acts = []
    for i, ag in enumerate(agents):
        if i == 0:
            keeper = torch.stack(
                [torch.full_like(ball[:, 0], own[0] + 0.05), ball[:, 1]], dim=-1)
            target = keeper if (branch == "success" or t < fork) else own.expand_as(ball) + torch.tensor([-0.5, 0.0])
        else:
            target = ball
        vec = target - ag.state.pos
        acts.append((vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)))
    return acts


# ---------------------------------------------------------------------- runner


def run(scenario, seed, fork, branch, steps, out):
    if scenario == "nav_cross":
        env = vmas.make_env("navigation", num_envs=1, device="cpu",
                            continuous_actions=True, seed=seed, n_agents=4,
                            collisions=True, agent_collision_penalty=-1.0)
        setup, policy = nav_setup, nav_policy
        info_keys = ["agent_collisions"]
    elif scenario == "give_way":
        env = vmas.make_env("give_way", num_envs=1, device="cpu",
                            continuous_actions=True, seed=seed,
                            observe_rel_pos=True,
                            agent_collision_penalty=-1.0,
                            passage_collision_penalty=-2.0,
                            obstacle_collision_penalty=-1.0)
        setup, policy = gw_setup, gw_policy
        info_keys = ["agent_collision_rew", "obstacle_collision_rew", "final_rew"]
    elif scenario == "football":
        env = vmas.make_env("football", num_envs=1, device="cpu",
                            continuous_actions=True, seed=seed,
                            n_blue_agents=3, n_red_agents=3)
        setup, policy = fb_setup, fb_policy
        info_keys = ["sparse_reward"]
    else:
        raise ValueError(scenario)

    env.reset(seed=seed)
    setup(env)
    rec = {k: [] for k in ("obs", "action", "pos", "vel", "done", "reward")}
    for k in info_keys:
        rec[k] = []
    steps_done = 0
    for t in range(steps):
        acts = policy(env, t, fork, branch)
        obs, rews, dones, infos = env.step(acts)
        agents_pos = [a.state.pos[0].tolist() for a in env.scenario.world.agents[: len(env.agents)]]
        rec["obs"].append(np.stack([o[0].numpy() for o in obs]))
        rec["action"].append(np.stack([a[0].numpy() for a in acts]))
        rec["pos"].append(np.array(agents_pos))
        rec["vel"].append(np.stack([a.state.vel[0].numpy() for a in env.scenario.world.agents[: len(env.agents)]]))
        rec["done"].append(bool(dones[0]))
        rec["reward"].append(np.array([float(r[0]) for r in rews]))
        for k in info_keys:
            rec[k].append(np.array([float(infos[i][k][0]) for i in range(len(env.agents))]))
        steps_done += 1
        if bool(dones[0]):
            break
    meta = {"scenario": scenario, "seed": seed, "fork": fork, "branch": branch,
            "steps": steps_done, "vmas": vmas.__version__}
    save(out, {k: np.stack(v) for k, v in rec.items()}, meta)
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=["nav_cross", "give_way", "football"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--fork", type=int, required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--out", required=True)
    p.add_argument("--branches", default="fail,success")
    p.add_argument("--sign", type=float, default=1.0)
    args = p.parse_args()
    global SIGN
    SIGN = args.sign
    metas = []
    for branch in args.branches.split(","):
        meta = run(args.scenario, args.seed, args.fork, branch, args.steps,
                   f"{args.out}/{args.scenario}_s{args.seed}_{branch}.npz")
        metas.append(meta)
    print(json.dumps(metas))


if __name__ == "__main__":
    main()
