#!/usr/bin/env python3
"""Collect one fixed navigation scene with three seen failure types and one
fully held-out failure type (forbidden-zone entry).

Scene (identical in every episode):
  ego goal (0.8, 0), radius 0.05, deadline 64 steps
  parked teammate (collide=True), two static box obstacles, one static
  no-physics forbidden circle at (0.35, 0.55) r=0.14.
Failure types: deadline not reached, teammate contact, obstacle contact
(seen); forbidden-zone entry (held out). Normal: arrived without violation.
"""
import argparse
import json
import os

import numpy as np
import torch
import vmas

D = 64
RADIUS = 0.05
GOAL = np.array([0.8, 0.0], dtype=np.float32)
AGENT_R = 0.1
OBS_CENTERS = np.array([[-0.2, 0.55], [-0.2, -0.55]], dtype=np.float32)
OBS_R = 0.12
ZONE_C = np.array([0.35, 0.55], dtype=np.float32)
ZONE_R = 0.14
CONTACT_EPS = 0.01
NUM_ENVS = 8
FEAT_DIM = 25
SEEN = ["deadline", "teammate", "obstacle"]
WANT = ["normal", "deadline", "teammate", "obstacle", "forbidden"]

FULL = {
    "train": {"normal": 400, "deadline": 400, "teammate": 400, "obstacle": 400},
    "dev": {"normal": 100, "deadline": 100, "teammate": 100, "obstacle": 100},
    "test_seen": {"normal": 820, "deadline": 400, "teammate": 400, "obstacle": 400},
    "test_holdout": {"forbidden": 400},
}
PILOT = {
    "train": {"normal": 8, "deadline": 8, "teammate": 8, "obstacle": 8},
    "dev": {"normal": 8, "deadline": 8, "teammate": 8, "obstacle": 8},
    "test_seen": {"normal": 8, "deadline": 8, "teammate": 8, "obstacle": 8},
    "test_holdout": {"forbidden": 8},
}
STEP_LIMIT = 385000


def make_env(batch, seed):
    env = vmas.make_env("navigation", num_envs=batch, device="cpu",
                        continuous_actions=True, seed=seed, n_agents=2,
                        collisions=True)
    from vmas.simulator.core import Landmark, Sphere
    for k, c in enumerate(OBS_CENTERS):
        lm = Landmark(name=f"obs{k}", shape=Sphere(radius=OBS_R), collide=True)
        env.scenario.world.add_landmark(lm)
        lm.set_pos(torch.tensor(c).repeat(batch, 1), None)
        lm.set_vel(torch.zeros(batch, 2), None)
    env.scenario.world.agents[0].goal.set_pos(torch.tensor(GOAL).repeat(batch, 1), None)
    return env


def geom(env):
    a = env.scenario.world.agents
    p0, p1 = a[0].state.pos, a[1].state.pos
    gd = (p0 - a[0].goal.state.pos).norm(dim=-1)
    tg = (p0 - p1).norm(dim=-1) - 2 * AGENT_R
    og = torch.stack([(p0 - torch.tensor(c)).norm(dim=-1) - (OBS_R + AGENT_R) for c in OBS_CENTERS], -1).min(-1).values
    zm = (p0 - torch.tensor(ZONE_C)).norm(dim=-1) - ZONE_R
    return gd, tg, og, zm


def valid_start(p, tm, allow_zone=False):
    if np.abs(p).max() > 0.92:
        return False
    if any(np.linalg.norm(p - c) < OBS_R + AGENT_R + 0.06 for c in OBS_CENTERS):
        return False
    zm = np.linalg.norm(p - ZONE_C) - ZONE_R
    if not allow_zone and zm < 0.03:
        return False
    if np.linalg.norm(p - tm) < 2 * AGENT_R + 0.15:
        return False
    return True


def sample_park(rng, mode):
    if mode == "deadline":
        for _ in range(200):
            p = np.array([-0.85, rng.choice([-0.85, 0.85])], dtype=np.float32)
            if valid_start(p, np.array([9.0, 9.0])):
                return p
    for _ in range(200):
        p = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.3, 0.3)], dtype=np.float32)
        if valid_start(p, np.array([9.0, 9.0])):
            return p
    return np.array([0.0, -0.2], dtype=np.float32)


def sample_start(rng, mode, tm, variant):
    for _ in range(400):
        if mode == "normal":
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0.30, 0.75)
            p = GOAL + r * np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
            if np.linalg.norm(p - ZONE_C) < ZONE_R + 0.05:
                continue
        elif mode == "normal_near":
            p = np.array([rng.uniform(0.20, 0.50), rng.uniform(0.12, 0.30)], dtype=np.float32)
        elif mode == "deadline":
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(1.0, 1.55)
            p = GOAL + r * np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
        elif mode == "teammate":
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0.35, 0.70)
            p = tm + r * np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
        elif mode == "obstacle":
            k = variant % 2
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0.32, 0.55)
            p = OBS_CENTERS[k] + r * np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
        else:  # forbidden
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0.35, 0.65)
            p = ZONE_C + r * np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
        if not valid_start(p, tm, allow_zone=(mode == "forbidden")):
            continue
        if mode == "normal":
            # direct path must stay clear of both boxes and the forbidden circle
            v = GOAL - p; v = v / max(np.linalg.norm(v), 1e-6)
            if any(np.linalg.norm((c - p) - np.dot(c - p, v) * v) < OBS_R + AGENT_R + 0.03
                   for c in OBS_CENTERS):
                continue
            if np.linalg.norm((ZONE_C - p) - np.dot(ZONE_C - p, v) * v) < ZONE_R + 0.03:
                continue
        if mode == "deadline":
            if np.linalg.norm(p - GOAL) < 0.9:
                continue
            if abs(p[1]) > 0.25:
                continue
        return p
    return None


def run_batch(seed, batch, mode, variant_ids, rng):
    env = make_env(batch, seed)
    agents = env.scenario.world.agents
    p_tm = np.stack([sample_park(rng, mode) for _ in range(batch)])
    p0 = []
    for i in range(batch):
        p = sample_start(rng, mode, p_tm[i], variant_ids[i])
        if p is None:
            p = p_tm[i] + np.array([0.5, 0.0], dtype=np.float32)
            p = np.clip(p, -0.9, 0.9)
        p0.append(p)
    p0 = np.stack(p0).astype(np.float32)
    agents[0].set_pos(torch.from_numpy(p0), None)
    agents[0].set_vel(torch.zeros(batch, 2), None)
    agents[1].set_pos(torch.from_numpy(p_tm), None)
    agents[1].set_vel(torch.zeros(batch, 2), None)
    agents[1].goal.set_pos(torch.from_numpy(p_tm), None)
    noise = rng.normal(0, 0.01, size=(D, batch, 2)).astype(np.float32)

    feats = np.zeros((batch, D, FEAT_DIM), dtype=np.float32)
    first_t = np.full(batch, -1, dtype=np.int64)
    first_type = np.array(["none"] * batch, dtype=object)
    arrived = np.full(batch, -1, dtype=np.int64)
    zone_entered = np.zeros(batch, dtype=bool)
    hmin = np.full((batch, 4), np.inf, dtype=np.float32)
    paused = [int(rng.integers(24, 31)) for _ in range(batch)]
    speed = np.array([0.14 if v % 2 else 0.20 for v in variant_ids], dtype=np.float32)
    near_wp = ZONE_C + np.array([0.0, -(ZONE_R + 0.025)], dtype=np.float32)
    passed = np.zeros(batch, dtype=bool)

    for t in range(D):
        a = env.scenario.world.agents
        p0t, p1t = a[0].state.pos, a[1].state.pos
        v0, v1 = a[0].state.vel, a[1].state.vel
        g0 = a[0].goal.state.pos
        delta = g0 - p0t
        n = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sp = torch.tensor(speed).unsqueeze(-1) if mode in ("teammate", "obstacle") else torch.full((batch, 1), 0.18)
        u0 = (delta / n * sp - v0) * 8.0
        if mode == "deadline":
            if variant_ids[0] % 2 == 0:
                stop = torch.tensor([t < paused[i] for i in range(batch)]).unsqueeze(-1)
                u0 = torch.where(stop, torch.zeros_like(u0), u0)
            else:
                u0 = (delta / n * 0.07 - v0) * 8.0
        elif mode == "teammate":
            delta = p1t - p0t
            n = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            u0 = (delta / n * sp - v0) * 8.0
        elif mode == "obstacle":
            k = torch.tensor([v % 2 for v in variant_ids])
            c = torch.stack([torch.tensor(OBS_CENTERS[v % 2]) for v in variant_ids])
            target = c + torch.stack([torch.tensor([0.0, 0.06 if v % 2 else 0.0]) for v in variant_ids])
            delta = target - p0t
            n = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            u0 = (delta / n * sp - v0) * 8.0
        elif mode == "forbidden":
            zc = torch.tensor(ZONE_C).repeat(batch, 1)
            delta_z = zc - p0t
            nz = delta_z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            inside = (p0t - zc).norm(dim=-1, keepdim=True) <= ZONE_R - 0.01
            u0 = torch.where(inside, u0, (delta_z / nz * 0.18 - v0) * 8.0)
        elif mode == "normal_near":
            wp = torch.tensor(near_wp).repeat(batch, 1)
            dist_wp = (p0t - wp).norm(dim=-1, keepdim=True)
            delta_w = wp - p0t
            nw = delta_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            u_wp = (delta_w / nw * 0.18 - v0) * 8.0
            passed |= (dist_wp[:, 0] < 0.05).numpy()
            u0 = torch.where(torch.from_numpy(passed).unsqueeze(-1), u0, u_wp)
        u0 = (u0 + torch.from_numpy(noise[t])).clamp(-1, 1)
        u1 = torch.zeros(batch, 2)
        obs, rews, dones, infos = env.step([u0, u1])

        a = env.scenario.world.agents
        p0t, p1t = a[0].state.pos, a[1].state.pos
        v0, v1 = a[0].state.vel, a[1].state.vel
        gd, tg, og, zm = geom(env)
        gd, tg, og, zm = gd.numpy(), tg.numpy(), og.numpy(), zm.numpy()
        cur = np.stack([gd, tg, og, zm], -1)
        hmin = np.minimum(hmin, cur)
        zone_entered |= zm <= 0
        p0n = p0t.numpy(); v0n = v0.numpy(); p1n = p1t.numpy(); v1n = v1.numpy()
        g0n = a[0].goal.state.pos.numpy()
        zc = np.repeat(ZONE_C[None], batch, 0)
        o = np.stack([np.repeat(c[None], batch, 0) - p0n for c in OBS_CENTERS], 1)
        feats[:, t, :16] = np.concatenate(
            [p0n, v0n, p0n - g0n, p1n - p0n, v1n, o[:, 0], o[:, 1], zc - p0n], axis=-1)
        feats[:, t, 16:20] = cur
        feats[:, t, 20:24] = hmin
        feats[:, t, 24] = t / (D - 1)

        newly = (arrived < 0) & (gd < RADIUS)
        arrived[newly] = t
        for i in range(batch):
            if first_t[i] >= 0:
                continue
            vio = []
            if tg[i] <= CONTACT_EPS:
                vio.append("teammate")
            if og[i] <= CONTACT_EPS:
                vio.append("obstacle")
            if zm[i] <= 0.0:
                vio.append("forbidden")
            if len(vio) == 1:
                first_t[i] = t; first_type[i] = vio[0]
            elif len(vio) > 1:
                first_t[i] = t; first_type[i] = "ambiguous"
    return feats, first_t, first_type, arrived, zone_entered, p0, p_tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--plan", choices=["pilot", "full"], default="full")
    ap.add_argument("--seed", type=int, default=9300)
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    plan = PILOT if args.plan == "pilot" else FULL
    rng = np.random.default_rng(args.seed)
    budget = {"steps": 0, "limit": STEP_LIMIT, "attempts": 0}
    summary = {}

    for split, quotas in plan.items():
        for want, quota in quotas.items():
            mode = "normal_near" if want == "normal" and rng.random() < 0.5 else want
            kept = {k: [] for k in ("feats", "meta")}
            tried = 0
            steps = 0
            max_attempts = quota * 3
            while len(kept["feats"]) < quota and tried < max_attempts and budget["steps"] < STEP_LIMIT:
                batch = min(NUM_ENVS, quota - len(kept["feats"]))
                variants = [int(rng.integers(0, 2)) for _ in range(batch)]
                feats, ft, ftype, arr, zent, p0, ptm = run_batch(
                    args.seed + 1000 * len(summary) + tried, batch, mode, variants, rng)
                tried += batch
                steps += batch * D
                budget["steps"] += batch * D
                budget["attempts"] += batch
                for i in range(batch):
                    typ = ftype[i]
                    if typ == "none":
                        typ = "normal" if arr[i] >= 0 else "deadline"
                        if typ == "deadline":
                            ft[i] = D - 1
                    if typ != want:
                        continue
                    if split in ("train", "dev") and zent[i]:
                        continue
                    kept["feats"].append(feats[i])
                    kept["meta"].append({
                        "split": split, "want": want, "mode": mode,
                        "agent_type": typ, "first_t": int(ft[i]),
                        "arrived_t": int(arr[i]), "zone_entered": bool(zent[i]),
                        "variant": int(variants[i]), "start": p0[i].tolist(),
                        "teammate": ptm[i].tolist(),
                        "min_goal": float(np.min(feats[i, :, 16])),
                        "min_teammate": float(np.min(feats[i, :, 17])),
                        "min_obstacle": float(np.min(feats[i, :, 18])),
                        "min_zone_margin": float(np.min(feats[i, :, 19])),
                    })
            fname = f"{split}_{want}.npz"
            np.savez_compressed(os.path.join(args.dir, fname),
                                feats=np.stack(kept["feats"]) if kept["feats"] else np.zeros((0, D, FEAT_DIM), np.float32),
                                meta=np.array(kept["meta"], dtype=object), allow_pickle=True)
            got = len(kept["feats"])
            summary[f"{split}_{want}"] = {
                "want": want, "mode": mode, "quota": quota, "kept": got,
                "attempts": tried, "steps": steps,
                "yield": got / max(tried, 1),
                "zone_entered_kept": int(sum(m["zone_entered"] for m in kept["meta"])),
            }
            print(json.dumps(summary[f"{split}_{want}"]), flush=True)
            if budget["steps"] >= STEP_LIMIT:
                print("STEP LIMIT REACHED", flush=True)
                break
        if budget["steps"] >= STEP_LIMIT:
            break
    with open(os.path.join(args.dir, "summary.json"), "w") as f:
        json.dump({"plan": args.plan, "seed": args.seed, "budget": budget,
                   "files": summary}, f, indent=2)


if __name__ == "__main__":
    main()
