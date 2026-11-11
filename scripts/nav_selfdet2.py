#!/usr/bin/env python3
"""High-confidence self-failure detection with held-out mechanisms and targets.

Groups (frozen before collection):
  train 1800, dev 300, test_same 900, test_target 900, test_mech_reverse 450,
  test_mech_lateral 450.
Seen controllers: direct/pause/slow/leave/noisy. Held-out mechanisms:
reverse-mid-path and lateral-oscillation-near-goal.
Model target: "has arrived by now" (binary). Deployment maps arrival to
success and uses the clock to declare failure only at the deadline.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import vmas

WINDOW = 8
NUM_ENVS = 8
SEEN_CTRLS = ["direct", "pause", "slow", "leave", "noisy"]

GROUPS = {
    "train": dict(n=1800, region="seen", radius=0.05, deadline=64, ctrl="seen"),
    "dev": dict(n=300, region="seen", radius=0.05, deadline=64, ctrl="seen"),
    "test_same": dict(n=900, region="seen", radius=0.05, deadline=64, ctrl="seen"),
    "test_target": dict(n=900, region="holdout", radius=0.08, deadline=80, ctrl="seen"),
    "test_mech_reverse": dict(n=450, region="seen", radius=0.05, deadline=64, ctrl="reverse", dmin=0.30, dmax=0.90),
    "test_mech_lateral": dict(n=450, region="seen", radius=0.05, deadline=64, ctrl="lateral", dmin=0.30, dmax=0.90),
}


def sample_goal(rng, start, region, dmin, dmax):
    for _ in range(300):
        ang = rng.uniform(0, 2 * np.pi)
        d = rng.uniform(dmin, dmax)
        g = start + d * np.array([np.cos(ang), np.sin(ang)])
        if np.abs(g).max() > 0.98:
            continue
        if region == "seen" and g[0] >= 0:
            continue
        if region == "holdout" and g[0] <= 0:
            continue
        return g
    return np.array([-0.7, 0.0]) if region == "seen" else np.array([0.7, 0.0])


def collect_group(name, cfg, seed, out_dir):
    n = cfg["n"]; region = cfg["region"]; radius = cfg["radius"]; D = cfg["deadline"]
    rng = np.random.default_rng(seed)
    recs = []
    steps = 0
    ep = 0
    while ep < n:
        batch = min(NUM_ENVS, n - ep)
        env = vmas.make_env("navigation", num_envs=batch, device="cpu",
                            continuous_actions=True, seed=seed + ep, n_agents=1,
                            collisions=False)
        env.reset(seed=seed + ep)
        ag = env.scenario.world.agents[0]
        goals, ctrls, params = [], [], []
        for i in range(batch):
            idx = ep + i
            if cfg["ctrl"] == "seen":
                c = SEEN_CTRLS[idx % len(SEEN_CTRLS)]
            else:
                c = cfg["ctrl"]
            g = sample_goal(rng, ag.state.pos[i].numpy(), region,
                            cfg.get("dmin", 0.35), cfg.get("dmax", 1.05 if D == 64 else 2.0))
            goals.append(g); ctrls.append(c)
            if c == "reverse":
                params.append((int(rng.integers(10, 30)), int(rng.integers(5, 13))))
            elif c == "lateral":
                params.append((float(rng.uniform(0.4, 0.9)), float(rng.uniform(0.15, 0.3))))
            else:
                params.append((0, 0))
        g_t = torch.tensor(np.stack(goals), dtype=torch.float32)
        ag.goal.set_pos(g_t, None)
        ag.pos_shaping = (ag.state.pos - ag.goal.state.pos).norm(dim=-1)
        arrived = torch.full((batch,), -1, dtype=torch.long)
        prev = []
        rec = {k: [] for k in ("obs", "act", "pos", "dist")}
        for t in range(D):
            pos = ag.state.pos; vel = ag.state.vel
            delta = ag.goal.state.pos - pos
            nrm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            speed = torch.tensor([0.10 if c == "slow" else 0.18 for c in ctrls]).unsqueeze(-1)
            u = (delta / nrm * speed - vel) * 8.0
            pause = torch.tensor([(c == "pause" and 15 <= t < 30) for c in ctrls])
            u = torch.where(pause.unsqueeze(-1), torch.zeros_like(u), u)
            leave = torch.tensor([c == "leave" for c in ctrls]) & (arrived >= 0)
            u = torch.where(leave.unsqueeze(-1), -u, u)
            noise = torch.tensor([c == "noisy" for c in ctrls])
            phase = torch.arange(batch, dtype=u.dtype).unsqueeze(-1)
            u = torch.where(noise.unsqueeze(-1), u + 0.35 * torch.sin(0.7 * t + phase), u)
            if t >= 2 and len(prev) >= 2:
                u = torch.where(noise.unsqueeze(-1), prev[-2], u)
            # reverse mechanism
            rev = torch.tensor([(c == "reverse" and p[0] <= t < p[0] + p[1])
                                for c, p in zip(ctrls, params)])
            u = torch.where(rev.unsqueeze(-1), -u, u)
            # lateral oscillation near goal
            lat = torch.tensor([c == "lateral" for c in ctrls])
            dist_now = (ag.state.pos - ag.goal.state.pos).norm(dim=-1)
            near = lat & (dist_now < 0.30)
            perp = torch.stack([-delta[:, 1], delta[:, 0]], dim=-1)
            perp = perp / perp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            amp = torch.tensor([p[1] if c == "lateral" else 0.0 for c, p in zip(ctrls, params)]).unsqueeze(-1)
            freq = torch.tensor([p[0] for c, p in zip(ctrls, params)]).unsqueeze(-1)
            u = torch.where(near.unsqueeze(-1), u + amp * perp * torch.sin(freq * t), u)
            u = u.clamp(-1.0, 1.0)
            prev.append(u.clone())
            obs, rews, dones, infos = env.step([u])
            d = (ag.state.pos - ag.goal.state.pos).norm(dim=-1)
            newly = (d < radius) & (arrived < 0)
            arrived = torch.where(newly, torch.full_like(arrived, t), arrived)
            rec["obs"].append(obs[0].numpy()); rec["act"].append(u.numpy())
            rec["pos"].append(ag.state.pos.numpy()); rec["dist"].append(d.numpy())
        steps += batch * D
        for i in range(batch):
            recs.append({"episode": ep + i, "group": name, "ctrl": ctrls[i],
                         "region": region, "radius": radius, "deadline": D,
                         "goal": goals[i].tolist(), "arrived_t": int(arrived[i]),
                         "min_dist": float(np.min([rec["dist"][t][i] for t in range(D)])),
                         "obs": np.stack([rec["obs"][t][i] for t in range(D)]),
                         "act": np.stack([rec["act"][t][i] for t in range(D)]),
                         "pos": np.stack([rec["pos"][t][i] for t in range(D)])})
        ep += batch
    np.save(os.path.join(out_dir, f"{name}.npy"),
            np.array([{"episode": r["episode"], "group": r["group"], "ctrl": r["ctrl"],
                       "region": r["region"], "radius": r["radius"],
                       "deadline": r["deadline"], "goal": r["goal"],
                       "arrived_t": r["arrived_t"], "min_dist": r["min_dist"]}
                      for r in recs], dtype=object), allow_pickle=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}_arrays.npz"),
                        obs=np.stack([r["obs"] for r in recs]),
                        act=np.stack([r["act"] for r in recs]),
                        pos=np.stack([r["pos"] for r in recs]))
    return {"group": name, "episodes": n, "steps": steps,
            "success": sum(1 for r in recs if r["arrived_t"] >= 0),
            "failure": sum(1 for r in recs if r["arrived_t"] < 0)}


def make_features(meta, arrays):
    obs, act, pos = arrays["obs"], arrays["act"], arrays["pos"]
    E, T, _ = obs.shape
    X, Y, KEYS = [], [], []
    for e in range(E):
        m = meta[e]
        g = np.array(m["goal"], dtype=np.float32)
        r = float(m["radius"]); D = int(m["deadline"])
        rel = pos[e] - g[None, :]
        d = np.linalg.norm(rel, axis=-1)
        arr = int(m["arrived_t"])
        for t in range(WINDOW - 1, T):
            win = obs[e, t - WINDOW + 1:t + 1].copy()
            win[:, 4:6] = rel[t - WINDOW + 1:t + 1]
            flat = np.concatenate([win.reshape(-1), act[e, t - WINDOW + 1:t + 1].reshape(-1)])
            hist = d[:t + 1].min()
            gv = np.array([g[0], g[1], r, D, d[t] / r, hist / r,
                           (t / (D - 1)), 1.0], dtype=np.float32)
            X.append(np.concatenate([flat, gv]))
            Y.append(1 if (arr >= 0 and t >= arr) else 0)
            KEYS.append((e, t))
    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.int64), KEYS


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden[0]), nn.ReLU(),
                                 nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                                 nn.Linear(hidden[1], 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(x, y, seed, hidden, epochs=12, bs=1024):
    torch.manual_seed(seed)
    m = MLP(x.shape[1], hidden)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    pos = float(y.sum()); neg = len(y) - pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1.0)))
    sw = np.where(y > 0.5, 0.5 / max(pos, 1.0), 0.5 / max(neg, 1.0))
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(sw, dtype=torch.double),
                                                     num_samples=len(y), replacement=True)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, sampler=sampler)
    m.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); loss_fn(m(xb), yb.float()).backward(); opt.step()
    return m


def predict(m, x):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.from_numpy(x))).numpy()


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def eval_group(probs, keys, meta, threshold):
    # one decision per episode: success if any step prob >= threshold
    best = {}
    for (e, t), p in zip(keys, probs):
        best[e] = max(best.get(e, 0.0), float(p))
    tp = fp = fn = tn = 0
    for e, p in best.items():
        truth = meta[e]["arrived_t"] >= 0
        pred = p >= threshold
        if truth and pred: tp += 1
        elif truth and not pred: fn += 1
        elif (not truth) and pred: fp += 1
        else: tn += 1
    n = tp + fp + fn + tn
    rec = tp / max(tp + fn, 1)          # success recall
    prec = tp / max(tp + fp, 1)
    fail_rec = tn / max(tn + fp, 1)     # failure recall (predicted failure correctly)
    fail_prec = tn / max(tn + fn, 1)
    return {"n": n, "success_recall": rec, "success_recall_ci": wilson(tp, tp + fn),
            "failure_recall": fail_rec, "failure_recall_ci": wilson(tn, tn + fp),
            "failure_precision": fail_prec, "failure_precision_ci": wilson(tn, tn + fn),
            "success_misjudged": fn / max(tp + fn, 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--collect", action="store_true")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=8100)
    args = p.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    summary = {"groups": {}}
    for name, cfg in GROUPS.items():
        cfg = dict(cfg)
        cfg["n"] = max(8, int(cfg["n"] * args.scale))
        summary["groups"][name] = collect_group(name, cfg, args.seed, args.dir)
        print(json.dumps(summary["groups"][name]))

    # load
    data = {}
    for name in GROUPS:
        meta = np.load(os.path.join(args.dir, f"{name}.npy"), allow_pickle=True)
        arrays = dict(np.load(os.path.join(args.dir, f"{name}_arrays.npz")))
        data[name] = (list(meta), arrays)

    feats = {}
    for name, (meta, arrays) in data.items():
        feats[name] = make_features(meta, arrays)
    x_tr, y_tr, _ = feats["train"]
    mu = x_tr.mean(0); sd = x_tr.std(0); sd[sd < 1e-6] = 1.0
    x_tr = ((x_tr - mu) / sd).astype(np.float32)

    dev_x, dev_y, dev_keys = feats["dev"]
    dev_x = ((dev_x - mu) / sd).astype(np.float32)

    models = {}
    dev_scores = {}
    for hidden in [(32, 16), (64, 32)]:
        for seed in (0, 1, 2):
            m = train_model(x_tr, y_tr, seed, hidden)
            models[(hidden, seed)] = m
            probs = predict(m, dev_x)
            best = {}
            for (e, t), p in zip(dev_keys, probs):
                best[e] = max(best.get(e, 0.0), float(p))
            truth = np.array([data["dev"][0][e]["arrived_t"] >= 0 for e in range(len(best))])
            pred = np.array([best[e] for e in range(len(best))])
            # threshold scan on dev for best balanced accuracy
            best_th, best_ba = 0.5, -1
            for th in np.arange(0.1, 0.91, 0.05):
                pp = pred >= th
                ba = 0.5 * ((pp & truth).sum() / max(truth.sum(), 1) +
                            (~pp & ~truth).sum() / max((~truth).sum(), 1))
                if ba > best_ba:
                    best_th, best_ba = float(th), ba
            dev_scores[(hidden, seed)] = (best_th, float(best_ba), hidden)
    # choose config by mean dev balanced accuracy
    cfg_scores = {}
    for (hidden, seed), (th, ba, _) in dev_scores.items():
        cfg_scores.setdefault(hidden, []).append(ba)
    chosen_hidden = max(cfg_scores, key=lambda h: np.mean(cfg_scores[h]))
    chosen_th = float(np.mean([dev_scores[(chosen_hidden, s)][0] for s in (0, 1, 2)]))

    results = {"chosen_hidden": chosen_hidden, "threshold": chosen_th,
               "dev": {str(k): v for k, v in dev_scores.items()}, "groups": {}}
    test_names = ["test_same", "test_target", "test_mech_reverse", "test_mech_lateral"]
    for gname in test_names:
        meta, _ = data[gname]
        x, y, keys = feats[gname]
        x = ((x - mu) / sd).astype(np.float32)
        per_seed = []
        for seed in (0, 1, 2):
            probs = predict(models[(chosen_hidden, seed)], x)
            per_seed.append(eval_group(probs, keys, meta, chosen_th))
        results["groups"][gname] = per_seed

    torch.save({"hidden": chosen_hidden,
                "state_dicts": {s: models[(chosen_hidden, s)].state_dict() for s in (0, 1, 2)},
                "threshold": chosen_th, "mu": mu, "sd": sd},
               os.path.join(args.dir, "model.pt"))
    preds = {}
    for gname in test_names:
        meta, _ = data[gname]
        x, y, keys = feats[gname]
        x = ((x - mu) / sd).astype(np.float32)
        for seed in (0, 1, 2):
            probs = predict(models[(chosen_hidden, seed)], x)
            best = {}
            for (e, t), p in zip(keys, probs):
                best[e] = max(best.get(e, 0.0), float(p))
            preds[f"{gname}_seed{seed}"] = np.array([best[e] for e in range(len(meta))])
    np.savez(os.path.join(args.dir, "predictions.npz"), **preds)
    with open(os.path.join(args.dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
