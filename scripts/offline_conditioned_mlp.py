#!/usr/bin/env python3
"""Offline conditioned-contact MLP on existing navigation-A trajectories.

Each legal source window is duplicated into an allow-contact and a
forbid-contact condition. Target y = forbid AND collision at the window's
current step. Source labels (original agent_collisions) are kept separate from
this offline requirement.

Splits are made on source trajectories first: seed101 episodes 0-2 train,
episodes 3-4 dev; seed202 is the holdout. Condition variants and adjacent
windows of one source never cross splits.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn

WINDOW = 8
CURRENT_OBS = 18
CURRENT_ACT = 2
LIDAR = slice(6, 18)


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files if k != "meta"}, json.loads(str(d["meta"]))


def first_done(done_seg):
    idx = np.flatnonzero(done_seg)
    return int(idx[0]) if len(idx) else None


def build_windows(path, seed, ep_len):
    """Return arrays of legal windows and their metadata."""
    d, meta = load(path)
    T, B, A = d["collision"].shape
    obs = d["obs"].astype(np.float32)
    act = d["action"].astype(np.float32)
    recs = []
    for e in range(T // ep_len):
        base = e * ep_len
        for b in range(B):
            term = first_done(d["done"][base:base + ep_len, b])
            for a in range(A):
                for t in range(base + WINDOW - 1, base + ep_len):
                    local = t - base
                    if term is not None and local > term:
                        continue  # post-terminal
                    if term is not None and t > base + term:
                        continue
                    if t - WINDOW + 1 < base:
                        continue
                    win_obs = obs[t - WINDOW + 1:t + 1, b, a]      # [8,18]
                    win_act = act[t - WINDOW + 1:t + 1, b, a]      # [8,2]
                    coll = float(d["collision"][t, b, a] < 0)
                    gap = float(d["gap"][t, b, a])
                    recs.append({
                        "seed": seed, "env": b, "episode": e, "t": t,
                        "agent": a, "group": f"{seed}:{b}:{e}",
                        "obs": win_obs, "act": win_act,
                        "collision": coll, "gap": gap,
                        "lidar_max": float(win_obs[-1, LIDAR].max()),
                    })
    return recs


def split_of(r):
    if r["seed"] == 202:
        return "holdout"
    return "train" if r["episode"] < 3 else "dev"


def make_variants(recs):
    xs, ys, meta = [], [], []
    for r in recs:
        flat = np.concatenate([r["obs"].reshape(-1), r["act"].reshape(-1)])
        for forbid in (0.0, 1.0):
            xs.append(np.concatenate([flat, [forbid]]))
            ys.append(forbid * r["collision"])
            m = dict(r)
            m["forbid"] = forbid
            m["y"] = forbid * r["collision"]
            meta.append(m)
    return np.stack(xs), np.array(ys, dtype=np.float32), meta


class MLP(nn.Module):
    def __init__(self, dim, use_req=True):
        super().__init__()
        self.use_req = use_req
        self.net = nn.Sequential(
            nn.Linear(dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        if not self.use_req:
            # zero out the requirement bit so the trajectory-only control sees
            # the same input layout but no condition information
            x = x.clone()
            x[:, -1] = 0.0
        return self.net(x).squeeze(-1)


def train(x_tr, y_tr, seed, use_req=True, epochs=15, lr=1e-3, bs=256):
    torch.manual_seed(seed)
    random.seed(seed)
    model = MLP(x_tr.shape[1], use_req=use_req)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    w = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=w)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.where(torch.from_numpy(y_tr) > 0.5,
                    torch.full((len(y_tr),), 0.5 / max(float(pos), 1.0)),
                    torch.full((len(y_tr),), 0.5 / max(float(neg), 1.0))),
        num_samples=len(y_tr), replacement=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, sampler=sampler)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x))).numpy()


def paired_metrics(meta, scores, thr=0.5):
    pred = scores >= thr
    pair = {}
    for m, p in zip(meta, pred):
        key = (m["group"], m["t"], m["agent"])
        pair.setdefault(key, {})[m["forbid"]] = (bool(m["collision"]), bool(p))
    contact_ok = contact_n = nocontact_ok = nocontact_n = 0
    for key, v in pair.items():
        if len(v) != 2:
            continue
        forbid_pred = v[1.0][1]
        allow_pred = v[0.0][1]
        if v[1.0][0]:  # contact at this source window
            contact_n += 1
            contact_ok += int(forbid_pred and not allow_pred)
        else:
            nocontact_n += 1
            nocontact_ok += int((not forbid_pred) and (not allow_pred))
    return {"contact_pairs": contact_n, "contact_pair_acc": contact_ok / max(contact_n, 1),
            "nocontact_pairs": nocontact_n, "nocontact_pair_acc": nocontact_ok / max(nocontact_n, 1)}


def variant_metrics(meta, scores, thr=0.5):
    pred = scores >= thr
    y = np.array([m["y"] for m in meta])
    tp = int((pred & (y > 0.5)).sum()); fp = int((pred & (y < 0.5)).sum())
    fn = int((~pred & (y > 0.5)).sum()); tn = int((~pred & (y < 0.5)).sum())
    near = np.array([m["gap"] <= 0.05 and m["collision"] < 0.5 for m in meta])
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "near_false": float((pred & near).sum()) / max(int(near.sum()), 1),
            "n": int(len(y))}


def rule_scores(meta):
    return np.array([1.0 if (m["forbid"] > 0.5 and m["lidar_max"] >= 0.25) else 0.0
                     for m in meta])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nav-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    recs = []
    for seed in (101, 202):
        recs += build_windows(f"{args.nav_root}/nav_seed{seed}.npz", seed, 128)

    splits = {"train": [], "dev": [], "holdout": []}
    for r in recs:
        splits[split_of(r)].append(r)
    x_all, y_all, meta_all = make_variants(recs)
    idx = {"train": [], "dev": [], "holdout": []}
    for i, m in enumerate(meta_all):
        idx[split_of(m)].append(i)
    for k in idx:
        idx[k] = np.array(idx[k], dtype=np.int64)

    x_tr, y_tr = x_all[idx["train"]], y_all[idx["train"]]
    x_dv, y_dv = x_all[idx["dev"]], y_all[idx["dev"]]
    x_ho, y_ho = x_all[idx["holdout"]], y_all[idx["holdout"]]
    meta = {"train": [meta_all[i] for i in idx["train"]],
            "dev": [meta_all[i] for i in idx["dev"]],
            "holdout": [meta_all[i] for i in idx["holdout"]]}

    # normalization fit on train only
    mu = x_tr.mean(0); sd = x_tr.std(0); sd[sd < 1e-6] = 1.0
    for arr_name in ("x_tr", "x_dv", "x_ho"):
        pass
    x_tr = ((x_tr - mu) / sd).astype(np.float32)
    x_dv = ((x_dv - mu) / sd).astype(np.float32)
    x_ho = ((x_ho - mu) / sd).astype(np.float32)

    # programmatic checks
    checks = {}
    checks["unit"] = {"train": len(splits["train"]), "dev": len(splits["dev"]),
                      "holdout": len(splits["holdout"])}
    groups = {k: {m["group"] for m in meta[k]} for k in meta}
    checks["group_disjoint"] = bool(groups["train"].isdisjoint(groups["dev"]) and
                                    groups["train"].isdisjoint(groups["holdout"]) and
                                    groups["dev"].isdisjoint(groups["holdout"]))
    checks["variant_counts"] = {k: int(len(meta[k])) for k in meta}
    # every source window has both conditions
    pair_keys = {}
    for m in meta_all:
        pair_keys.setdefault((m["group"], m["t"], m["agent"]), 0)
        pair_keys[(m["group"], m["t"], m["agent"])] += 1
    checks["all_pairs_have_two_conditions"] = all(v == 2 for v in pair_keys.values())

    results = {}
    models = {}
    for seed in (0, 1, 2):
        for name, use_req in (("full", True), ("traj_only", False)):
            model = train(x_tr, y_tr, seed, use_req=use_req)
            models[f"{name}_{seed}"] = model
            s_ho = predict(model, x_ho)
            s_dv = predict(model, x_dv)
            results[f"{name}_{seed}"] = {
                "paired": paired_metrics(meta["holdout"], s_ho),
                "variant": variant_metrics(meta["holdout"], s_ho),
                "dev_paired": paired_metrics(meta["dev"], s_dv),
                "dev_variant": variant_metrics(meta["dev"], s_dv),
            }
            torch.save(model.state_dict(), f"{args.out}/{name}_{seed}.pt")

    # controls
    s = rule_scores(meta["holdout"])
    results["rule_lidar"] = {"paired": paired_metrics(meta["holdout"], s),
                             "variant": variant_metrics(meta["holdout"], s)}
    s_req = np.array([1.0 if m["forbid"] > 0.5 else 0.0 for m in meta["holdout"]])
    results["req_only"] = {"paired": paired_metrics(meta["holdout"], s_req),
                           "variant": variant_metrics(meta["holdout"], s_req)}

    # example pairs
    def pairs_for(name):
        model = models[name]
        s = predict(model, x_ho)
        pred = s >= 0.5
        pair = {}
        for m, p, sc in zip(meta["holdout"], pred, s):
            key = (m["group"], m["t"], m["agent"])
            pair.setdefault(key, {})[m["forbid"]] = (bool(m["collision"]), bool(p), float(sc))
        out = {"correct_contact": [], "correct_nocontact": [], "wrong_contact": [], "wrong_nocontact": []}
        for key, v in pair.items():
            if len(v) != 2:
                continue
            forbid_pred = v[1.0][1]; allow_pred = v[0.0][1]
            contact = v[1.0][0]
            ok = (forbid_pred and not allow_pred) if contact else ((not forbid_pred) and (not allow_pred))
            bucket = ("correct_" if ok else "wrong_") + ("contact" if contact else "nocontact")
            item = {"source": key, "contact": contact, "forbid_pred": forbid_pred,
                    "allow_pred": allow_pred, "forbid_score": round(v[1.0][2], 4),
                    "allow_score": round(v[0.0][2], 4)}
            if len(out[bucket]) < 2:
                out[bucket].append(item)
        return out

    # holdout source-window and event counts (independent units)
    ho = splits["holdout"]
    contact_w = [r for r in ho if r["collision"] > 0.5]
    near_w = [r for r in ho if r["collision"] < 0.5 and r["gap"] <= 0.05]
    normal_w = [r for r in ho if r["collision"] < 0.5 and r["gap"] > 0.15]
    events = set()
    by_group = {}
    for r in ho:
        by_group.setdefault((r["group"], r["agent"]), []).append((r["t"], r["collision"]))
    for key, seq in by_group.items():
        seq.sort()
        prev_t, prev_c = None, 0.0
        for t, c in seq:
            if c > 0.5 and not (prev_c > 0.5 and prev_t == t - 1):
                events.add((key, t))
            prev_t, prev_c = t, c
    stats = {"source_windows": len(ho), "contact_windows": len(contact_w),
             "near_windows": len(near_w), "normal_windows": len(normal_w),
             "contact_events": len(events)}

    report = {"checks": checks, "results": results, "examples": pairs_for("full_0"),
              "holdout_stats": stats,
              "config": {"window": WINDOW, "hidden": [32, 16], "epochs": 15, "lr": 1e-3,
                         "batch": 256, "thr": 0.5}}
    with open(f"{args.out}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
