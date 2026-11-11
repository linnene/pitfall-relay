#!/usr/bin/env python3
"""Train one shared binary failure MLP on three seen failure types + normal,
then blind-test on the seen types and on the fully held-out forbidden-zone
entry type. Two widths x three seeds; dev selects only the threshold."""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

D = 64
RADIUS = 0.05
SEEN = ["deadline", "teammate", "obstacle"]
CLASSES = ["normal", "deadline", "teammate", "obstacle", "forbidden"]
WIDTHS = [(64, 32), (128, 64)]
TSEEDS = (0, 1, 2)
WIN = 2
D_FEAT = 25


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden[0]), nn.ReLU(),
                                 nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                                 nn.Linear(hidden[1], 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load(path):
    z = np.load(path, allow_pickle=True)
    return z["feats"].astype(np.float32), list(z["meta"])


def episode_labels(meta):
    y = np.zeros(D, dtype=np.float32)
    if meta["agent_type"] in SEEN and meta["first_t"] >= 0:
        y[meta["first_t"]:] = 1.0
    return y


def stack_split(d, prefix, classes):
    X, Y, M = [], [], []
    for c in classes:
        p = os.path.join(d, f"{prefix}_{c}.npz")
        if not os.path.exists(p):
            continue
        f, m = load(p)
        for i, md in enumerate(m):
            X.append(f[i]); Y.append(episode_labels(md)); M.append(md)
    if not X:
        return None, None, None
    return np.stack(X), np.stack(Y), M


def probs_episodes(model, X, mu, sd):
    Z = ((X - mu) / sd).astype(np.float32)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(len(Z)):
            out.append(torch.sigmoid(model(torch.from_numpy(Z[i]))).numpy())
    return np.stack(out)


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def metrics(episodes, class_name, masks):
    """masks: list of boolean alarm vectors of length D."""
    idx = [i for i, m in enumerate(episodes) if m["agent_type"] == class_name]
    n = len(idx)
    if class_name == "normal":
        k = sum(bool(masks[i].any()) for i in idx)
        hist = np.zeros(8, dtype=int)
        for i in idx:
            hit = np.where(masks[i])[0]
            if len(hit):
                hist[min(int(hit[0]) // 8, 7)] += 1
        return {"n": n, "fp": k, "fp_rate": k / max(n, 1),
                "fp_ci": wilson(k, n), "first_alarm_hist_8step": hist.tolist()}
    hits = 0; state_hits = 0; pre = 0; delays = []
    for i in idx:
        ft = episodes[i]["first_t"]; mask = masks[i]
        hit = np.where(mask)[0]
        at = int(hit[0]) if len(hit) else -1
        if at >= 0 and ft <= at <= ft + WIN:
            hits += 1; delays.append(at - ft)
        if mask[ft:ft + WIN + 1].any():
            state_hits += 1
        if at >= 0 and at < ft:
            pre += 1
    return {"n": n, "hits": hits, "recall": hits / max(n, 1),
            "recall_ci": wilson(hits, n), "state_hits": state_hits,
            "state_recall": state_hits / max(n, 1),
            "state_recall_ci": wilson(state_hits, n),
            "pre_alarm": pre, "pre_alarm_rate": pre / max(n, 1),
            "delay_mean": float(np.mean(delays)) if delays else None,
            "delay_median": float(np.median(delays)) if delays else None,
            "delays": delays}


def rule_masks(X):
    """Same-information deterministic rule on continuous features."""
    out = []
    for e in range(len(X)):
        tg = X[e, :, 17]; og = X[e, :, 18]; zm = X[e, :, 19]
        gd = X[e, :, 16]
        alarm = (tg <= 0.02) | (og <= 0.02) | (zm <= 0.02)
        alarm[-1] = alarm[-1] or (gd[-1] >= RADIUS)
        out.append(alarm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = args.dir
    train = stack_split(d, "train", ["normal"] + SEEN)
    dev = stack_split(d, "dev", ["normal"] + SEEN)
    test = stack_split(d, "test_seen", ["normal"] + SEEN)
    hold = stack_split(d, "test_holdout", ["forbidden"])
    assert train[0] is not None and dev[0] is not None and test[0] is not None
    Xtr, Ytr, Mtr = train
    Xdv, Ydv, Mdv = dev
    Xte = np.concatenate([test[0], hold[0]]); Mte = test[2] + hold[2]
    Xte = np.asarray(Xte)
    mu = Xtr.reshape(-1, D_FEAT).mean(0)
    sd = Xtr.reshape(-1, D_FEAT).std(0); sd[sd < 1e-6] = 1.0
    Ztr = ((Xtr - mu) / sd).astype(np.float32)

    # training tensors
    x = torch.from_numpy(Ztr.reshape(-1, D_FEAT))
    y = torch.from_numpy(Ytr.reshape(-1).astype(np.float32))
    pos = float(y.sum()); neg = len(y) - pos
    sw = np.where(Ytr.reshape(-1) > 0.5, 0.5 / max(pos, 1), 0.5 / max(neg, 1))
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.tensor(sw, dtype=torch.double), num_samples=len(y), replacement=True)
    ds = torch.utils.data.TensorDataset(x, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, sampler=sampler)

    # dev threshold selection (seen classes + normal only)
    dev_eps = Mdv
    results = {"config": {"D": D, "feat_dim": D_FEAT, "win": WIN,
                          "train_episodes": len(Mtr), "dev_episodes": len(Mdv),
                          "test_episodes": len(Mte), "widths": WIDTHS, "seeds": TSEEDS},
               "class_counts": {c: sum(1 for m in Mte if m["agent_type"] == c) for c in CLASSES},
               "heldout_zone_entered_in_train": sum(bool(m.get("zone_entered")) for m in Mtr),
               "heldout_zone_entered_in_dev": sum(bool(m.get("zone_entered")) for m in Mdv),
               "widths": {}}
    models = {}
    thresholds = {}
    for hidden in WIDTHS:
        key = f"{hidden[0]}/{hidden[1]}"
        results["widths"][key] = {"params": None, "seeds": {}, "thresholds": {}}
        for seed in TSEEDS:
            torch.manual_seed(seed)
            m = MLP(D_FEAT, hidden)
            results["widths"][key]["params"] = sum(p.numel() for p in m.parameters())
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            loss_fn = nn.BCEWithLogitsLoss()
            m.train()
            for _ in range(12):
                for xb, yb in dl:
                    opt.zero_grad(); loss_fn(m(xb), yb).backward(); opt.step()
            models[(hidden, seed)] = m
            dv_probs = probs_episodes(m, Xdv, mu, sd)
            feasible_cands = []
            all_cands = []
            for th in np.arange(0.05, 0.951, 0.025):
                masks = [dv_probs[i] >= th for i in range(len(dv_probs))]
                normal_m = metrics(dev_eps, "normal", masks)
                recs = [metrics(dev_eps, c, masks)["recall"] for c in SEEN]
                fp = normal_m["fp_rate"]
                score = float(np.mean(recs))
                rec = (float(th), score, fp)
                all_cands.append(rec)
                if fp <= 0.05:
                    feasible_cands.append(rec)
            if feasible_cands:
                best = max(feasible_cands, key=lambda r: (r[1], r[0]))
                feasible = True
            else:
                best = min(all_cands, key=lambda r: (r[2], -r[1]))
                feasible = False
            th, best_score, best_fp = best
            thresholds[(hidden, seed)] = th
            results["widths"][key]["thresholds"][seed] = {
                "threshold": th, "dev_fp_rate": best_fp,
                "dev_mean_seen_recall": best_score, "feasible_fp_le_5pct": bool(feasible)}
            # dev detail at chosen threshold
            dv_masks = [dv_probs[i] >= th for i in range(len(dv_probs))]
            results["widths"][key]["seeds"][seed] = {
                "dev": {c: metrics(dev_eps, c, dv_masks) for c in CLASSES if any(x["agent_type"] == c for x in dev_eps)},
            }
            torch.save(m.state_dict(), os.path.join(d, f"model_{hidden[0]}_{hidden[1]}_seed{seed}.pt"))

    # test evaluation per width/seed
    preds_dump = {}
    for hidden in WIDTHS:
        key = f"{hidden[0]}/{hidden[1]}"
        for seed in TSEEDS:
            th = thresholds[(hidden, seed)]
            probs = probs_episodes(models[(hidden, seed)], Xte, mu, sd)
            masks = [probs[i] >= th for i in range(len(probs))]
            results["widths"][key]["seeds"][seed]["test"] = {
                c: metrics(Mte, c, masks) for c in CLASSES}
            first = np.array([int(np.where(mk)[0][0]) if mk.any() else -1 for mk in masks])
            preds_dump[f"{hidden[0]}_{hidden[1]}_seed{seed}"] = np.stack([first, probs.max(1)], 1).astype(np.float32)
    # rule baseline
    rm = rule_masks(Xte)
    results["rule"] = {c: metrics(Mte, c, rm) for c in CLASSES}
    results["rule"]["note"] = "alarm if teammate gap<=0.02, obstacle gap<=0.02, zone margin<=0.02, or deadline with distance>=radius"

    nat_path = os.path.join(d, "test_natural.npz")
    if os.path.exists(nat_path):
        z = np.load(nat_path, allow_pickle=True)
        Xn = z["feats"].astype(np.float32); Mn = list(z["meta"])
        results["natural"] = {"episodes": len(Mn),
                              "counts": {c: sum(1 for m in Mn if m["agent_type"] == c) for c in CLASSES},
                              "widths": {}}
        for hidden in WIDTHS:
            key = f"{hidden[0]}/{hidden[1]}"
            per_seed = {}
            for seed in TSEEDS:
                th = thresholds[(hidden, seed)]
                probs = probs_episodes(models[(hidden, seed)], Xn, mu, sd)
                masks = [probs[i] >= th for i in range(len(probs))]
                per_seed[seed] = {c: metrics(Mn, c, masks) for c in CLASSES}
            results["natural"]["widths"][key] = per_seed
        results["natural"]["rule"] = {c: metrics(Mn, c, rule_masks(Xn)) for c in CLASSES}
    np.savez(os.path.join(d, "predictions.npz"), **preds_dump)
    torch.save({"mu": mu, "sd": sd, "widths": WIDTHS,
                "state_dicts": {f"{h[0]}_{h[1]}_{s}": models[(h, s)].state_dict()
                                for h in WIDTHS for s in TSEEDS},
                "thresholds": {f"{h[0]}_{h[1]}_{s}": thresholds[(h, s)]
                               for h in WIDTHS for s in TSEEDS}},
               os.path.join(d, "model.pt"))
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
