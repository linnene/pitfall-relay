#!/usr/bin/env python3
"""Three-class training/evaluation for navigation self-failure detection.

Reuses collection from nav_selfdet.py. Classes per step:
  0 success (already entered), 1 in-progress, 2 failure confirmed at deadline.
"""
import argparse
import json
import os

import numpy as np

import nav_selfdet as ns


def subset(x, keys, eps):
    idx = [i for i, (e, t) in enumerate(keys) if e in eps]
    return x[idx], [keys[i] for i in idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    data, meta = ns.load(args.dir)
    splits = ns.split_episodes(meta)
    test_eps = splits["test_seen"] + splits["test_holdout"]

    feats = {
        "win": ns.build_features3(data, meta, history=False),
        "hist": ns.build_features3(data, meta, history=True),
    }
    x_ng, y_ng, keys_ng = ns.build_features3_nogoal(data, meta)

    results = {}
    models = {}
    train_set = set(splits["train"])
    for seed in (0, 1, 2):
        for name in ("win", "hist"):
            x, y, keys = feats[name]
            tr_x, _ = subset(x, keys, train_set)
            tr_y = np.array([y[i] for i, (e, t) in enumerate(keys) if e in train_set])
            mu = tr_x.mean(0); sd = tr_x.std(0); sd[sd < 1e-6] = 1.0
            model = ns.train_mlp3(((tr_x - mu) / sd).astype(np.float32), tr_y, seed)
            models[name] = (model, mu, sd)
            te_x, te_keys = subset(x, keys, set(test_eps))
            pred = ns.predict3(model, ((te_x - mu) / sd).astype(np.float32))
            pred_by_key = dict(zip(te_keys, pred))
            results[f"{name}_{seed}"] = {
                "test_all": ns.evaluate3(pred_by_key, meta, test_eps),
                "test_seen": ns.evaluate3(pred_by_key, meta, splits["test_seen"]),
                "test_holdout": ns.evaluate3(pred_by_key, meta, splits["test_holdout"]),
            }

    rule = ns.rule_predictions(data, meta, test_eps)
    results["rule"] = {"test_all": ns.evaluate3(rule, meta, test_eps),
                       "test_seen": ns.evaluate3(rule, meta, splits["test_seen"]),
                       "test_holdout": ns.evaluate3(rule, meta, splits["test_holdout"])}

    deadline_only = {}
    for e in test_eps:
        for t in range(ns.WINDOW - 1, ns.DEADLINE):
            deadline_only[(e, t)] = 2 if t == ns.DEADLINE - 1 else 1
    results["deadline_only"] = {
        "test_all": ns.evaluate3(deadline_only, meta, test_eps),
        "test_seen": ns.evaluate3(deadline_only, meta, splits["test_seen"]),
        "test_holdout": ns.evaluate3(deadline_only, meta, splits["test_holdout"])}

    tr_x, _ = subset(x_ng, keys_ng, train_set)
    tr_y = np.array([y_ng[i] for i, (e, t) in enumerate(keys_ng) if e in train_set])
    mu2 = tr_x.mean(0); sd2 = tr_x.std(0); sd2[sd2 < 1e-6] = 1.0
    m2 = ns.train_mlp3(((tr_x - mu2) / sd2).astype(np.float32), tr_y, 0)
    te_x, te_keys = subset(x_ng, keys_ng, set(test_eps))
    p2 = ns.predict3(m2, ((te_x - mu2) / sd2).astype(np.float32))
    results["nogoal_0"] = {"test_all": ns.evaluate3(dict(zip(te_keys, p2)), meta, test_eps)}

    # target-swap counterfactual on the hist model (offline relabeling)
    model, mu, sd = models["hist"]
    cf_x, cf_keys, cf_labels, orig_y = [], [], [], []
    pos, obs, act = data["pos"], data["obs"], data["act"]
    ep_info = {e["episode"]: e for e in meta["episodes"]}
    for e in test_eps:
        g0 = np.array(ep_info[e]["goal"], dtype=np.float32)
        g1 = np.array([-g0[0], g0[1]], dtype=np.float32)  # reflect to the other x half
        rel1 = pos[:, e] - g1[None, :]
        entered1 = (np.linalg.norm(rel1, axis=-1) < ns.RADIUS)
        arr1 = int(np.argmax(entered1)) if entered1.any() else -1
        y0 = 2 if ep_info[e]["arrived_t"] < 0 else 0
        y1 = 2 if arr1 < 0 else 0
        if y0 != y1:
            orig_y.append(y0)
            for t in range(ns.WINDOW - 1, ns.DEADLINE):
                win = obs[t - ns.WINDOW + 1:t + 1, e].copy()
                win[:, 4:6] = rel1[t - ns.WINDOW + 1:t + 1]
                flat = np.concatenate([win.reshape(-1), act[t - ns.WINDOW + 1:t + 1, e].reshape(-1)])
                clock = np.array([t / (ns.DEADLINE - 1)], dtype=np.float32)
                gv = np.array([g1[0], g1[1], ns.RADIUS, ns.DEADLINE], dtype=np.float32)
                hist = np.array([np.linalg.norm(rel1[:t + 1], axis=-1).min()], dtype=np.float32)
                cf_x.append(np.concatenate([flat, gv, clock, hist]))
                cf_keys.append((e, t))
    cf_x = ((np.stack(cf_x) - mu) / sd).astype(np.float32)
    cf_pred = ns.predict3(model, cf_x)
    cf_map = dict(zip(cf_keys, cf_pred))
    flips = 0; ok = 0
    for e, y0 in zip([k[0] for k in [(e, 0) for e in test_eps]], []):
        pass
    flips = len(orig_y)
    for e in test_eps:
        key = (e, ns.DEADLINE - 1)
        if key not in cf_map:
            continue
        g0 = np.array(ep_info[e]["goal"], dtype=np.float32)
        g1 = np.array([-g0[0], g0[1]], dtype=np.float32)
        rel1 = pos[:, e] - g1[None, :]
        y1 = 2 if not (np.linalg.norm(rel1, axis=-1) < ns.RADIUS).any() else 0
        y0 = 2 if ep_info[e]["arrived_t"] < 0 else 0
        if y0 != y1:
            ok += int(cf_map[key] == y1)
    results["counterfactual"] = {"flips": flips, "flip_accuracy": ok / max(flips, 1)}

    strat = {"late_arrival": 0, "leave_after": 0, "near_boundary_fail": 0}
    for e in test_eps:
        info = ep_info[e]
        if 0 <= info["arrived_t"] >= ns.DEADLINE - 10:
            strat["late_arrival"] += 1
        if info["arrived_t"] >= 0:
            g = np.array(info["goal"])
            if float(np.linalg.norm(data["pos"][-1, e] - g)) > 0.2:
                strat["leave_after"] += 1
        if info["arrived_t"] < 0 and info["min_dist"] <= ns.RADIUS + 0.10:
            strat["near_boundary_fail"] += 1
    results["strata"] = strat
    results["splits"] = {k: len(v) for k, v in splits.items()}
    results["test_failure_episodes"] = int(sum(1 for e in test_eps
                                               if ep_info[e]["arrived_t"] < 0))

    out = args.json_out or os.path.join(args.dir, "results3.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    for k, v in results.items():
        if isinstance(v, dict) and "test_all" in v:
            a = v["test_all"]; b = v["test_seen"]; c = v["test_holdout"]
            print("%-10s miss=%.3f alarm=%.3f inprog=%.3f | seen miss=%.3f alarm=%.3f | hold miss=%.3f alarm=%.3f" % (
                k, a["failure_miss_rate"], a["predeadline_alarm_rate"], a["inprogress_recall"],
                b["failure_miss_rate"], b["predeadline_alarm_rate"],
                c["failure_miss_rate"], c["predeadline_alarm_rate"]))
    print(json.dumps({"strata": strat, "splits": results["splits"],
                      "test_failure_episodes": results["test_failure_episodes"],
                      "counterfactual": results["counterfactual"]}))


if __name__ == "__main__":
    main()
