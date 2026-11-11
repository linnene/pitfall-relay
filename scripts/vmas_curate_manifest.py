#!/usr/bin/env python3
"""Zero-step curation of existing VMAS collections.

Builds episode-bounded event tables, cuts post-terminal frames, assigns labels
with explicit 'unclassified', and writes a reviewable manifest.

Episode boundary: collectors reset manually every ep_len steps; VMAS does not
auto-reset on done. Valid frames per episode end at (and include) the first
done frame; frames after it are post-terminal and never used.
"""
import argparse
import json
import os

import numpy as np

PRE = 16   # required valid pre-event frames
POST = 2   # included post-event frames (may be cut at terminal)


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files if k != "meta"}, json.loads(str(d["meta"]))


def episodes(T, ep_len):
    return T // ep_len


def first_done(done_seg):
    idx = np.flatnonzero(done_seg)
    return int(idx[0]) if len(idx) else None


def window(ep_start, ep_len, event_t, terminal):
    """Valid frame window [event_t-PRE, event_t+POST] inside the episode."""
    lo = event_t - PRE
    hi = event_t + POST
    if lo < 0:
        return None, "insufficient_prehistory"
    if terminal is not None and hi > terminal:
        hi = terminal
    return (lo, hi), None


def emit(rows, rejected, rec):
    if rec.get("reject"):
        rejected.append(rec)
    else:
        rows.append(rec)


def curate_navigation(path, seed, ep_len, split_envs, rows, rejected):
    d, meta = load(path)
    T, B, A = d["collision"].shape
    lidar = d["obs"][..., 6:18].max(-1)
    E = episodes(T, ep_len)
    hard_miss = 0
    for e in range(E):
        for b in range(B):
            base = e * ep_len
            done_seg = d["done"][base:base + ep_len, b]
            terminal = first_done(done_seg)
            for a in range(A):
                flags = (d["collision"][base:base + ep_len, b, a] < 0)
                t = 0
                while t < len(flags):
                    if not flags[t]:
                        t += 1
                        continue
                    s = t
                    while t + 1 < len(flags) and flags[t + 1]:
                        t += 1
                    end = t
                    if terminal is not None and s > terminal:
                        t += 1
                        continue
                    win, why = window(0, ep_len, s, terminal)
                    rec = {
                        "scenario": "navigation", "seed": seed, "env": b,
                        "split": "dev" if b in split_envs else "heldout",
                        "episode": e, "event_start": base + s, "event_end": base + end,
                        "label": "collision", "window": [base + win[0], base + win[1]] if win else None,
                        "rule_flag": bool(lidar[base + s, b, a] >= 0.25),
                        "local_fields": ["obs[16]", "action[2]"],
                        "visible": "lidar proximity, own pos/vel, goal rel",
                        "paired_with": "same layout, no-collision episode",
                        "terminal": bool(terminal is not None),
                    }
                    if why:
                        rec["reject"] = why
                    elif not rec["rule_flag"]:
                        hard_miss += 1
                    emit(rows, rejected, rec)
                    t += 1
    return {"scenario": "navigation", "file": path, "episodes_envs": E * B,
            "terminated": int(sum(first_done(d["done"][e * ep_len:(e + 1) * ep_len, b]) is not None
                                  for e in range(E) for b in range(B))),
            "hard_rule_misses": hard_miss}


def curate_balance(path, seed, ep_len, split_envs, rows, rejected):
    d, meta = load(path)
    T, B = d["done"].shape
    E = episodes(T, ep_len)
    for e in range(E):
        for b in range(B):
            base = e * ep_len
            done_seg = d["done"][base:base + ep_len, b]
            terminal = first_done(done_seg)
            if terminal is None:
                continue
            n = base + terminal
            ground = float(d["ground_rew"][n, b, 0])
            if ground >= 0:
                continue
            lg = float(d["line_floor_gap"][n, b]); pg = float(d["pkg_floor_gap"][n, b])
            cause = "line" if lg <= 0.005 and pg > 0.005 else "package" if pg <= 0.005 and lg > 0.005 else "both" if lg <= 0.005 else "unclassified"
            win, why = window(0, ep_len, terminal, terminal)
            rec = {"scenario": "balance", "seed": seed, "env": b,
                   "split": "dev" if b in split_envs else "heldout",
                   "episode": e, "event_start": n, "event_end": n,
                   "label": "drop", "drop_cause": cause,
                   "window": [base + win[0], base + win[1]] if win else None,
                   "rule_flag": bool(abs(float(d["obs"][n, b, 0, 15])) >= 0.4),
                   "local_fields": ["obs[16]", "action[2]"],
                   "visible": "own kinematics, package/line relative, line rot",
                   "paired_with": "no-drop episode, same seed",
                   "terminal": True}
            if why:
                rec["reject"] = why
            emit(rows, rejected, rec)
    return {"scenario": "balance", "file": path, "episodes_envs": E * B}


def curate_give_way(path, seed, ep_len, split_envs, rows, rejected):
    d, meta = load(path)
    T, B, A = d["agent_collision"].shape
    E = episodes(T, ep_len)
    for e in range(E):
        for b in range(B):
            base = e * ep_len
            for a in range(A):
                mate = (d["agent_collision"][base:base + ep_len, b, a] < 0)
                obst = (d["obstacle_collision"][base:base + ep_len, b, a] < 0)
                for name, flags in (("mate_contact", mate), ("obstacle_contact", obst)):
                    t = 0
                    while t < len(flags):
                        if not flags[t]:
                            t += 1
                            continue
                        s = t
                        while t + 1 < len(flags) and flags[t + 1]:
                            t += 1
                        end = t
                        win, why = window(0, ep_len, s, None)
                        rec = {"scenario": "give_way", "seed": seed, "env": b,
                               "split": "dev" if b in split_envs else "heldout",
                               "episode": e, "event_start": base + s, "event_end": base + end,
                               "label": name,
                               "window": [base + win[0], base + win[1]] if win else None,
                               "rule_flag": bool(np.linalg.norm(d["obs"][base + s, b, a, 4:6]) <= 0.30),
                               "local_fields": ["obs[6]", "action[2]"],
                               "visible": "own pos/vel, teammate relative position",
                               "paired_with": "same episode, no-contact segment",
                               "terminal": False}
                        if why:
                            rec["reject"] = why
                        emit(rows, rejected, rec)
                        t += 1
    return {"scenario": "give_way", "file": path, "episodes_envs": E * B}


def curate_football(path, seed, ep_len, split_envs, rows, rejected):
    d, meta = load(path)
    T, B = d["done"].shape
    E = episodes(T, ep_len)
    for e in range(E):
        for b in range(B):
            base = e * ep_len
            done_seg = d["done"][base:base + ep_len, b]
            terminal = first_done(done_seg)
            if terminal is None:
                continue
            n = base + terminal
            sparse = float(d["sparse"][n, b, 0])
            if sparse >= 0:
                continue
            win, why = window(0, ep_len, terminal, terminal)
            rec = {"scenario": "football", "seed": seed, "env": b,
                   "split": "dev" if b in split_envs else "heldout",
                   "episode": e, "event_start": n, "event_end": n,
                   "label": "conceded",
                   "window": [base + win[0], base + win[1]] if win else None,
                   "rule_flag": bool(float(d["ball_pos"][n, b, 0]) + 1.5 <= 0.1),
                   "local_fields": ["obs[56]", "action[2]"],
                   "visible": "ball pos/vel, teammates/adversaries relative",
                   "paired_with": "near-goal episode without concession",
                   "terminal": True}
            if why:
                rec["reject"] = why
            emit(rows, rejected, rec)
    return {"scenario": "football", "file": path, "episodes_envs": E * B}


def add_negatives(rows, nav_pairs):
    """Near-miss / normal / unclassified windows as separate samples."""
    for path, seed, ep_len, split_envs in nav_pairs:
        d, meta = load(path)
        T, B = d["done"].shape[:2]
        gap = d["gap"]; coll = d["collision"] < 0
        E = episodes(T, ep_len)
        for e in range(E):
            for b in range(B):
                base = e * ep_len
                term = first_done(d["done"][base:base + ep_len, b])
                for t in range(PRE, ep_len - POST):
                    if term is not None and t + POST > term:
                        continue
                    if coll[base + t, b].any():
                        continue
                    g = float(gap[base + t, b].min())
                    label = "near" if g <= 0.05 else "normal" if g > 0.15 else None
                    if label is None:
                        continue
                    rows.append({"scenario": "navigation", "seed": seed, "env": b,
                                 "split": "dev" if b in split_envs else "heldout",
                                 "episode": e, "event_start": base + t, "event_end": base + t,
                                 "label": label, "window": [base + t - PRE, base + t + POST],
                                 "rule_flag": False, "local_fields": ["obs[16]", "action[2]"],
                                 "visible": "lidar, own state", "paired_with": None,
                                 "terminal": False})


def add_nav_success(rows, nav_pairs):
    """Terminated navigation episodes are team goal completions (success)."""
    for path, seed, ep_len, split_envs in nav_pairs:
        d, meta = load(path)
        T, B = d["done"].shape[:2]
        coll = d["collision"] < 0
        E = episodes(T, ep_len)
        for e in range(E):
            for b in range(B):
                base = e * ep_len
                term = first_done(d["done"][base:base + ep_len, b])
                if term is None or term < PRE:
                    continue
                had_coll = bool(coll[base:base + term + 1, b].any())
                rows.append({"scenario": "navigation", "seed": seed, "env": b,
                             "split": "dev" if b in split_envs else "heldout",
                             "episode": e, "event_start": base + term, "event_end": base + term,
                             "label": "success_with_collision" if had_coll else "success_no_collision",
                             "window": [base + term - PRE, base + term],
                             "rule_flag": False, "local_fields": ["obs[16]", "action[2]"],
                             "visible": "lidar, own state", "paired_with": "same layout, failed episode",
                             "terminal": True})


def add_scenario_negatives(rows, multi_root):
    # balance near frames
    for s in (301, 302):
        d, _ = load(f"{multi_root}/balance_seed{s}.npz")
        T, B = d["done"].shape
        gap = np.minimum(d["pkg_floor_gap"], d["line_floor_gap"])
        for e in range(T // 128):
            for b in range(B):
                base = e * 128
                term = first_done(d["done"][base:base + 128, b])
                for t in range(PRE, 128 - POST):
                    if term is not None and t + POST > term:
                        continue
                    g = float(gap[base + t, b])
                    if g >= 0.03 and g <= 0.08:
                        continue  # unclassified band
                    if g < 0.03:
                        label = "near"
                    else:
                        label = "normal"
                    rows.append({"scenario": "balance", "seed": s, "env": b,
                                 "split": "dev" if b < 4 else "heldout",
                                 "episode": e, "event_start": base + t, "event_end": base + t,
                                 "label": label, "window": [base + t - PRE, base + t + POST],
                                 "rule_flag": bool(abs(float(d["obs"][base + t, b, 0, 15])) >= 0.4),
                                 "local_fields": ["obs[16]", "action[2]"],
                                 "visible": "own kinematics, package/line relative, line rot",
                                 "paired_with": None, "terminal": False})
    # football near / normal
    for s in (301, 302):
        d, _ = load(f"{multi_root}/football_seed{s}.npz")
        T, B = d["done"].shape
        for e in range(T // 256):
            for b in range(B):
                base = e * 256
                term = first_done(d["done"][base:base + 256, b])
                for t in range(PRE, 256 - POST):
                    if term is not None and t + POST > term:
                        continue
                    x = float(d["ball_pos"][base + t, b, 0])
                    if x < -1.0:
                        label = "near"
                    elif x > -0.3:
                        label = "normal"
                    else:
                        continue  # unclassified band
                    rows.append({"scenario": "football", "seed": s, "env": b,
                                 "split": "dev" if b < 4 else "heldout",
                                 "episode": e, "event_start": base + t, "event_end": base + t,
                                 "label": label, "window": [base + t - PRE, base + t + POST],
                                 "rule_flag": bool(x + 1.5 <= 0.1),
                                 "local_fields": ["obs[56]", "action[2]"],
                                 "visible": "ball pos/vel, teammates/adversaries",
                                 "paired_with": None, "terminal": False})
    # give_way near / normal (per agent, no contact)
    for s in (301, 302):
        d, _ = load(f"{multi_root}/give_way_seed{s}.npz")
        T, B, A = d["agent_collision"].shape
        for e in range(T // 128):
            for b in range(B):
                for a in range(A):
                    base = e * 128
                    mate = d["agent_collision"][base:base + 128, b, a] < 0
                    obst = d["obstacle_collision"][base:base + 128, b, a] < 0
                    rel = np.linalg.norm(d["obs"][base:base + 128, b, a, 4:6], axis=-1)
                    for t in range(PRE, 128 - POST):
                        if mate[t] or obst[t]:
                            continue
                        if 0.30 < rel[t] < 0.40:
                            label = "near"
                        elif rel[t] >= 0.40:
                            label = "normal"
                        else:
                            continue
                        rows.append({"scenario": "give_way", "seed": s, "env": b,
                                     "split": "dev" if b < 3 else "heldout",
                                     "episode": e, "event_start": base + t, "event_end": base + t,
                                     "label": label, "window": [base + t - PRE, base + t + POST],
                                     "rule_flag": bool(rel[t] <= 0.30),
                                     "local_fields": ["obs[6]", "action[2]"],
                                     "visible": "own pos/vel, teammate relative position",
                                     "paired_with": None, "terminal": False})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nav-root", required=True)
    p.add_argument("--nav-cross-root", required=True)
    p.add_argument("--multi-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows, rejected = [], []
    stats = []
    nav = [(f"{args.nav_root}/nav_seed{s}.npz", s, 128, set(range(0, 4))) for s in (101, 202)]
    nav_cross = [(f"{args.nav_cross_root}/nav_seed{s}.npz", s, 80, set(range(0, 4))) for s in (101, 202)]
    for path, seed, ep, dev in nav + nav_cross:
        stats.append(curate_navigation(path, seed, ep, dev, rows, rejected))
    add_negatives(rows, nav + nav_cross)
    add_nav_success(rows, nav + nav_cross)
    add_scenario_negatives(rows, args.multi_root)
    for s in (301, 302):
        stats.append(curate_balance(f"{args.multi_root}/balance_seed{s}.npz", s, 128, set(range(0, 4)), rows, rejected))
        stats.append(curate_give_way(f"{args.multi_root}/give_way_seed{s}.npz", s, 128, set(range(0, 3)), rows, rejected))
        stats.append(curate_football(f"{args.multi_root}/football_seed{s}.npz", s, 256, set(range(0, 4)), rows, rejected))

    summary = {"raw_events": {}, "curated_events": {}, "rejected": {}, "stats": stats}
    for r in rows + rejected:
        key = (r["scenario"], r["label"])
        summary["raw_events"]["/".join(key)] = summary["raw_events"].get("/".join(key), 0) + 1
        if r in rejected:
            k = "/".join(key)
            summary["rejected"][k] = summary["rejected"].get(k, 0) + 1
        else:
            k = "/".join(key)
            summary["curated_events"][k] = summary["curated_events"].get(k, 0) + 1

    with open(f"{args.out}/manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{args.out}/rejected.jsonl", "w") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
