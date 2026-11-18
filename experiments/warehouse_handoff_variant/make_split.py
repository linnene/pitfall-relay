"""Stratify variant probe episodes by stall duration and window presence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def stable(seed, salt=20261118):
    text = f"{seed}|{salt}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def load_meta(eval_dir):
    rows = []
    for path in sorted((Path(eval_dir) / "episodes").glob("*.json")):
        meta = json.loads(path.read_text())
        stall = meta.get("stall_intervals", [])
        total = sum(int(b) - int(a) + 1 for a, b in stall)
        rows.append({
            "episode": int(meta["episode"]),
            "seed": int(meta["seed"]),
            "stall_total": total,
            "window": int(meta.get("window_count", 0) > 0),
            "failure": int(meta.get("stall_failure_count", 0)),
        })
    return rows


def allocate(rows, counts):
    total_target = sum(counts.values())
    if len(rows) != total_target:
        raise ValueError(f"episodes {len(rows)} != targets {total_target}")
    # Stall duration bins; window presence is a second stratum.
    bins = [0, 15, 21, 27, 10**9]
    strata = {}
    for row in rows:
        b = int(np.searchsorted(bins, row["stall_total"], side="right") - 1)
        strata.setdefault((b, row["window"]), []).append(row)
    split = {name: [] for name in counts}
    names = ["train", "cal", "test"]
    # Largest-remainder allocation per stratum, then fill by target counts.
    for key in sorted(strata):
        items = sorted(strata[key], key=lambda r: stable(r["seed"]))
        n = len(items)
        quotas = {name: counts[name] * n / total_target for name in names}
        assigned = {name: int(np.floor(quotas[name])) for name in names}
        remainder = n - sum(assigned.values())
        order = sorted(names, key=lambda name: quotas[name] - assigned[name], reverse=True)
        for name in order[:remainder]:
            assigned[name] += 1
        pos = 0
        for name in names:
            split[name].extend(item["episode"] for item in items[pos:pos + assigned[name]])
            pos += assigned[name]
    # Enforce exact totals by moving from the largest split.
    for name in names:
        while len(split[name]) > counts[name]:
            moved = split[name].pop()
            other = min(names, key=lambda x: len(split[x]) / counts[x])
            split[other].append(moved)
    for name in names:
        while len(split[name]) < counts[name]:
            other = max(names, key=lambda x: len(split[x]) - counts[x])
            split[name].append(split[other].pop())
    return {name: sorted(set(split[name])) for name in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", action="append", required=True)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--cal-count", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=1500)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    counts = {"train": args.train_count, "cal": args.cal_count, "test": args.test_count}
    payload = {}
    for eval_dir in args.eval_dir:
        rows = load_meta(eval_dir)
        split = allocate(rows, counts)
        meta = {r["episode"]: r for r in rows}
        payload[eval_dir] = {
            "split": split,
            "counts": {k: len(v) for k, v in split.items()},
            "window_events": {k: sum(meta[i]["window"] for i in v) for k, v in split.items()},
            "normal_episodes": {k: sum(1 - meta[i]["window"] and meta[i]["failure"] == 0
                                       for i in v) for k, v in split.items()},
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
