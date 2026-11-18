"""Compare hook-on and hook-off evaluation runs for the same actor/seeds."""

import argparse
import json
from pathlib import Path

import numpy as np


def load(dir_path):
    root = Path(dir_path)
    items = {}
    for path in sorted(root.glob("episodes/*.json")):
        meta = json.loads(path.read_text())
        items[meta["episode"]] = {
            "meta": meta,
            "arrays": np.load(path.with_suffix(".npz")),
        }
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--on-dir", required=True)
    parser.add_argument("--off-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    on = load(args.on_dir)
    off = load(args.off_dir)
    episodes = sorted(set(on) & set(off))
    mismatches = []
    results = []
    for ep in episodes:
        a, b = on[ep], off[ep]
        same_actions = np.array_equal(a["arrays"]["selected_action"], b["arrays"]["selected_action"])
        same_actual = np.array_equal(a["arrays"]["actual_action"], b["arrays"]["actual_action"])
        same_success = a["meta"]["success"] == b["meta"]["success"]
        same_steps = a["meta"]["steps"] == b["meta"]["steps"]
        same_returns = (abs(a["meta"]["undiscounted_return"] - b["meta"]["undiscounted_return"]) < 1e-9)
        same_events = a["meta"]["event_counts"] == b["meta"]["event_counts"]
        row = {
            "episode": ep,
            "seed_on": a["meta"]["seed"],
            "seed_off": b["meta"]["seed"],
            "same_seed": a["meta"]["seed"] == b["meta"]["seed"],
            "same_actions": same_actions,
            "same_actual_actions": same_actual,
            "same_success": same_success,
            "same_steps": same_steps,
            "same_return": same_returns,
            "same_event_counts": same_events,
            "success_on": a["meta"]["success"],
            "success_off": b["meta"]["success"],
        }
        results.append(row)
        if not all(v for k, v in row.items() if k.startswith("same_")):
            mismatches.append(row)
    summary = {
        "on_dir": args.on_dir,
        "off_dir": args.off_dir,
        "episodes": len(episodes),
        "all_actions_equal": all(r["same_actions"] and r["same_actual_actions"] for r in results),
        "all_results_equal": all(r["same_success"] and r["same_steps"] and r["same_return"] for r in results),
        "mismatches": mismatches,
        "per_episode": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({k: summary[k] for k in ("episodes", "all_actions_equal", "all_results_equal")}, indent=2))


if __name__ == "__main__":
    main()
