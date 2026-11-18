"""Aggregate fixed-actor evaluation and paired initial-vs-final differences."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_summary(path):
    payload = json.loads(Path(path).read_text())
    return payload


def paired_ci(values, rng, n_boot=2000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    draws = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"mean": float(np.mean(values)), "ci_low": float(lo), "ci_high": float(hi), "n": int(len(values))}


def summarize(summary):
    eps = summary["episodes"]
    return {
        "label": summary.get("label"),
        "training_seed": summary.get("training_seed"),
        "eval_seed_base": summary.get("eval_seed"),
        "episodes": len(eps),
        "successes": sum(1 for e in eps if e["success"]),
        "success_rate": float(np.mean([e["success"] for e in eps])) if eps else None,
        "mean_steps": float(np.mean([e["steps"] for e in eps])) if eps else None,
        "mean_steps_success": float(np.mean([e["steps"] for e in eps if e["success"]]))
        if any(e["success"] for e in eps) else None,
        "mean_return": float(np.mean([e["undiscounted_return"] for e in eps])) if eps else None,
        "event_counts": summary.get("event_counts"),
        "human_finish_steps": {
            str(i): [e.get("human_finish_steps", [None, None])[i] for e in eps]
            for i in range(2)
        },
        "human_finish_mean": {
            str(i): float(np.mean([e["human_finish_steps"][i] for e in eps
                                   if e.get("human_finish_steps", [None, None])[i] is not None]))
            if any(e.get("human_finish_steps", [None, None])[i] is not None for e in eps) else None
            for i in range(2)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", action="append", required=True,
                        help="label=path/to/summary.json; repeat per seed")
    parser.add_argument("--final", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    initial = {}
    final = {}
    for item in args.initial:
        label, path = item.split("=", 1)
        initial[label] = load_summary(path)
    for item in args.final:
        label, path = item.split("=", 1)
        final[label] = load_summary(path)
    rng = np.random.RandomState(20261118)
    result = {"initial": {}, "final": {}, "paired": {}}
    for label in sorted(initial):
        result["initial"][label] = summarize(initial[label])
    for label in sorted(final):
        result["final"][label] = summarize(final[label])
    for label in sorted(set(initial) & set(final)):
        a = initial[label]
        b = final[label]
        by_seed_a = {e["seed"]: e for e in a["episodes"]}
        by_seed_b = {e["seed"]: e for e in b["episodes"]}
        seeds = sorted(set(by_seed_a) & set(by_seed_b))
        succ_diff = [int(by_seed_b[s]["success"]) - int(by_seed_a[s]["success"]) for s in seeds]
        step_diff = [by_seed_b[s]["steps"] - by_seed_a[s]["steps"] for s in seeds]
        both_success = [s for s in seeds if by_seed_a[s]["success"] and by_seed_b[s]["success"]]
        step_success_diff = [by_seed_b[s]["steps"] - by_seed_a[s]["steps"] for s in both_success]
        ret_diff = [by_seed_b[s]["undiscounted_return"] - by_seed_a[s]["undiscounted_return"] for s in seeds]
        result["paired"][label] = {
            "paired_episodes": len(seeds),
            "success_difference": paired_ci(succ_diff, rng),
            "steps_difference_final_minus_initial": paired_ci(step_diff, rng),
            "success_pairs": len(both_success),
            "steps_difference_success_pairs_final_minus_initial": paired_ci(step_success_diff, rng),
            "return_difference_final_minus_initial": paired_ci(ret_diff, rng),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True)[:4000])


if __name__ == "__main__":
    main()
