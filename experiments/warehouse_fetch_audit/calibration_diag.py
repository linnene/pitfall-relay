"""Diagnose the old Fetch-probe calibration without selecting a new threshold."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# The frozen probe code lives in the archived variant worktree.
sys.path.insert(0, "/data/amgg_flex_demo/flex-warehouse-handoff-variant-20261118-a92b6f31/FLEX_eval/experiments/warehouse_safe")
import probe_latent as pl  # noqa: E402


def per_episode_max(samples, score):
    by_ep = defaultdict(list)
    for s in samples:
        by_ep[s["episode"]].append(s)
    return {ep: float(np.max(score(rows))) for ep, rows in sorted(by_ep.items())}


def summarize_threshold(maxima, threshold):
    vals = np.asarray(list(maxima.values()), dtype=float)
    return {
        "episodes": int(len(vals)),
        "below": int((vals < threshold).sum()),
        "equal": int((vals == threshold).sum()),
        "above": int((vals > threshold).sum()),
        "fpr_ge": float((vals >= threshold).mean()) if len(vals) else None,
        "fpr_gt": float((vals > threshold).mean()) if len(vals) else None,
        "min": float(vals.min()) if len(vals) else None,
        "median": float(np.median(vals)) if len(vals) else None,
        "max": float(vals.max()) if len(vals) else None,
    }


def order_threshold(maxima, q=0.95):
    vals = np.sort(np.asarray(list(maxima.values()), dtype=float))
    n = len(vals)
    k = math.ceil((n + 1) * q)
    return {
        "n": n,
        "k": k,
        "threshold": float(vals[k - 1]) if k <= n else None,
        "beyond_n": bool(k > n),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    split_map = json.loads(Path(args.split_file).read_text())
    split = split_map[args.eval_dir]["split"]
    episodes = pl.load_eval(args.eval_dir)
    by_ep = {int(ep["meta"]["episode"]): ep for ep in episodes}
    train_eps = [by_ep[i] for i in split["train"]]
    cal_eps = [by_ep[i] for i in split["cal"]]
    test_eps = [by_ep[i] for i in split["test"]]
    train, train_events = pl.build_samples(train_eps, 2, "fetch_handoff_failure", window=20)
    cal, cal_events = pl.build_samples(cal_eps, 2, "fetch_handoff_failure", window=20)
    test, test_events = pl.build_samples(test_eps, 2, "fetch_handoff_failure", window=20)
    for part in (train, cal, test):
        sh = pl.shuffled_latents(part, args.seed)
        for sample, latent in zip(part, sh):
            sample["latent_shuffled"] = latent
    normal_cal = [s for s in cal if s["label"] == 0 and not any(
        e["_episode_index"] == s["episode"] for e in cal_events)]
    normal_test = [s for s in test if s["label"] == 0 and not any(
        e["_episode_index"] == s["episode"] for e in test_events)]
    out = {
        "eval_dir": args.eval_dir,
        "seed": args.seed,
        "counts": {
            "train_episodes": len(train_eps), "cal_episodes": len(cal_eps),
            "test_episodes": len(test_eps),
            "train_samples": len(train), "cal_samples": len(cal), "test_samples": len(test),
            "normal_cal_episodes": len({s["episode"] for s in normal_cal}),
            "normal_test_episodes": len({s["episode"] for s in normal_test}),
            "test_window_events": sum(1 for e in test_events if e.get("window")),
            "test_events": len(test_events),
        },
        "split_disjoint": not (set(split["train"]) & set(split["cal"])
                               or set(split["train"]) & set(split["test"])
                               or set(split["cal"]) & set(split["test"])),
        "normal_episode_overlap": bool({s["episode"] for s in normal_cal}
                                       & {s["episode"] for s in normal_test}),
        "arms": {},
    }
    for arm in ("baseline", "real", "shuffled"):
        score, _ = pl.fit_predict(train, cal, test, arm, seed=args.seed)
        cal_max = per_episode_max(normal_cal, score)
        test_max = per_episode_max(normal_test, score)
        old_thr = float(np.quantile(np.asarray(list(cal_max.values())), 0.95,
                                    interpolation="higher")) if cal_max else None
        order = order_threshold(cal_max)
        cal_summary = summarize_threshold(cal_max, old_thr) if old_thr is not None else None
        test_summary = summarize_threshold(test_max, old_thr) if old_thr is not None else None
        all_scores = np.asarray([score([s])[0] for s in (normal_cal + normal_test)]) if (normal_cal or normal_test) else np.asarray([])
        out["arms"][arm] = {
            "old_quantile_interpolation_higher": old_thr,
            "order_statistic": order,
            "cal_max": cal_summary,
            "test_max": test_summary,
            "score_precision": {
                "min": float(all_scores.min()) if all_scores.size else None,
                "max": float(all_scores.max()) if all_scores.size else None,
                "mean": float(all_scores.mean()) if all_scores.size else None,
                "n_at_0": int((all_scores == 0.0).sum()),
                "n_at_1": int((all_scores == 1.0).sum()),
            },
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
