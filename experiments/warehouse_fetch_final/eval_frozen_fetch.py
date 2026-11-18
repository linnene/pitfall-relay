"""Evaluate the frozen Fetch probe once on a fresh holdout view.

Loads the frozen models/thresholds produced before collection and never fits
or adjusts anything.  Alarm rule is the pre-registered strict ``score >
threshold`` with the ceil((n+1)*0.95) order-statistic threshold.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_probe_dir = os.environ.get("PROBE_CODE_DIR")
sys.path.insert(0, _probe_dir if _probe_dir else str(HERE.parent / "warehouse_safe"))
import probe_latent as pl  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen(frozen_dir):
    frozen_dir = Path(frozen_dir)
    threshold_data = json.loads((frozen_dir / "thresholds.json").read_text())
    model_path = frozen_dir / "models.pkl"
    model_sha = sha256_file(model_path)
    if model_sha != threshold_data["model_sha256"]:
        raise SystemExit("frozen model hash does not match thresholds.json")
    with open(model_path, "rb") as handle:
        arms = pickle.load(handle)
    return arms, threshold_data, model_sha, sha256_file(frozen_dir / "thresholds.json")


def make_score(arms, arm):
    frozen = arms[arm]

    def score(rows):
        X = np.stack([s["features"] for s in rows])
        if arm == "real":
            X = np.concatenate([X, np.stack([s["latent"] for s in rows])], axis=1)
        elif arm == "shuffled":
            X = np.concatenate([X, np.stack([s["latent_shuffled"] for s in rows])], axis=1)
        return frozen["model"].predict_proba(frozen["scaler"].transform(X))[:, 1]

    return score


def bootstrap_fpr(alarms, seed=5151, draws=1000):
    values = np.asarray(alarms, dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.RandomState(seed)
    means = [float(values[rng.choice(len(values), size=len(values), replace=True)].mean())
             for _ in range(draws)]
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n_episodes": int(len(values))}


def bootstrap_fpr_diff(real_alarms, control_alarms, seed=5252, draws=1000):
    real = np.asarray(real_alarms, dtype=np.float64)
    control = np.asarray(control_alarms, dtype=np.float64)
    if not len(real) or len(real) != len(control):
        return None
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(draws):
        idx = rng.choice(len(real), size=len(real), replace=True)
        diffs.append(float(real[idx].mean() - control[idx].mean()))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"mean": float(real.mean() - control.mean()),
            "ci_low": float(lo), "ci_high": float(hi),
            "n_episodes": int(len(real))}


def arm_metrics(test, events, event_windows, normal_samples, normal_episode_ids,
                score, threshold):
    results = pl.evaluate_events(test, events, score, threshold, event_windows)
    legal = [r for r in results if r["reachable"]]
    recalled = [r for r in legal if r["recalled"]]
    early_legal = [r for r in legal if r["early"]]
    early_all = [r for r in results if r["early"]]
    delays = [r["delay"] for r in recalled if r["delay"] is not None]

    by_ep = defaultdict(list)
    for s in normal_samples:
        by_ep[s["episode"]].append(s)
    ep_alarm = {}
    decision_hits = 0
    decision_total = 0
    for ep, rows in sorted(by_ep.items()):
        sc = score(rows)
        alarm = sc > threshold
        ep_alarm[int(ep)] = bool(alarm.any())
        decision_hits += int(alarm.sum())
        decision_total += len(sc)
    alarms = [ep_alarm.get(int(ep), False) for ep in normal_episode_ids]
    fpr = bootstrap_fpr(alarms)
    all_scores = score(normal_samples) if normal_samples else np.asarray([])
    return {
        "threshold": float(threshold),
        "events_all": len(results),
        "legal_windows": len(legal),
        "recalled_legal_windows": len(recalled),
        "legal_window_recall": (len(recalled) / len(legal)) if legal else None,
        "all_failure_timely_coverage": (len(recalled) / len(results)) if results else None,
        "early_alarm_events_all": len(early_all),
        "early_alarm_events_legal": len(early_legal),
        "early_alarm_rate_all": (len(early_all) / len(results)) if results else None,
        "early_alarm_rate_legal": (len(early_legal) / len(legal)) if legal else None,
        "early_alarm_episodes_all": len({r["episode"] for r in early_all}),
        "median_delay": float(np.median(delays)) if delays else None,
        "mean_delay": float(np.mean(delays)) if delays else None,
        "normal_episodes": len(normal_episode_ids),
        "normal_decision_samples": decision_total,
        "normal_episode_fpr": fpr["mean"] if fpr else None,
        "normal_episode_fpr_ci": [fpr["ci_low"], fpr["ci_high"]] if fpr else None,
        "normal_decision_fpr": (decision_hits / decision_total) if decision_total else None,
        "normal_episode_alarms": int(sum(alarms)),
        "score_min": float(all_scores.min()) if all_scores.size else None,
        "score_max": float(all_scores.max()) if all_scores.size else None,
        "score_at_0": int((all_scores == 0.0).sum()) if all_scores.size else 0,
        "score_at_1": int((all_scores == 1.0).sum()) if all_scores.size else 0,
        "_ep_alarm": ep_alarm,
        "_results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True,
                        help="shuffled-latent seed; must match the frozen fit")
    parser.add_argument("--min-legal-windows", type=int, default=50)
    parser.add_argument("--min-normal-episodes", type=int, default=100)
    args = parser.parse_args()

    arms, threshold_data, model_sha, threshold_sha = load_frozen(args.frozen_dir)
    episodes = pl.load_eval(args.eval_dir)
    test, events = pl.build_samples(episodes, 2, "fetch_handoff_failure", window=20)
    shuffled = pl.shuffled_latents(test, args.seed)
    for sample, latent in zip(test, shuffled):
        sample["latent_shuffled"] = latent

    normal_episode_ids = []
    for idx, ep in enumerate(episodes):
        meta = ep["meta"]
        failures = int(meta.get("stall_failure_count",
                                len(meta.get("probe_events", []))))
        if failures == 0:
            normal_episode_ids.append(idx)
    normal_samples = [s for s in test if s["episode"] in set(normal_episode_ids)]
    event_windows = pl.window_rows_for_events(test, events)
    legal_windows = sum(1 for rows, _, _ in event_windows if rows)

    counts = {
        "episodes": len(episodes),
        "normal_episodes": len(normal_episode_ids),
        "normal_episodes_with_samples": len({s["episode"] for s in normal_samples}),
        "failures_all": len(events),
        "failures_with_two_options": sum(1 for e in events if e.get("two_options")),
        "failures_interface_relevant": sum(
            1 for e in events if e.get("two_options") and e.get("chosen_tool")),
        "failures_with_fetch_legal_latent": sum(
            1 for e in events if e.get("fetch_legal_forward_steps")),
        "window_events": sum(1 for e in events if e.get("window")),
        "legal_windows": legal_windows,
        "samples": len(test),
        "successful_episodes": sum(1 for ep in episodes
                                   if bool(ep["meta"].get("success"))),
    }
    gate_ok = (legal_windows >= args.min_legal_windows
               and len(normal_episode_ids) >= args.min_normal_episodes)
    payload = {
        "eval_dir": args.eval_dir,
        "frozen_dir": str(Path(args.frozen_dir).resolve()),
        "seed": args.seed,
        "model_sha256": model_sha,
        "threshold_sha256": threshold_sha,
        "frozen_thresholds": threshold_data["thresholds"],
        "counts": counts,
        "gates": {
            "min_legal_windows": args.min_legal_windows,
            "min_normal_episodes": args.min_normal_episodes,
            "legal_windows_ok": legal_windows >= args.min_legal_windows,
            "normal_episodes_ok": len(normal_episode_ids) >= args.min_normal_episodes,
            "ok": gate_ok,
        },
    }
    if not gate_ok:
        payload["status"] = "insufficient_samples"
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(2)

    scores = {arm: make_score(arms, arm) for arm in ("baseline", "real", "shuffled")}
    thresholds = {arm: float(threshold_data["thresholds"][arm]["threshold"])
                  for arm in scores}
    arms_out = {}
    for arm in ("baseline", "real", "shuffled"):
        metrics = arm_metrics(test, events, event_windows, normal_samples,
                              normal_episode_ids, scores[arm], thresholds[arm])
        ep_alarm = metrics.pop("_ep_alarm")
        results = metrics.pop("_results")
        metrics["_ep_alarm"] = ep_alarm
        metrics["_results"] = results
        arms_out[arm] = metrics

    paired = {}
    for control in ("baseline", "shuffled"):
        event_diff = pl.bootstrap_diff(events, event_windows, scores, thresholds,
                                       control, "real")
        episode_diff = pl.bootstrap_episode_diff(events, event_windows, scores,
                                                 thresholds, control, "real")
        alarm_control = [arms_out[control]["_ep_alarm"].get(int(ep), False)
                         for ep in normal_episode_ids]
        alarm_real = [arms_out["real"]["_ep_alarm"].get(int(ep), False)
                      for ep in normal_episode_ids]
        paired[control] = {
            "event_level_recall_diff": event_diff,
            "episode_level_recall_diff": episode_diff,
            "fpr_diff_real_minus_control": bootstrap_fpr_diff(alarm_real, alarm_control),
        }

    for arm in arms_out:
        arms_out[arm].pop("_ep_alarm")
        arms_out[arm].pop("_results")

    acceptance = {
        "legal_windows_ge_50": counts["legal_windows"] >= 50,
        "normal_episodes_ge_100": len(normal_episode_ids) >= 100,
        "all_arms_fpr_le_7_5pct": all(
            arms_out[a]["normal_episode_fpr"] is not None
            and arms_out[a]["normal_episode_fpr"] <= 0.075
            for a in ("baseline", "real", "shuffled")),
        "real_fpr_within_2_5pp_of_baseline": (
            paired["baseline"]["fpr_diff_real_minus_control"]["mean"] <= 0.025),
        "real_fpr_within_2_5pp_of_shuffled": (
            paired["shuffled"]["fpr_diff_real_minus_control"]["mean"] <= 0.025),
        "real_minus_baseline_recall_ge_10pp": (
            paired["baseline"]["event_level_recall_diff"]["mean"] >= 0.10),
        "real_minus_baseline_event_ci_low_gt_0": (
            paired["baseline"]["event_level_recall_diff"]["ci_low"] > 0),
        "real_minus_baseline_episode_ci_low_gt_0": (
            paired["baseline"]["episode_level_recall_diff"]["ci_low"] > 0),
        "real_gt_shuffled_recall": (
            paired["shuffled"]["event_level_recall_diff"]["mean"] > 0),
        "real_minus_shuffled_event_ci_low_gt_0": (
            paired["shuffled"]["event_level_recall_diff"]["ci_low"] > 0),
    }
    acceptance["ok"] = all(acceptance.values())
    payload.update({
        "status": "evaluated",
        "arms": arms_out,
        "paired_real_minus_control": paired,
        "acceptance": acceptance,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in payload.items() if k != "arms"}, indent=2,
                     sort_keys=True))
    for arm, metrics in arms_out.items():
        print(json.dumps({"arm": arm, **{k: metrics[k] for k in (
            "threshold", "legal_window_recall", "all_failure_timely_coverage",
            "normal_episode_fpr", "normal_episode_fpr_ci", "median_delay",
            "early_alarm_events_legal")}}, sort_keys=True))


if __name__ == "__main__":
    main()
