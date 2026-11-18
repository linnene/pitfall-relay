"""Combine per-policy latent-probe summaries into a compact comparison."""

import argparse
import json
from pathlib import Path


def load_summary(path):
    path = Path(path)
    if path.is_dir():
        path = path / "summary.json"
    return json.loads(path.read_text())


def arm_row(result, name):
    arm = result.get("arms", {}).get(name, {})
    return {
        "error": arm.get("error"),
        "threshold": arm.get("threshold"),
        "recall": arm.get("recall"),
        "recall_all_events": arm.get("recall_all_events"),
        "reachable_events": arm.get("reachable_events"),
        "reachable_window_coverage": arm.get("reachable_window_coverage"),
        "total_events": arm.get("total_events"),
        "early_alarm_events": arm.get("early_alarm_events"),
        "median_delay": arm.get("median_delay"),
        "episode_fpr": (arm.get("normal_fpr") or {}).get("episode_fpr"),
        "decision_fpr": (arm.get("normal_fpr") or {}).get("decision_fpr"),
        "pr_auc": arm.get("pr_auc"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="append", required=True,
                        help="summary.json or its directory; repeat per policy")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for item in args.probe:
        for result in load_summary(item):
            if result.get("error"):
                rows.append({"eval_dir": result.get("eval_dir"),
                             "event_type": result.get("event_type"),
                             "agent": result.get("agent"),
                             "error": result["error"],
                             "counts": result.get("counts")})
                continue
            counts = result.get("counts", {})
            base = arm_row(result, "baseline")
            real = arm_row(result, "real")
            shuffled = arm_row(result, "shuffled")
            paired = result.get("paired_baseline_vs_real")
            paired_episode = result.get("paired_episode_baseline_vs_real")
            paired_shuffled = result.get("paired_shuffled_vs_real")
            paired_episode_shuffled = result.get("paired_episode_shuffled_vs_real")
            delta = None
            if base.get("recall") is not None and real.get("recall") is not None:
                delta = real["recall"] - base["recall"]
            rows.append({
                "eval_dir": result.get("eval_dir"),
                "event_type": result.get("event_type"),
                "agent": result.get("agent"),
                "counts": counts,
                "baseline": base,
                "real": real,
                "shuffled": shuffled,
                "timing": {k: v for k, v in (result.get("timing") or {}).items() if k != "per_event"},
                "delta_real_minus_baseline": delta,
                "paired_baseline_vs_real": paired,
                "paired_episode_baseline_vs_real": paired_episode,
                "paired_shuffled_vs_real": paired_shuffled,
                "paired_episode_shuffled_vs_real": paired_episode_shuffled,
            })

    supported = []
    by_key = {}
    for row in rows:
        if row.get("error"):
            continue
        key = (row["event_type"], row["agent"])
        by_key.setdefault(key, []).append(row)
    for key, group in sorted(by_key.items()):
        qualified = []
        for row in group:
            paired = row.get("paired_baseline_vs_real") or {}
            paired_episode = row.get("paired_episode_baseline_vs_real") or {}
            if (row.get("delta_real_minus_baseline") is not None
                    and row["delta_real_minus_baseline"] >= 0.10
                    and paired.get("ci_low") is not None
                    and paired["ci_low"] > 0.0
                    and paired_episode.get("ci_low") is not None
                    and paired_episode["ci_low"] > 0.0):
                qualified.append(row["eval_dir"])
        events_ok = all((row.get("counts") or {}).get("test_events", 0) >= 50
                        for row in group)
        normal_ok = all((row.get("counts") or {}).get("test_normal_episodes", 0) >= 100
                        for row in group)
        supported.append({
            "event_type": key[0],
            "agent": key[1],
            "policies": len(group),
            "qualified_policies": len(qualified),
            "events_ok": events_ok,
            "normal_episodes_ok": normal_ok,
            "support": len(qualified) >= 2 and events_ok and normal_ok,
            "qualified": qualified,
        })

    payload = {"rows": rows, "support": supported}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"rows": len(rows), "support": supported}, indent=2))


if __name__ == "__main__":
    main()
