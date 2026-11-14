"""Aggregate ORION episode records: coverage, activation audit, hook on/off."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

KINDS = ["network", "rapid_convergence", "back_path", "other_bypass"]


def load_episodes(out_dir):
    records = []
    for path in sorted(Path(out_dir).rglob("episode_*.json")):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_path"] = str(path)
        records.append(payload)
    return records


def coverage_summary(records):
    totals = defaultdict(lambda: defaultdict(int))
    for record in records:
        if record["contract"]["mode"] != "record":
            continue
        for agent_id, stats in record["per_agent"].items():
            key = f"agent_{agent_id}"
            totals[key]["episodes"] += 1
            totals[key]["actual"] += stats["actual_decisions_executed"]
            totals[key]["network"] += stats["network_decisions"]
            totals[key]["rapid_convergence"] += stats["rapid_convergence"]
            totals[key]["back_path"] += stats["back_path"]
            totals[key]["other_bypass"] += stats["other_bypass"]
    overall = defaultdict(int)
    summary = {}
    for key, values in sorted(totals.items()):
        actual = values["actual"]
        row = dict(values)
        row["network_share"] = values["network"] / actual if actual else None
        row["rapid_share"] = values["rapid_convergence"] / actual if actual else None
        row["back_share"] = values["back_path"] / actual if actual else None
        row["other_share"] = values["other_bypass"] / actual if actual else None
        summary[key] = row
        for field in ("actual", "network", "rapid_convergence", "back_path", "other_bypass"):
            overall[field] += values[field]
    if overall["actual"]:
        overall["network_share"] = overall["network"] / overall["actual"]
        overall["rapid_share"] = overall["rapid_convergence"] / overall["actual"]
        overall["back_share"] = overall["back_path"] / overall["actual"]
        overall["other_share"] = overall["other_bypass"] / overall["actual"]
    summary["overall"] = dict(overall)
    return summary


def activation_audit(out_dir, records):
    per_episode = []
    for record in records:
        if record["contract"]["mode"] != "record":
            continue
        stem = Path(record["_path"]).with_suffix("")
        npz_path = Path(f"{stem}_activations.npz")
        decisions_path = Path(f"{stem}_decisions.jsonl")
        decisions = []
        with open(decisions_path, encoding="utf-8") as handle:
            for line in handle:
                decisions.append(json.loads(line))
        entry = {
            "episode": record["contract"]["episode_index"],
            "decisions": len(decisions),
            "activation_present": sum(1 for d in decisions if d["activation_present"]),
            "argmax_mismatch": sum(
                1 for d in decisions if d["logp_present"] and not d["argmax_matches_action"]
            ),
            "next_node_mismatch": sum(
                1 for d in decisions if not d["next_node_matches_returned_position"]
            ),
            "option_switches": sum(1 for d in decisions if d["option_switched"]),
            "phase_counts": {
                phase: sum(1 for d in decisions if d["phase"] == phase)
                for phase in ("en_route", "reached_target")
            },
        }
        if npz_path.is_file():
            with np.load(npz_path) as data:
                ids = data["decision_id"]
                activation = data["activation"]
            entry["npz_rows"] = int(activation.shape[0])
            entry["npz_cols"] = int(activation.shape[1]) if activation.ndim == 2 else None
            entry["npz_finite"] = bool(np.isfinite(activation).all()) if activation.size else None
            entry["npz_nonzero"] = int(np.count_nonzero(activation))
            entry["npz_ids_match"] = bool(
                len(ids) == entry["activation_present"]
                and np.array_equal(
                    ids,
                    np.array(
                        [d["decision_id"] for d in decisions if d["activation_present"]],
                        dtype=np.int64,
                    ),
                )
            )
        else:
            entry["npz_rows"] = None
        per_episode.append(entry)
    return per_episode


def _positions(entries):
    return [tuple(round(float(x), 9) for x in entry["to"]) for entry in entries]


def hook_comparison(records):
    on = {}
    off = {}
    for record in records:
        key = (record["contract"]["episode_index"], record["contract"]["seed"])
        if record["contract"]["mode"] == "record":
            on[key] = record
        else:
            off[key] = record
    comparisons = []
    for key in sorted(set(on) & set(off)):
        a, b = on[key], off[key]
        agents = sorted(set(a["per_agent"]) | set(b["per_agent"]))
        trajectory_match = all(
            _positions([e for e in a.get("executed", []) if e["agent_id"] == int(agent)])
            == _positions([e for e in b.get("executed", []) if e["agent_id"] == int(agent)])
            for agent in agents
        )
        comparisons.append({
            "episode_index": key[0],
            "seed": key[1],
            "initial_match": a["initial"] == b["initial"],
            "success_on": a["success"],
            "success_off": b["success"],
            "total_step_on": a["total_step"],
            "total_step_off": b["total_step"],
            "metrics_on": a["metrics"],
            "metrics_off": b["metrics"],
            "metrics_equal": a["metrics"] == b["metrics"],
            "trajectory_equal_per_agent": trajectory_match,
            "per_agent_equal": {
                agent: {
                    "on": {k: a["per_agent"][agent][k] for k in ("travel_dist", "final_location", "explore_over", "reach_target")},
                    "off": {k: b["per_agent"][agent][k] for k in ("travel_dist", "final_location", "explore_over", "reach_target")},
                }
                for agent in agents
            },
        })
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    records = load_episodes(args.out_dir)
    executed_missing = []
    for record in records:
        if "executed" not in record:
            executed_missing.append(record["_path"])
    # Executed moves live in the per-episode JSON under "executed" only if the
    # runner wrote them; otherwise rebuild from the decisions file is impossible.
    payload = {
        "episodes": [
            {
                "episode_index": r["contract"]["episode_index"],
                "mode": r["contract"]["mode"],
                "seed": r["contract"]["seed"],
                "success": r.get("success"),
                "total_step": r.get("total_step"),
                "run_s": r.get("timing", {}).get("run_episode_s"),
                "max_rss_kb": r.get("timing", {}).get("max_rss_kb"),
                "per_agent": r.get("per_agent"),
            }
            for r in records
        ],
        "coverage": coverage_summary(records),
        "activation_audit": activation_audit(args.out_dir, records),
        "hook_comparison": hook_comparison(records),
    }
    if executed_missing:
        payload["executed_missing"] = executed_missing
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.write:
        path = Path(args.out_dir) / "analysis.json"
        path.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {path}", file=sys.stderr)
    print(text)


if __name__ == "__main__":
    main()
