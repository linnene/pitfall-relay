"""Build a read-only Fetch-probe view over existing variant archives."""

import argparse
import json
import os
from pathlib import Path


def load_audit(path):
    data = json.loads(Path(path).read_text())
    return {a["eval_dir"]: a for a in data["audits"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.source)
    out = Path(args.out)
    episodes_dir = out / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    audit = load_audit(args.audit)[str(source)]
    rows = {}
    for row in audit["events"]:
        rows.setdefault(int(row["episode"]), []).append(row)

    for src_json in sorted((source / "episodes").glob("*.json")):
        meta = json.loads(src_json.read_text())
        ep = int(meta["episode"])
        events = []
        for row in rows.get(ep, []):
            legal = bool(row["fetch_legal_forward_steps"])
            window = bool(row["two_options"] and row["chosen_tool"] and legal)
            events.append({
                "type": "fetch_handoff_failure",
                "agent": 2,
                "target": 0,
                "step": int(row["failure_step"]),
                "next_relevant_decision_step": int(row["t1_decision_step"])
                if window else None,
                "t1_decision_step": row["t1_decision_step"],
                "two_options": bool(row["two_options"]),
                "chosen_tool": bool(row["chosen_tool"]),
                "stall_active_at_decision": bool(row["stall_active_at_decision"]),
                "fetch_legal_forward_steps": row["fetch_legal_forward_steps"],
                "window": window,
                "event_episode": ep,
                "event_seed": int(meta["seed"]),
            })
        meta["probe_events"] = events
        meta["fetch_probe_events"] = events
        (episodes_dir / src_json.name).write_text(json.dumps(meta, sort_keys=True))
        npz_src = src_json.with_suffix(".npz")
        npz_dst = episodes_dir / npz_src.name
        if npz_dst.exists() or npz_dst.is_symlink():
            npz_dst.unlink()
        os.symlink(npz_src.resolve(), npz_dst)
    print(json.dumps({"source": str(source), "out": str(out), "episodes": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
