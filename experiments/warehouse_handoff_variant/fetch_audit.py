"""Audit whether Fetch has a legal new latent strictly before T1's choice.

Existing Warehouse handoff-variant archives only; no new rollouts.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_episodes(root):
    summary = json.loads((Path(root) / "summary.json").read_text())
    episodes = {}
    for ep in summary["episodes"]:
        with np.load(ep["arrays"]) as handle:
            arrays = {k: handle[k] for k in handle.files}
        episodes[int(ep["episode"])] = (ep, arrays)
    return summary, episodes


def new_latent(arrays, step, previous_step):
    h = arrays["h_out"][step, 2]
    if previous_step is None:
        return bool(np.linalg.norm(h) > 1e-8), None
    prev = arrays["h_out"][previous_step, 2]
    dist = float(np.linalg.norm(h - prev))
    return bool(dist > 1e-8), dist


def audit_policy(root):
    summary, episodes = load_episodes(root)
    rows = []
    for ep_idx, (meta, arrays) in episodes.items():
        failures = meta.get("stall_failures", [])
        links = {int(x["failure_step"]): x for x in meta.get("window_links", [])}
        decisions = {int(d["step"]): d for d in meta.get("t1_decisions", [])}
        valid = arrays["valid"]
        selected = arrays["selected_action"]
        for failure in failures:
            f = int(failure["step"])
            link = links.get(f, {})
            d = link.get("t1_decision_step")
            two = bool(link.get("two_options"))
            chosen_tool = bool(link.get("chosen_tool"))
            stall_active = bool(link.get("stall_active_at_decision"))
            fetch_steps = []
            if d is not None:
                last_h = None
                for t in range(f + 1, int(d)):
                    if not bool(valid[t, 2]):
                        continue
                    is_new, dist = new_latent(arrays, t, last_h)
                    if is_new:
                        fetch_steps.append({
                            "step": int(t),
                            "action": int(selected[t, 2]),
                            "latent_delta": dist,
                        })
                        last_h = int(t)
            rows.append({
                "episode": int(ep_idx),
                "seed": int(meta["seed"]),
                "failure_step": f,
                "t1_decision_step": int(d) if d is not None else None,
                "two_options": two,
                "chosen_tool": chosen_tool,
                "interface_relevant": chosen_tool,
                "stall_active_at_decision": stall_active,
                "fetch_legal_forward_steps": fetch_steps,
                "fetch_has_legal_new_latent": bool(fetch_steps),
                "failure_to_t1_delay": (int(d) - f) if d is not None else None,
            })
    normal_episodes = sum(1 for meta, _ in episodes.values()
                          if int(meta.get("stall_failure_count", 0)) == 0)
    counts = {
        "episodes": len(episodes),
        "normal_episodes": normal_episodes,
        "all_failures": len(rows),
        "t1_two_options": sum(1 for r in rows if r["two_options"]),
        "interface_relevant": sum(1 for r in rows if r["two_options"] and r["interface_relevant"]),
        "stall_active_at_decision": sum(1 for r in rows if r["two_options"]
                                        and r["interface_relevant"]
                                        and r["stall_active_at_decision"]),
        "fetch_legal_new_latent": sum(1 for r in rows if r["two_options"]
                                      and r["interface_relevant"]
                                      and r["fetch_has_legal_new_latent"]),
        "fetch_legal_any_failure": sum(1 for r in rows if r["fetch_has_legal_new_latent"]),
    }
    return {"eval_dir": str(root), "counts": counts, "events": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    audits = [audit_policy(root) for root in args.eval_dir]
    payload = {"audits": audits}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    for a in audits:
        print(json.dumps({"eval_dir": a["eval_dir"], **a["counts"]}, indent=2))


if __name__ == "__main__":
    main()
