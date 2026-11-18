"""Direct local interface/inventory rule for the handoff variant.

The rule uses only T1's native observation at its decision step.  It is a
reported alternative to the learned probes, not a trained model and not an
input to them.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_episodes(root):
    summary = json.loads((Path(root) / "summary.json").read_text())
    out = {}
    for ep in summary["episodes"]:
        with np.load(ep["arrays"]) as handle:
            arrays = {k: handle[k] for k in handle.files}
        out[int(ep["episode"])] = (ep, arrays)
    return summary, out


def local_alarm(obs_1, rule):
    # T1 observation: 5 location + 8 human-step + 3 basket + 2 ready.
    basket = obs_1[13:16]
    ready = obs_1[16:18]
    if rule == "ready_no_basket":
        return bool(ready.any() and basket.sum() <= 0.0)
    if rule == "ready_target0_no_basket":
        return bool(ready[0] > 0.0 and basket.sum() <= 0.0)
    if rule == "ready_target1_no_basket":
        return bool(ready[1] > 0.0 and basket.sum() <= 0.0)
    raise ValueError(rule)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for root in args.eval_dir:
        summary, episodes = load_episodes(root)
        for rule in ("ready_no_basket", "ready_target0_no_basket", "ready_target1_no_basket"):
            events = []
            for ep_idx, (meta, arrays) in episodes.items():
                for e in meta.get("probe_events", []):
                    if e.get("type") != "interface_handoff_failure" or not e.get("window"):
                        continue
                    decision = e.get("t1_decision_step")
                    if decision is None or decision <= 0:
                        continue
                    obs = arrays["obs"][int(decision) - 1, 1, :18]
                    events.append({
                        "episode": ep_idx,
                        "decision_step": int(decision),
                        "alarm": bool(local_alarm(obs, rule)),
                        "ready": obs[16:18].tolist(),
                        "basket": obs[13:16].tolist(),
                    })
            normal_eps = []
            for ep_idx, (meta, arrays) in episodes.items():
                if any(e.get("type") == "interface_handoff_failure"
                       for e in meta.get("probe_events", [])):
                    continue
                alarms = []
                for t in range(1, int(meta["steps"])):
                    if not bool(arrays["valid"][t, 1]):
                        continue
                    alarms.append(bool(local_alarm(arrays["obs"][t - 1, 1, :18], rule)))
                normal_eps.append(bool(any(alarms)))
            rows.append({
                "eval_dir": root,
                "rule": rule,
                "window_events": len(events),
                "rule_alarm_events": sum(1 for e in events if e["alarm"]),
                "rule_recall": (sum(1 for e in events if e["alarm"]) / len(events)) if events else None,
                "normal_episodes": len(normal_eps),
                "normal_episode_fpr": (sum(1 for x in normal_eps if x) / len(normal_eps))
                if normal_eps else None,
                "events": events,
            })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True))
    print(json.dumps([{k: r[k] for k in ("eval_dir", "rule", "window_events", "rule_recall",
                                        "normal_episodes", "normal_episode_fpr")}
                      for r in rows], indent=2))


if __name__ == "__main__":
    main()
