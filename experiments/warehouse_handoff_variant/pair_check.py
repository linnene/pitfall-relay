"""Check controlled stall-on/off pairs share the same state prefix."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_episodes(path):
    summary = json.loads((Path(path) / "summary.json").read_text())
    out = {}
    for ep in summary["episodes"]:
        arrays = np.load(ep["arrays"])
        out[int(ep["episode"])] = (ep, arrays)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    natural = load_episodes(args.natural)
    control = load_episodes(args.control)
    pairs = []
    for ep_idx in sorted(set(natural) & set(control)):
        n_meta, n_arr = natural[ep_idx]
        c_meta, c_arr = control[ep_idx]
        failures = n_meta.get("stall_failures", [])
        if not failures:
            continue
        failure = failures[0]
        step = int(failure["step"])
        n_fail = [e for e in n_meta.get("events", [])
                  if e["type"] == "handoff_failure" and int(e["step"]) == step]
        c_succ = [e for e in c_meta.get("events", [])
                  if e["type"] == "handoff_success" and int(e["step"]) == step]
        prefix_len = min(step + 1, n_arr["selected_action"].shape[0], c_arr["selected_action"].shape[0])
        same_actions = np.array_equal(
            n_arr["selected_action"][:prefix_len], c_arr["selected_action"][:prefix_len])
        same_actual = np.array_equal(
            n_arr["actual_action"][:prefix_len], c_arr["actual_action"][:prefix_len])
        same_obs = np.array_equal(n_arr["obs"][:step], c_arr["obs"][:step])
        links = [r for r in n_meta.get("window_links", [])
                 if int(r["failure_step"]) == step]
        pairs.append({
            "episode": ep_idx,
            "seed": n_meta["seed"],
            "failure_step": step,
            "stall": failure["stall"],
            "natural_handoff_failure": bool(n_fail),
            "control_handoff_success": bool(c_succ),
            "same_selected_actions_prefix": bool(same_actions),
            "same_actual_actions_prefix": bool(same_actual),
            "same_obs_prefix": bool(same_obs),
            "prefix_steps": int(prefix_len),
            "valid_pair": bool(same_actions and same_actual and same_obs and n_fail and c_succ),
            "window": bool(links and links[0].get("window")),
            "window_link": links[0] if links else None,
        })
    payload = {
        "natural": args.natural,
        "control": args.control,
        "pairs_considered": len(pairs),
        "valid_pairs": sum(1 for p in pairs if p["valid_pair"]),
        "windowed_pairs": sum(1 for p in pairs if p["valid_pair"] and p["window"]),
        "pairs": pairs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({k: payload[k] for k in ("pairs_considered", "valid_pairs", "windowed_pairs")}, indent=2))


if __name__ == "__main__":
    main()
