"""Apply the pre-fixed handoff-variant window gate to natural/control runs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--natural", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())["gate"]
    natural = json.loads((Path(args.natural) / "summary.json").read_text())
    control = json.loads((Path(args.control) / "summary.json").read_text())
    pairs = json.loads(Path(args.pairs).read_text())
    result = {
        "natural_episodes": natural["episodes_completed"],
        "natural_success_rate": natural["success_rate"],
        "stall_failures": natural["stall_failure_count"],
        "window_events": natural["window_count"],
        "window_coverage": natural["window_coverage"],
        "control_episodes": control["episodes_completed"],
        "valid_pairs": pairs["valid_pairs"],
        "windowed_pairs": pairs["windowed_pairs"],
        "criteria": {
            "min_window_events": config["min_window_events"],
            "min_valid_pairs": config["min_valid_pairs"],
            "min_windowed_pairs": config["min_windowed_pairs"],
            "max_natural_episodes": config["natural_episodes"],
            "max_control_episodes": config["controlled_pairs"],
        },
    }
    result["passed"] = (
        result["natural_episodes"] <= config["natural_episodes"]
        and result["control_episodes"] <= config["controlled_pairs"]
        and result["window_events"] >= config["min_window_events"]
        and result["valid_pairs"] >= config["min_valid_pairs"]
        and result["windowed_pairs"] >= config["min_windowed_pairs"]
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
