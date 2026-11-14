"""Run one ORION 3-agent evaluation episode on CPU (no training).

Example (on amgg):
    env/bin/python run_episode.py --orion-root .../ORION \
        --checkpoint .../ORION/model/checkpoint.pth \
        --episode-index 0 --seed 0 --mode record --out-dir .../runs/map000
"""

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

import orion_adapter
from recorder import DecisionRecorder, write_json


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orion-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["record", "nohook"], default="record")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    orion_root = orion_adapter.enter_orion_root(args.orion_root)
    import torch

    orion_adapter.seed_all(args.seed, args.torch_threads)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    net, contract = orion_adapter.load_policy(orion_root, args.checkpoint)
    contract["source_sha256"] = orion_adapter.source_fingerprint(orion_root)
    contract["python"] = sys.version.split()[0]
    contract["platform"] = platform.platform()
    contract["torch_threads"] = int(torch.get_num_threads())
    contract["seed"] = int(args.seed)
    contract["episode_index"] = int(args.episode_index)
    contract["mode"] = args.mode

    build_started = time.perf_counter()
    worker = orion_adapter.build_test_worker(net, args.episode_index, device="cpu")
    build_seconds = time.perf_counter() - build_started

    record = args.mode == "record"
    recorder = DecisionRecorder(
        net,
        worker,
        activation_dir=out_dir,
        record_activations=record,
        record_decisions=record,
    )

    run_started = time.perf_counter()
    worker.run_episode()
    run_seconds = time.perf_counter() - run_started
    summary = recorder.finalize()
    recorder.close()

    prefix = f"{args.tag}_" if args.tag else ""
    stem = f"{prefix}episode_{args.episode_index:03d}_{args.mode}"
    summary["contract"] = contract
    summary["timing"] = {
        "worker_build_s": build_seconds,
        "run_episode_s": run_seconds,
        "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cpu_count": cpu_count(),
    }
    summary["initial_locations"] = summary["initial"]["locations"]
    summary["initial_targets"] = summary["initial"]["targets"]
    write_json(out_dir / f"{stem}.json", json_safe(summary))

    if record:
        with open(out_dir / f"{stem}_decisions.jsonl", "w", encoding="utf-8") as handle:
            for decision in recorder.decisions:
                handle.write(json.dumps(json_safe(decision), sort_keys=True) + "\n")
        ids, activations = recorder.activation_array()
        np.savez_compressed(
            out_dir / f"{stem}_activations.npz",
            decision_id=ids,
            activation=activations,
        )

    print(json.dumps(json_safe({
        "episode_index": args.episode_index,
        "mode": args.mode,
        "success": summary["success"],
        "total_step": summary["total_step"],
        "per_agent": {
            k: {
                "actual_decisions_executed": v["actual_decisions_executed"],
                "network_decisions": v["network_decisions"],
                "rapid_convergence": v["rapid_convergence"],
                "back_path": v["back_path"],
                "explore_over": v["explore_over"],
            }
            for k, v in summary["per_agent"].items()
        },
        "activation_stats": summary["activation_stats"],
        "timing": summary["timing"],
        "warnings": summary["warnings"],
    }), indent=2))


if __name__ == "__main__":
    main()
