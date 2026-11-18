"""Run the official Warehouse-A MacIAICC configuration with safe milestones.

Upstream: https://github.com/yuchen-x/MacroMARL
Pinned commit: f5899cd7b96bb3dbf7b265494a3ba8a15b2e3fe2
Upstream code is imported unchanged; this launcher only replicates the
official training loop to add immutable checkpoints/manifests. Every checkpoint
keeps actor, critic, target nets, optimizers, RNG and per-env RNG states.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp.replace(path)


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def build_env(config):
    import gym
    import macro_marl.my_env  # noqa: F401  (registers OSD-D-v7)

    run = config["run"]
    params = {
        "grid_dim": [6, 6],
        "big_box_reward": 300.0,
        "small_box_reward": 20.0,
        "penalty": -10.0,
        "n_agent": run["n_agent"],
        "terminate_step": run["env_terminate_step"],
        "TB_move_speed": run["tb_m_speed"],
        "TB_move_noisy": 0.0,
        "TB_move_cost": 0.0,
        "fetch_pass_obj_tc": 4,
        "fetch_look_for_obj_tc": 6,
        "fetch_manip_noisy": 0.0,
        "delay_delivery_penalty": run["d_pen"],
        "obs_human_wait": False,
        "human_speed_per_step": [run["h0_speed_ps"], run["h1_speed_ps"]],
    }
    return gym.make(config["env_id"], **params)


def model_kwargs(config, seed, save_dir, resume):
    run = config["run"]
    return {
        "env_terminate_step": run["env_terminate_step"],
        "n_env": run["n_env"],
        "n_agent": run["n_agent"],
        "l_mode": run["l_mode"],
        "seed": seed,
        "run_id": seed,
        "save_dir": save_dir,
        "resume": resume,
        "device": run["device"],
        "total_epi": run["total_epi"],
        "gamma": run["gamma"],
        "a_lr": run["a_lr"],
        "c_lr": run["c_lr"],
        "c_train_iteration": 1,
        "eps_start": run["eps_start"],
        "eps_end": run["eps_end"],
        "eps_stable_at": run["eps_stable_at"],
        "c_hys_start": 1.0,
        "c_hys_end": 1.0,
        "adv_hys_start": 1.0,
        "adv_hys_end": 1.0,
        "hys_stable_at": 4000,
        "critic_hys": False,
        "adv_hys": False,
        "etrpy_w_start": 0.0,
        "etrpy_w_end": 0.0,
        "etrpy_w_stable_at": 4000,
        "train_freq": run["train_freq"],
        "c_target_update_freq": run["c_target_update_freq"],
        "c_target_soft_update": False,
        "tau": 0.01,
        "n_step_TD": run["n_step_TD"],
        "TD_lambda": 0.0,
        "a_mlp_layer_size": [32, 32],
        "a_rnn_layer_size": run["a_rnn_layer_size"],
        "c_mlp_layer_size": [32, 32],
        "c_rnn_layer_size": run["c_rnn_layer_size"],
        "grad_clip_value": run["grad_clip_norm"],
        "grad_clip_norm": run["grad_clip_norm"],
        "obs_last_action": False,
        "eval_policy": run["eval_policy"],
        "eval_freq": run["eval_freq"],
        "eval_num_epi": run["eval_num_epi"],
        "sample_epi": run["sample_epi"],
        "trace_len": 10,
    }


def save_extra_state(run_id, controller, save_dir):
    """Upstream omits actor_tgt_net from its agent checkpoint; save it separately."""
    import torch
    ckpt_dir = Path("./performance") / save_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for idx, agent in enumerate(controller.agents):
        torch.save(
            {"actor_tgt_net_state_dict": agent.actor_tgt_net.state_dict()},
            ckpt_dir / f"{run_id}_agent_{idx}_extra.tar",
        )


def load_extra_state(snapshot_dir, controller):
    import torch
    snapshot_dir = Path(snapshot_dir)
    for idx, agent in enumerate(controller.agents):
        candidates = sorted(snapshot_dir.glob(f"*_agent_{idx}_extra.tar"))
        if not candidates:
            raise FileNotFoundError(f"missing actor_tgt_net extra checkpoint for agent {idx} in {snapshot_dir}")
        checkpoint = torch.load(candidates[0], map_location="cpu")
        agent.actor_tgt_net.load_state_dict(checkpoint["actor_tgt_net_state_dict"])


def checkpoint_snapshot(seed_root, work_dir, save_dir, epi_count, extra):
    source = Path(work_dir) / "performance" / save_dir / "ckpt"
    target = Path(seed_root) / "checkpoints" / f"epi_{epi_count:06d}"
    target.mkdir(parents=True, exist_ok=True)
    files = {}
    for path in sorted(source.glob("*.tar")):
        dst = target / path.name
        shutil.copy2(path, dst)
        files[dst.name] = {"sha256": sha256_file(dst), "bytes": dst.stat().st_size}
    manifest = {
        "epi_count": int(epi_count),
        "checkpoint_dir": str(target),
        "files": files,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "parent_checkpoint": extra.get("parent_checkpoint"),
        "flex_sha": extra["flex_sha"],
        "upstream_sha": extra["upstream_sha"],
        "config_sha256": extra["config_sha256"],
        "train_return_mean_recent": extra.get("train_return_mean_recent"),
        "eval_return": extra.get("eval_return"),
        "wall_s": extra.get("wall_s"),
        "note": extra.get("note", ""),
    }
    write_json(target / "manifest.json", manifest)
    append_jsonl(Path(seed_root) / "checkpoint_index.jsonl", manifest)
    return manifest


def restore_snapshot(snapshot_dir, work_dir, save_dir):
    source = Path(snapshot_dir)
    target = Path(work_dir) / "performance" / save_dir / "ckpt"
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.tar"):
        shutil.copy2(path, target / path.name)
    return snapshot_dir


def latest_snapshot(seed_root):
    snapshots = sorted((Path(seed_root) / "checkpoints").glob("epi_*"))
    if not snapshots:
        return None
    return snapshots[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--flex-sha", default=os.environ.get("FLEX_SHA", "unknown"))
    parser.add_argument("--flex-dirty", default=os.environ.get("FLEX_DIRTY", "unknown"))
    parser.add_argument("--max-epi", type=int, default=None,
                        help="calibration override; formal runs must omit this")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    config_sha = sha256_file(config_path)
    upstream_root = Path(args.upstream_root).resolve()
    sys.path.insert(0, str(upstream_root / "src"))

    import torch
    import gym
    from macro_marl.algs.pg_based.mac_iaicc_V import MacIAICC
    from macro_marl.cores.pg_based.mac_niacc.utils import (
        save_train_data, save_test_data, save_policies,
    )
    from macro_marl.cores.pg_based.mac_iaicc.utils import (
        save_checkpoint, load_checkpoint,
    )

    torch.set_num_threads(int(args.torch_threads))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    total_epi = int(args.max_epi if args.max_epi is not None else config["run"]["total_epi"])
    calibration = args.max_epi is not None
    seed_root = Path(args.out_root).resolve() / f"seed_{args.seed}"
    work_dir = seed_root / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    (work_dir / "performance").mkdir(exist_ok=True)
    (work_dir / "policy_nns").mkdir(exist_ok=True)
    save_dir = f"warehouse_A_maciaicc_s{args.seed}"
    for sub in ("performance", "policy_nns"):
        (work_dir / sub / save_dir).mkdir(parents=True, exist_ok=True)

    manifest_path = seed_root / "run_manifest.json"
    resume_snapshot = None
    if args.resume:
        resume_snapshot = latest_snapshot(seed_root)
        if resume_snapshot is None:
            raise FileNotFoundError(f"no checkpoint snapshot under {seed_root}/checkpoints")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"resume requested without {manifest_path}")
        restore_snapshot(resume_snapshot, work_dir, save_dir)
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("resumes", []).append({
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "from_snapshot": str(resume_snapshot),
            "flex_sha": args.flex_sha,
        })
        write_json(manifest_path, manifest)
    else:
        if manifest_path.exists():
            raise FileExistsError(f"{manifest_path} exists; use --resume or a fresh out-root")
        manifest = {
            "task_id": config["task_id"],
            "upstream_repo": config["upstream"]["repo"],
            "upstream_commit": config["upstream"]["commit"],
            "upstream_tarball_sha256": config["upstream"]["source_tarball_sha256"],
            "flex_sha": args.flex_sha,
            "flex_dirty": args.flex_dirty,
            "config_path": str(config_path),
            "config": config,
            "config_sha256": config_sha,
            "seed": args.seed,
            "run_id": args.seed,
            "save_dir": save_dir,
            "calibration": calibration,
            "total_epi_requested": total_epi,
            "formal_total_epi": config["run"]["total_epi"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "gym": gym.__version__,
            "numpy": np.__version__,
            "torch_threads": int(torch.get_num_threads()),
            "command": sys.argv,
        }
        write_json(manifest_path, manifest)

    events = seed_root / "run_events.jsonl"
    append_jsonl(events, {"event": "start", "time": time.time(), "epi": None,
                          "seed": args.seed, "calibration": calibration,
                          "total_epi": total_epi, "flex_sha": args.flex_sha,
                          "resume_snapshot": str(resume_snapshot) if resume_snapshot else None})

    env = build_env(config)
    kwargs = model_kwargs(config, args.seed, save_dir, args.resume)
    kwargs["total_epi"] = total_epi
    model = MacIAICC(env, **kwargs)
    started = time.time()
    epi_count = 0
    eval_returns = []
    if args.resume:
        epi_count, eval_returns = load_checkpoint(args.seed, save_dir, model.controller, model.envs_runner)
        load_extra_state(resume_snapshot, model.controller)
        model._last_checkpoint = str(resume_snapshot)
        append_jsonl(events, {"event": "resumed", "time": time.time(), "epi": epi_count,
                              "from_snapshot": str(resume_snapshot)})

    interval = int(config["checkpoint_interval_epi"])
    if calibration:
        interval = min(interval, max(1, total_epi))

    def snapshot(epi, note="", eval_return=None):
        extra = {
            "flex_sha": args.flex_sha,
            "upstream_sha": config["upstream"]["commit"],
            "config_sha256": config_sha,
            "parent_checkpoint": model._last_checkpoint,
            "train_return_mean_recent": (
                float(np.mean(model.envs_runner.train_returns[-model.train_freq:]))
                if model.envs_runner.train_returns else None
            ),
            "eval_return": eval_return,
            "wall_s": time.time() - started,
            "note": note,
        }
        manifest = checkpoint_snapshot(seed_root, work_dir, save_dir, epi, extra)
        model._last_checkpoint = manifest["checkpoint_dir"]
        append_jsonl(events, {"event": "checkpoint", "time": time.time(), "epi": epi,
                              "dir": manifest["checkpoint_dir"], "note": note})
        return manifest

    # Immutable initial state, before any update or evaluation.
    save_checkpoint(args.seed, 0, [], model.controller, model.envs_runner, save_dir)
    save_extra_state(args.seed, model.controller, save_dir)
    model._last_checkpoint = None
    snapshot(0, note="initial actor/critic before training")

    eval_every = config["run"]["eval_freq"] - (config["run"]["eval_freq"] % config["run"]["train_freq"])
    metrics_path = seed_root / "train_metrics.jsonl"

    try:
        while epi_count < total_epi:
            if model.eval_policy and epi_count % eval_every == 0:
                model.envs_runner.run(n_epis=model.eval_num_epi, test_mode=True)
                assert len(model.envs_runner.eval_returns) >= model.eval_num_epi
                eval_return = float(np.mean(model.envs_runner.eval_returns[-model.eval_num_epi:]))
                eval_returns.append(eval_return)
                model.envs_runner.eval_returns = []
                if eval_returns[-1] == np.max(eval_returns):
                    save_policies(args.seed, model.controller.agents, save_dir)
                append_jsonl(events, {"event": "eval", "time": time.time(), "epi": epi_count,
                                      "eval_return": eval_return})

            eps = model.eps_call.get_value(epi_count)
            c_hys_value = model.c_hys_call.get_value(epi_count)
            adv_hys_value = model.adv_hys_call.get_value(epi_count)
            etrpy_w = model.etrpy_w_call.get_value(epi_count)
            model.envs_runner.run(eps=eps, n_epis=model.train_freq)
            model.learner.train(eps, c_hys_value, adv_hys_value, etrpy_w,
                                model.critic_hys, model.adv_hys)
            if not model.sample_epi:
                model.memory.buf.clear()
            epi_count += model.train_freq

            if model.c_target_soft_update:
                model.learner.update_critic_target_net(soft=True)
                model.learner.update_actor_target_net(soft=True)
            elif epi_count % model.c_target_update_freq == 0:
                model.learner.update_critic_target_net()
                model.learner.update_actor_target_net()

            if epi_count % 20 == 0 or epi_count == total_epi:
                append_jsonl(metrics_path, {
                    "epi_count": epi_count,
                    "eps": eps,
                    "eval_return_last": eval_returns[-1] if eval_returns else None,
                    "train_return_mean_recent": (
                        float(np.mean(model.envs_runner.train_returns[-model.train_freq:]))
                        if model.envs_runner.train_returns else None
                    ),
                    "wall_s": time.time() - started,
                    "time": time.time(),
                })

            if epi_count % interval == 0 or epi_count >= total_epi:
                save_checkpoint(args.seed, epi_count, eval_returns,
                                model.controller, model.envs_runner, save_dir)
                save_extra_state(args.seed, model.controller, save_dir)
                snapshot(epi_count, note="periodic milestone" if epi_count < total_epi else "final",
                         eval_return=eval_returns[-1] if eval_returns else None)

        save_train_data(args.seed, model.envs_runner.train_returns, save_dir)
        save_test_data(args.seed, eval_returns, save_dir)
        save_checkpoint(args.seed, epi_count, eval_returns, model.controller, model.envs_runner, save_dir)
        save_extra_state(args.seed, model.controller, save_dir)
        if not (Path(seed_root) / "checkpoints" / f"epi_{epi_count:06d}").exists():
            snapshot(epi_count, note="final", eval_return=eval_returns[-1] if eval_returns else None)
        append_jsonl(events, {"event": "finish", "time": time.time(), "epi": epi_count,
                              "wall_s": time.time() - started})
    except Exception:
        append_jsonl(events, {"event": "exception", "time": time.time(), "epi": epi_count,
                              "traceback": traceback.format_exc()})
        try:
            save_checkpoint(args.seed, epi_count, eval_returns,
                            model.controller, model.envs_runner, save_dir)
            save_extra_state(args.seed, model.controller, save_dir)
            snapshot(epi_count, note="exception checkpoint", eval_return=eval_returns[-1] if eval_returns else None)
        except Exception:
            append_jsonl(events, {"event": "exception_checkpoint_failed", "time": time.time(),
                                  "traceback": traceback.format_exc()})
        raise
    finally:
        try:
            model.envs_runner.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
