"""Adapter for the public ORION test worker.

Upstream: https://github.com/marmotlab/ORION
Pinned commit: a935c4bab6aed90e36a24d596bfb9a74a8b234e9
License: MIT, Copyright (c) 2025 MARMot Lab @ NUS-ME

This module only imports upstream code; it does not modify it. It loads an
explicit checkpoint path (the public ``model/checkpoint.pth``) instead of the
``checkpoint_50000.pth`` path hard-coded in upstream ``test_driver.py`` and
enforces a strict state-dict contract.
"""

import hashlib
import os
import random
import sys
from pathlib import Path

UPSTREAM_COMMIT = "a935c4bab6aed90e36a24d596bfb9a74a8b234e9"


def resolve_orion_root(orion_root):
    root = Path(orion_root).expanduser().resolve()
    required = ["model.py", "agent.py", "env.py", "test_worker.py", "test_parameter.py"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{root} is not an ORION checkout; missing {missing}")
    return root


def enter_orion_root(orion_root):
    """Make upstream relative map/result paths resolve; add it to sys.path."""
    root = resolve_orion_root(orion_root)
    os.chdir(root)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def seed_all(seed, torch_threads=1):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_threads:
        torch.set_num_threads(int(torch_threads))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(root):
    root = Path(root)
    files = sorted(p for p in root.glob("*.py"))
    return {p.name: sha256_file(p) for p in files}


def load_policy(orion_root, checkpoint_path):
    """Strictly load ``checkpoint['policy_model']`` into a fresh PolicyNet."""
    import torch
    from model import PolicyNet
    from test_parameter import EMBEDDING_DIM, INPUT_DIM

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "policy_model" not in checkpoint:
        raise RuntimeError(
            "checkpoint has no 'policy_model' entry; refusing to guess or substitute weights"
        )

    state_dict = checkpoint["policy_model"]
    net = PolicyNet(INPUT_DIM, EMBEDDING_DIM)
    load_result = net.load_state_dict(state_dict, strict=True)
    net.eval()

    contract = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_top_level_keys": sorted(checkpoint.keys()),
        "checkpoint_episode": int(checkpoint.get("episode", -1)),
        "policy_model_entries": len(state_dict),
        "policy_model_params": int(sum(p.numel() for p in net.parameters())),
        "input_dim": int(INPUT_DIM),
        "embedding_dim": int(EMBEDDING_DIM),
        "load_state_dict": "strict=True ok",
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_test_driver_default_path": "model/<FOLDER_NAME>/checkpoint_50000.pth",
    }
    return net, contract


def build_test_worker(policy_net, episode_index, device="cpu"):
    from test_worker import TestWorker

    return TestWorker(
        0,
        policy_net,
        episode_index,
        device=device,
        save_image=False,
        greedy=True,
        test=True,
    )
