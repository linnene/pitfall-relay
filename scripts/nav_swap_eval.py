#!/usr/bin/env python3
"""Target-swap paired evaluation for test_same episodes (offline relabeling)."""
import argparse
import json
import os

import numpy as np
import torch

import nav_selfdet2 as nd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    args = p.parse_args()
    d = args.dir

    meta = list(np.load(os.path.join(d, "test_same.npy"), allow_pickle=True))
    arrays = dict(np.load(os.path.join(d, "test_same_arrays.npz")))
    ckpt = torch.load(os.path.join(d, "model.pt"), map_location="cpu")
    mu, sd, th = ckpt["mu"], ckpt["sd"], ckpt["threshold"]
    hidden = tuple(ckpt["hidden"])

    model = nd.MLP(arrays["obs"].shape[-1] * 0 + 0, hidden) if False else None
    # rebuild model with the saved input width
    x_probe, _, _ = nd.make_features(meta, arrays)
    model = nd.MLP(x_probe.shape[1], hidden)
    model.load_state_dict(ckpt["state_dicts"][0])
    model.eval()

    def decide(md):
        x, _, keys = nd.make_features(md, arrays)
        x = ((x - mu) / sd).astype(np.float32)
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(x))).numpy()
        best = {}
        for (e, t), pr in zip(keys, probs):
            best[e] = max(best.get(e, 0.0), float(pr))
        return np.array([best[e] >= th for e in range(len(md))])

    pred0 = decide(meta)
    pos = arrays["pos"]
    cf_meta = []
    for m in meta:
        g0 = np.array(m["goal"], dtype=np.float32)
        g1 = np.array([-g0[0], g0[1]], dtype=np.float32)
        d1 = np.linalg.norm(pos[m["episode"]] - g1[None, :], axis=-1)
        arr = int(np.argmax(d1 < m["radius"])) if (d1 < m["radius"]).any() else -1
        cf = dict(m)
        cf["goal"] = g1.tolist()
        cf["arrived_t"] = arr
        cf["min_dist"] = float(d1.min())
        cf_meta.append(cf)
    pred1 = decide(cf_meta)

    y0 = np.array([m["arrived_t"] >= 0 for m in meta])
    y1 = np.array([m["arrived_t"] >= 0 for m in cf_meta])
    flip = y0 != y1
    s2f = flip & y0          # success -> failure
    f2s = flip & (~y0)       # failure -> success
    # paired correct: original decision correct AND swapped decision correct
    both = (pred0 == y0) & (pred1 == y1)
    out = {
        "episodes": len(meta),
        "flips": int(flip.sum()),
        "success_to_failure": {"n": int(s2f.sum()),
                               "both_correct": int(both[s2f].sum()),
                               "rate": float(both[s2f].mean()) if s2f.any() else None},
        "failure_to_success": {"n": int(f2s.sum()),
                               "both_correct": int(both[f2s].sum()),
                               "rate": float(both[f2s].mean()) if f2s.any() else None},
        "original_accuracy": float((pred0 == y0).mean()),
        "swapped_accuracy": float((pred1 == y1).mean()),
    }
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(d, "swap_eval.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
