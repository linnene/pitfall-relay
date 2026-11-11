#!/usr/bin/env python3
"""Natural-mixture test: controller modes sampled uniformly, all outcomes kept.

Supplementary to the stratified test. Uses the frozen collection protocol and
the same fixed scene; no model or threshold decision uses this file.
"""
import json
import os

import numpy as np

import heldout_type_collect as h

MODES = ["normal", "normal_near", "deadline", "teammate", "obstacle"]


def main():
    out = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(out, "data")
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(9400)
    feats, meta = [], []
    steps = 0
    n_batches = 60
    for b in range(n_batches):
        mode = MODES[b % len(MODES)]
        variants = [int(rng.integers(0, 2)) for _ in range(8)]
        f, ft, typ, arr, zent, p0, ptm = h.run_batch(
            9400 + b, 8, mode, variants, rng)
        for i in range(8):
            t = typ[i]
            if t == "none":
                t = "normal" if arr[i] >= 0 else "deadline"
                if t == "deadline":
                    ft[i] = h.D - 1
            if t == "ambiguous":
                continue
            feats.append(f[i])
            meta.append({"split": "test_natural", "want": mode, "mode": mode,
                         "agent_type": t, "first_t": int(ft[i]),
                         "arrived_t": int(arr[i]), "zone_entered": bool(zent[i]),
                         "variant": variants[i], "start": p0[i].tolist(),
                         "min_goal": float(np.min(f[i, :, 16])),
                         "min_teammate": float(np.min(f[i, :, 17])),
                         "min_obstacle": float(np.min(f[i, :, 18])),
                         "min_zone_margin": float(np.min(f[i, :, 19]))})
        steps += 8 * h.D
    np.savez_compressed(os.path.join(d, "test_natural.npz"),
                        feats=np.stack(feats), meta=np.array(meta, dtype=object),
                        allow_pickle=True)
    counts = {c: sum(1 for m in meta if m["agent_type"] == c) for c in h.WANT}
    print(json.dumps({"episodes": len(meta), "steps": steps, "counts": counts,
                      "zone_entered": sum(m["zone_entered"] for m in meta)}, indent=2))


if __name__ == "__main__":
    main()
