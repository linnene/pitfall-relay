"""Fit and freeze the three Fetch probes on the original fit/calibration data."""

import argparse
import hashlib
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "warehouse_safe"))
import probe_latent as pl  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_arm(train, arm, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def stack(rows):
        X = np.stack([s["features"] for s in rows])
        if arm == "real":
            X = np.concatenate([X, np.stack([s["latent"] for s in rows])], axis=1)
        elif arm == "shuffled":
            X = np.concatenate([X, np.stack([s["latent_shuffled"] for s in rows])], axis=1)
        return X

    X = stack(train)
    y = np.asarray([s["label"] for s in train], dtype=np.int32)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=2000, solver="lbfgs",
                               class_weight="balanced", random_state=seed)
    model.fit(scaler.transform(X), y)
    return {"model": model, "scaler": scaler, "arm": arm,
            "feature_dim": int(X.shape[1]), "train_samples": int(len(train)),
            "train_positives": int(y.sum())}


def score_arm(frozen, rows):
    X = np.stack([s["features"] for s in rows])
    if frozen["arm"] == "real":
        X = np.concatenate([X, np.stack([s["latent"] for s in rows])], axis=1)
    elif frozen["arm"] == "shuffled":
        X = np.concatenate([X, np.stack([s["latent_shuffled"] for s in rows])], axis=1)
    return frozen["model"].predict_proba(frozen["scaler"].transform(X))[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    split_map = json.loads(Path(args.split_file).read_text())
    split = split_map[args.eval_dir]["split"]
    episodes = pl.load_eval(args.eval_dir)
    by_ep = {int(ep["meta"]["episode"]): ep for ep in episodes}
    train_eps = [by_ep[i] for i in split["train"]]
    cal_eps = [by_ep[i] for i in split["cal"]]
    train, train_events = pl.build_samples(train_eps, 2, "fetch_handoff_failure", window=20)
    cal, cal_events = pl.build_samples(cal_eps, 2, "fetch_handoff_failure", window=20)
    for part in (train, cal):
        sh = pl.shuffled_latents(part, args.seed)
        for sample, latent in zip(part, sh):
            sample["latent_shuffled"] = latent
    normal_cal = [s for s in cal if s["label"] == 0 and not any(
        e["_episode_index"] == s["episode"] for e in cal_events)]
    normal_cal_eps = {s["episode"] for s in normal_cal}
    if len(normal_cal_eps) < 100:
        raise ValueError(f"calibration normal episodes too few: {len(normal_cal_eps)}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    package = {"seed": args.seed, "arms": {}, "thresholds": {}, "calibration": {}}
    for arm in ("baseline", "real", "shuffled"):
        frozen = fit_arm(train, arm, args.seed)
        cal_by_ep = defaultdict(list)
        for s in normal_cal:
            cal_by_ep[s["episode"]].append(s)
        maxima = [float(np.max(score_arm(frozen, rows)))
                  for _, rows in sorted(cal_by_ep.items())]
        vals = np.sort(np.asarray(maxima, dtype=float))
        n = len(vals)
        k = int(math.ceil((n + 1) * 0.95))
        threshold = float(vals[k - 1]) if k <= n else float("inf")
        cal_fpr_gt = float((vals > threshold).mean()) if n else None
        cal_equal = int((vals == threshold).sum())
        cal_below = int((vals < threshold).sum())
        cal_above = int((vals > threshold).sum())
        scores = score_arm(frozen, normal_cal)
        package["arms"][arm] = frozen
        package["thresholds"][arm] = {
            "threshold": threshold, "n": n, "k": k,
            "beyond_n": bool(k > n), "cal_fpr_strict_gt": cal_fpr_gt,
            "cal_equal": cal_equal, "cal_below": cal_below, "cal_above": cal_above,
            "score_min": float(np.min(scores)), "score_max": float(np.max(scores)),
            "score_at_1": int((scores == 1.0).sum()),
            "score_at_0": int((scores == 0.0).sum()),
        }
        package["calibration"][arm] = maxima
    if any(v["cal_fpr_strict_gt"] is None or v["cal_fpr_strict_gt"] > 0.05
           for v in package["thresholds"].values()):
        raise SystemExit("calibration FPR exceeds 5% for at least one arm; stop")

    model_path = out / "models.pkl"
    with open(model_path, "wb") as handle:
        pickle.dump(package["arms"], handle)
    threshold_path = out / "thresholds.json"
    threshold_path.write_text(json.dumps({
        "seed": args.seed,
        "eval_dir": args.eval_dir,
        "split_file": args.split_file,
        "normal_cal_episodes": len(normal_cal_eps),
        "thresholds": package["thresholds"],
        "model_sha256": sha256_file(model_path),
        "comparator": "strict_gt",
        "threshold_rule": "ceil((n+1)*0.95) order statistic; k>n means no alarm",
        "feature_contract": "local obs/action/mac_done/valid window; latent or shuffled latent per arm; no labels/rewards/stall",
    }, indent=2, sort_keys=True))
    threshold_sha = sha256_file(threshold_path)
    (out / "thresholds.sha256").write_text(threshold_sha + "\n")
    data = json.loads(threshold_path.read_text())
    print(json.dumps({"out": str(out), "model_sha256": data["model_sha256"],
                      "threshold_sha256": threshold_sha,
                      "thresholds": data["thresholds"]}, indent=2))


if __name__ == "__main__":
    main()
