"""Pre-holdout self-checks for the frozen Fetch probe.

Synthetic tests cover the repaired threshold/comparator, episode aggregation,
timing boundaries, shuffled-latent determinism, and the feature contract.
With --frozen-dir/--cal-eval-dir/--split-file it also recomputes the frozen
calibration thresholds and FPR from the original calibration split.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_probe_dir = os.environ.get("PROBE_CODE_DIR")
sys.path.insert(0, _probe_dir if _probe_dir else str(HERE.parent / "warehouse_safe"))
import probe_latent as pl  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def threshold_case(maxima, q=0.95):
    values = np.sort(np.asarray(maxima, dtype=float))
    n = len(values)
    k = int(math.ceil((n + 1) * q))
    threshold = float(values[k - 1]) if k <= n else float("inf")
    return threshold, k


def check_synthetic():
    results = {}

    # Exact ties at the threshold: strict > must not alarm on the tie mass.
    maxima = [0.1] * 10 + [0.2] * 10
    threshold, k = threshold_case(maxima)
    results["ties"] = {
        "threshold": threshold, "k": k,
        "fpr_strict_gt": float((np.asarray(maxima) > threshold).mean()),
        "fpr_ge_would_be": float((np.asarray(maxima) >= threshold).mean()),
    }
    assert threshold == 0.2 and k == 20
    assert results["ties"]["fpr_strict_gt"] == 0.0

    # Constant scores: every episode equals the threshold; no alarm.
    constant = [0.3] * 100
    threshold, k = threshold_case(constant)
    results["constant"] = {
        "threshold": threshold, "k": k,
        "fpr_strict_gt": float((np.asarray(constant) > threshold).mean()),
    }
    assert threshold == 0.3 and k == 96
    assert results["constant"]["fpr_strict_gt"] == 0.0

    # k beyond n: no calibration alarm point exists.
    threshold, k = threshold_case([0.7])
    results["beyond_n"] = {"threshold": threshold, "k": k,
                           "beyond_n": bool(k > 1)}
    assert k == 2 and math.isinf(threshold)

    # Episode aggregation, not decision aggregation, and unequal lengths.
    samples = ([{"episode": 0}] * 2) + ([{"episode": 1}] * 100)

    def uneven_score(rows):
        return np.asarray([0.4 if r["episode"] == 0 else 0.1 for r in rows])

    fpr = pl.normal_fpr(samples, uneven_score, 0.3)
    results["unequal_lengths"] = fpr
    assert fpr["episodes"] == 2
    assert abs(fpr["episode_fpr"] - 0.5) < 1e-12
    assert abs(fpr["decision_fpr"] - 2.0 / 102.0) < 1e-12

    # Strict timing boundaries: (failure, deadline) is exclusive at both ends.
    test = [{"episode": 0, "step": t, "agent": 2} for t in range(0, 12)]
    event = {"_episode_index": 0, "_T": 12, "step": 5, "agent": 2,
             "next_relevant_decision_step": 8}

    def score_at(step):
        return lambda rows: np.asarray([1.0 if r["step"] == step else 0.0
                                        for r in rows])

    rows, _, end = pl.window_rows_for_events(test, [event])[0]
    results["timing_window"] = {
        "steps": [r["step"] for r in rows], "effective_deadline": end}
    assert [r["step"] for r in rows] == [6, 7]
    assert pl.evaluate_events(test, [event], score_at(8), 0.5)[0]["recalled"] is False
    hit = pl.evaluate_events(test, [event], score_at(7), 0.5)[0]
    assert hit["recalled"] is True and hit["delay"] == 2
    early = pl.evaluate_events(test, [event], score_at(4), 0.5)[0]
    assert early["early"] is True and early["recalled"] is False

    # A later event truncates the earlier event's window.
    event2 = {"_episode_index": 0, "_T": 12, "step": 9, "agent": 2,
              "next_relevant_decision_step": 11}
    windows = pl.window_rows_for_events(test, [event, event2])
    results["later_event_truncation"] = {
        "first": [r["step"] for r in windows[0][0]],
        "second": [r["step"] for r in windows[1][0]],
    }
    assert [r["step"] for r in windows[0][0]] == [6, 7]

    # Strict comparator at the threshold itself.
    strict = [{"episode": 0, "value": 0.25}, {"episode": 1, "value": 0.25}]
    results["strict_comparator"] = {
        "at_threshold_alarm": bool(pl.normal_fpr(
            strict, lambda rows: np.asarray([r["value"] for r in rows]), 0.25
        )["episode_fpr"] > 0),
        "above_threshold_alarm": bool(pl.normal_fpr(
            strict, lambda rows: np.asarray([r["value"] + 1e-9 for r in rows]), 0.25
        )["episode_fpr"] > 0),
    }
    assert results["strict_comparator"]["at_threshold_alarm"] is False
    assert results["strict_comparator"]["above_threshold_alarm"] is True

    # Shuffled latents are deterministic per seed and change with the seed.
    latent_samples = []
    for ep in range(3):
        for step in range(2):
            latent_samples.append({"episode": ep, "step": step, "agent": 2,
                                   "latent": np.asarray([ep, step], dtype=np.float32)})
    a = pl.shuffled_latents(latent_samples, seed=0)
    b = pl.shuffled_latents(latent_samples, seed=0)
    c = pl.shuffled_latents(latent_samples, seed=1)
    results["shuffled_determinism"] = {
        "same_seed_equal": bool(np.array_equal(a, b)),
        "different_seed_differs": bool(not np.array_equal(a, c)),
    }
    assert results["shuffled_determinism"]["same_seed_equal"]
    assert results["shuffled_determinism"]["different_seed_differs"]

    # Feature contract: features are a deterministic function of local
    # observations/actions only, with no label or event metadata.
    T, n_agent, hidden = 4, 3, 8
    rng = np.random.RandomState(7)
    arrays = {
        "valid": np.ones((T, n_agent), dtype=np.float32),
        "obs": rng.randn(T, n_agent, 5).astype(np.float32),
        "selected_action": rng.randint(0, 6, size=(T, n_agent)).astype(np.int64),
        "actual_action": rng.randint(0, 6, size=(T, n_agent)).astype(np.int64),
        "mac_done": rng.randint(0, 2, size=(T, n_agent)).astype(np.float32),
        "h_out": rng.randn(T, n_agent, hidden).astype(np.float32),
    }
    meta = {"steps": T, "probe_events": [{
        "type": "fetch_handoff_failure", "agent": 2, "step": 1,
        "next_relevant_decision_step": 3, "window": True,
    }]}
    samples, _ = pl.build_samples([{"meta": meta, "arrays": arrays}], 2,
                                  "fetch_handoff_failure", window=20)
    target = [s for s in samples if s["step"] == 2][0]
    manual = []
    for s in range(2 - 20, 2):
        if s < 0:
            manual.extend([0.0] * (5 + 2 * 6 + 2))
            continue
        manual.extend(arrays["obs"][s, 2, :5].tolist())
        manual.extend(pl.one_hot(int(arrays["selected_action"][s, 2]), 6).tolist())
        manual.extend(pl.one_hot(int(arrays["actual_action"][s, 2]), 6).tolist())
        manual.extend([float(arrays["mac_done"][s, 2]), float(arrays["valid"][s, 2])])
    manual.extend([2.0 / T, 1.0])
    results["feature_contract"] = {
        "dim": int(target["features"].shape[0]),
        "manual_match": bool(np.allclose(target["features"], np.asarray(manual, dtype=np.float32))),
        "label": target["label"],
        "finite": bool(np.isfinite(target["features"]).all()),
    }
    assert target["features"].shape[0] == 382
    assert results["feature_contract"]["manual_match"]
    assert results["feature_contract"]["finite"]
    return results


def check_frozen(frozen_dir, cal_eval_dir, split_file, seed):
    frozen_dir = Path(frozen_dir)
    threshold_data = json.loads((frozen_dir / "thresholds.json").read_text())
    model_path = frozen_dir / "models.pkl"
    model_sha = sha256_file(model_path)
    threshold_sha = sha256_file(frozen_dir / "thresholds.json")
    assert model_sha == threshold_data["model_sha256"]
    assert threshold_data["comparator"] == "strict_gt"
    assert "ceil((n+1)*0.95)" in threshold_data["threshold_rule"]

    with open(model_path, "rb") as handle:
        arms = pickle.load(handle)

    split_map = json.loads(Path(split_file).read_text())
    split = split_map[cal_eval_dir]["split"]
    episodes = pl.load_eval(cal_eval_dir)
    by_ep = {int(ep["meta"]["episode"]): ep for ep in episodes}
    cal_eps = [by_ep[i] for i in split["cal"]]
    cal, cal_events = pl.build_samples(cal_eps, 2, "fetch_handoff_failure", window=20)
    shuffled = pl.shuffled_latents(cal, seed)
    for sample, latent in zip(cal, shuffled):
        sample["latent_shuffled"] = latent
    normal_cal = [s for s in cal if s["label"] == 0 and not any(
        e["_episode_index"] == s["episode"] for e in cal_events)]

    def score(arm, rows):
        X = np.stack([s["features"] for s in rows])
        if arm == "real":
            X = np.concatenate([X, np.stack([s["latent"] for s in rows])], axis=1)
        elif arm == "shuffled":
            X = np.concatenate([X, np.stack([s["latent_shuffled"] for s in rows])], axis=1)
        frozen = arms[arm]
        return frozen["model"].predict_proba(frozen["scaler"].transform(X))[:, 1]

    out = {"frozen_dir": str(frozen_dir), "cal_eval_dir": cal_eval_dir,
           "model_sha256": model_sha, "threshold_sha256": threshold_sha,
           "split_disjoint": not (set(split["train"]) & set(split["cal"])
                                  or set(split["train"]) & set(split["test"])
                                  or set(split["cal"]) & set(split["test"])),
           "normal_cal_episodes": len({s["episode"] for s in normal_cal}),
           "arms": {}}
    for arm in ("baseline", "real", "shuffled"):
        by_ep_rows = defaultdict(list)
        for s in normal_cal:
            by_ep_rows[s["episode"]].append(s)
        maxima = [float(np.max(score(arm, rows)))
                  for _, rows in sorted(by_ep_rows.items())]
        values = np.sort(np.asarray(maxima, dtype=float))
        n = len(values)
        k = int(math.ceil((n + 1) * 0.95))
        threshold = float(values[k - 1]) if k <= n else float("inf")
        saved = threshold_data["thresholds"][arm]
        recomputed_fpr = float((values > threshold).mean())
        all_scores = score(arm, normal_cal)
        out["arms"][arm] = {
            "threshold_recomputed": threshold,
            "threshold_saved": saved["threshold"],
            "threshold_match": bool(abs(threshold - saved["threshold"]) <= 1e-12),
            "n": n, "k": k, "n_match": n == saved["n"], "k_match": k == saved["k"],
            "cal_equal": int((values == threshold).sum()),
            "cal_below": int((values < threshold).sum()),
            "cal_above": int((values > threshold).sum()),
            "fpr_strict_gt": recomputed_fpr,
            "fpr_saved": saved["cal_fpr_strict_gt"],
            "fpr_match": bool(abs(recomputed_fpr - saved["cal_fpr_strict_gt"]) <= 1e-12),
            "fpr_le_5pct": bool(recomputed_fpr <= 0.05),
            "score_min": float(all_scores.min()),
            "score_max": float(all_scores.max()),
            "score_at_0": int((all_scores == 0.0).sum()),
            "score_at_1": int((all_scores == 1.0).sum()),
            "finite_features": bool(all(np.isfinite(s["features"]).all()
                                        for s in cal)),
        }
        assert out["arms"][arm]["threshold_match"]
        assert out["arms"][arm]["n_match"] and out["arms"][arm]["k_match"]
        assert out["arms"][arm]["fpr_match"]
        assert out["arms"][arm]["fpr_le_5pct"]
        assert out["arms"][arm]["finite_features"]
    assert out["split_disjoint"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-dir")
    parser.add_argument("--cal-eval-dir")
    parser.add_argument("--split-file")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = {"synthetic": check_synthetic()}
    if args.frozen_dir:
        out["frozen"] = check_frozen(args.frozen_dir, args.cal_eval_dir,
                                     args.split_file, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
