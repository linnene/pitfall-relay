"""Small frozen probe: local history vs local history + actor GRU latent.

Inputs are only the failing agent's local observations, selected/actual
macro-action history and (for the latent arms) the actor's post-forward GRU
hidden state. Rewards, environment internals and event labels are never used
as detector inputs. Labels come from the separately recorded event log.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def one_hot(value, dim):
    out = np.zeros(dim, dtype=np.float32)
    if 0 <= value < dim:
        out[value] = 1.0
    return out


def load_eval(eval_dir):
    episodes = []
    for path in sorted(Path(eval_dir).glob("episodes/*.json")):
        meta = json.loads(path.read_text())
        npz = np.load(path.with_suffix(".npz"))
        episodes.append({"meta": meta, "arrays": npz, "path": str(path)})
    return episodes


def build_samples(episodes, agent, event_type, window, include_latent=True):
    samples = []
    events = []
    for ep_idx, episode in enumerate(episodes):
        meta = episode["meta"]
        arrays = episode["arrays"]
        T = int(meta["steps"])
        valid = arrays["valid"]
        obs = arrays["obs"]
        selected = arrays["selected_action"]
        actual = arrays["actual_action"]
        mac_done = arrays["mac_done"]
        h_out = arrays["h_out"]
        obs_dim = 18 if agent < 2 else 5
        act_dim = 4 if agent < 2 else 6
        evs = [e for e in meta.get("events", []) if e.get("type") == event_type and int(e.get("agent", -1)) == agent]
        for e in evs:
            e = dict(e)
            e["_episode_index"] = ep_idx
            e["_T"] = T
            events.append(e)
        event_steps = sorted(int(e["step"]) for e in evs)
        deadlines = {}
        for e in evs:
            deadlines[int(e["step"])] = e.get("next_relevant_decision_step")
        for t in range(T):
            if not bool(valid[t, agent]):
                continue
            after_event = [s for s in event_steps if s < t]
            in_window = False
            if after_event:
                last = max(after_event)
                deadline = deadlines.get(last)
                if deadline is not None and t < deadline:
                    in_window = True
                else:
                    # Too late or no teammate window: exclude from probe labels.
                    continue
            feats = []
            for s in range(t - window + 1, t + 1):
                if s < 0:
                    feats.extend([0.0] * (obs_dim + 2 * act_dim + 2))
                    continue
                feats.extend(obs[s, agent, :obs_dim].tolist())
                feats.extend(one_hot(int(selected[s, agent]), act_dim).tolist())
                feats.extend(one_hot(int(actual[s, agent]), act_dim).tolist())
                feats.extend([float(mac_done[s, agent]), float(valid[s, agent])])
            feats.extend([float(t) / max(1, T), 1.0 if agent < 2 else 0.0])
            latent = h_out[t, agent].astype(np.float32).copy() if include_latent else np.zeros(0, dtype=np.float32)
            samples.append({
                "episode": ep_idx,
                "step": t,
                "agent": agent,
                "label": int(in_window),
                "features": np.asarray(feats, dtype=np.float32),
                "latent": latent,
                "event_type": event_type,
            })
    return samples, events


def episode_latent_map(samples):
    by_ep = defaultdict(list)
    for s in samples:
        by_ep[s["episode"]].append(s)
    return by_ep


def shuffled_latents(samples, seed=12345):
    rng = np.random.RandomState(seed)
    by_ep = episode_latent_map(samples)
    episodes = sorted(by_ep)
    donor = {}
    for ep in episodes:
        choices = [x for x in episodes if x != ep]
        donor[ep] = int(rng.choice(choices)) if choices else ep
    out = []
    for s in samples:
        d = donor[s["episode"]]
        donors = by_ep[d]
        pick = donors[rng.randint(len(donors))]
        out.append(pick["latent"])
    return np.stack(out) if out else np.zeros((0, 0), dtype=np.float32)


def fit_predict(train, cal, test, latent_mode, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_train = np.stack([s["features"] for s in train])
    y_train = np.array([s["label"] for s in train], dtype=np.int32)
    if latent_mode == "real":
        L_train = np.stack([s["latent"] for s in train])
        X_train = np.concatenate([X_train, L_train], axis=1)
    elif latent_mode == "shuffled":
        X_train = np.concatenate([X_train, shuffled_latents(train, seed)], axis=1)
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=seed)
    model.fit(scaler.transform(X_train), y_train)

    def score(rows):
        X = np.stack([s["features"] for s in rows])
        if latent_mode == "real":
            X = np.concatenate([X, np.stack([s["latent"] for s in rows])], axis=1)
        elif latent_mode == "shuffled":
            X = np.concatenate([X, shuffled_latents(rows, seed + 1)], axis=1)
        if len(X) == 0:
            return np.zeros((0,), dtype=np.float32)
        return model.predict_proba(scaler.transform(X))[:, 1]

    return score, model


def threshold_from_normal(cal_samples, score):
    by_ep = defaultdict(list)
    for s in cal_samples:
        by_ep[s["episode"]].append(s)
    maxima = []
    for rows in by_ep.values():
        maxima.append(float(np.max(score(rows))))
    if not maxima:
        return None, []
    threshold = float(np.quantile(np.array(maxima), 0.95, interpolation="higher"))
    return threshold, maxima


def evaluate_events(test, events, score, threshold):
    by_ep = defaultdict(list)
    for s in test:
        by_ep[s["episode"]].append(s)
    results = []
    for e in events:
        ep = e["_episode_index"]
        step = int(e["step"])
        deadline = e.get("next_relevant_decision_step")
        rows = [s for s in by_ep.get(ep, []) if s["agent"] == e.get("agent")]
        window_rows = [s for s in rows if step < s["step"] and deadline is not None and s["step"] < deadline]
        pre_rows = [s for s in rows if s["step"] < step]
        if not window_rows:
            results.append({"episode": ep, "event_step": step, "deadline": deadline,
                            "reachable": False, "recalled": False, "early": False,
                            "delay": None, "window_covered": False})
            continue
        scores = score(window_rows)
        alarm = scores >= threshold
        first_alarm = None
        if alarm.any():
            idx = int(np.argmax(alarm))
            first_alarm = int(window_rows[idx]["step"])
        pre_scores = score(pre_rows) if pre_rows else np.zeros((0,))
        results.append({
            "episode": ep,
            "event_step": step,
            "deadline": deadline,
            "reachable": True,
            "window_steps": int(deadline) - step,
            "window_covered": True,
            "recalled": bool(alarm.any()),
            "early": bool((pre_scores >= threshold).any()) if len(pre_scores) else False,
            "delay": (first_alarm - step) if first_alarm is not None else None,
            "first_alarm_step": first_alarm,
            "max_window_score": float(np.max(scores)) if len(scores) else None,
        })
    return results


def normal_fpr(test, score, threshold):
    by_ep = defaultdict(list)
    for s in test:
        by_ep[s["episode"]].append(s)
    ep_alarm = []
    decision_alarm = 0
    decision_total = 0
    for rows in by_ep.values():
        sc = score(rows)
        if len(sc) == 0:
            continue
        ep_alarm.append(bool((sc >= threshold).any()))
        decision_alarm += int((sc >= threshold).sum())
        decision_total += len(sc)
    return {
        "episodes": len(ep_alarm),
        "episode_fpr": float(np.mean(ep_alarm)) if ep_alarm else None,
        "decision_fpr": (decision_alarm / decision_total) if decision_total else None,
    }


def bootstrap_diff(test, events, scores, thresholds):
    # Paired event-level difference; resample episodes as clusters.
    ep_ids = sorted(set(e["_episode_index"] for e in events))
    by_ep = {ep: [e for e in events if e["_episode_index"] == ep] for ep in ep_ids}
    rng = np.random.RandomState(4242)
    diffs = []
    for _ in range(1000):
        draw = rng.choice(ep_ids, size=len(ep_ids), replace=True)
        base_hits = []
        lat_hits = []
        for ep in draw:
            for e in by_ep[ep]:
                for name, threshold in thresholds.items():
                    rows = [s for s in test if s["episode"] == ep and s["agent"] == e.get("agent")
                            and e["step"] < s["step"] and e.get("next_relevant_decision_step") is not None
                            and s["step"] < e["next_relevant_decision_step"]]
                    if not rows:
                        continue
                    sc = scores[name](rows)
                    hit = bool((sc >= threshold).any())
                    (base_hits if name == "baseline" else lat_hits).append(hit)
        if base_hits and lat_hits:
            diffs.append(float(np.mean(lat_hits) - np.mean(base_hits)))
    if not diffs:
        return None
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"mean": float(np.mean(diffs)), "ci_low": float(lo), "ci_high": float(hi), "n_bootstrap": len(diffs)}


def run_one(eval_dir, event_type, agent, train_count, cal_count, out_dir, seed=0):
    episodes = load_eval(eval_dir)
    if len(episodes) < train_count + cal_count + 1:
        raise ValueError(f"not enough episodes in {eval_dir}: {len(episodes)}")
    train_eps = episodes[:train_count]
    cal_eps = episodes[train_count:train_count + cal_count]
    test_eps = episodes[train_count + cal_count:]
    train, train_events = build_samples(train_eps, agent, event_type, window=20)
    cal, cal_events = build_samples(cal_eps, agent, event_type, window=20)
    test, test_events = build_samples(test_eps, agent, event_type, window=20)
    normal_cal = [s for s in cal if s["label"] == 0 and not any(
        e["_episode_index"] == s["episode"] for e in cal_events)]
    normal_test = [s for s in test if s["label"] == 0 and not any(
        e["_episode_index"] == s["episode"] for e in test_events)]
    if not train or not cal or not test:
        return {"eval_dir": str(eval_dir), "event_type": event_type, "agent": agent,
                "error": "empty train/cal/test sample set",
                "counts": {"train_samples": len(train), "cal_samples": len(cal), "test_samples": len(test)}}
    result = {
        "eval_dir": str(eval_dir),
        "event_type": event_type,
        "agent": agent,
        "splits": {
            "train": [train_eps[0]["meta"]["seed"], train_eps[-1]["meta"]["seed"]],
            "cal": [cal_eps[0]["meta"]["seed"], cal_eps[-1]["meta"]["seed"]],
            "test": [test_eps[0]["meta"]["seed"], test_eps[-1]["meta"]["seed"]],
        },
        "counts": {"train_samples": len(train), "cal_samples": len(cal), "test_samples": len(test),
                   "train_events": len(train_events), "cal_events": len(cal_events),
                   "test_events": len(test_events), "normal_cal_samples": len(normal_cal),
                   "normal_test_samples": len(normal_test)},
        "arms": {},
    }
    scores = {}
    thresholds = {}
    for arm in ("baseline", "real", "shuffled"):
        if arm == "shuffled" and not any(s["latent"].size for s in train):
            continue
        score, model = fit_predict(train, cal, test, arm, seed=seed)
        threshold, maxima = threshold_from_normal(normal_cal, score)
        if threshold is None:
            result["arms"][arm] = {"error": "no normal calibration samples"}
            continue
        thresholds[arm] = threshold
        scores[arm] = score
        event_results = evaluate_events(test, test_events, score, threshold)
        normal = normal_fpr(normal_test, score, threshold)
        test_scores = score(test)
        labels = np.array([s["label"] for s in test], dtype=np.int32)
        try:
            from sklearn.metrics import average_precision_score
            pr_auc = float(average_precision_score(labels, test_scores)) if labels.sum() > 0 else None
        except Exception:
            pr_auc = None
        result["arms"][arm] = {
            "threshold": threshold,
            "cal_normal_episode_max_scores": maxima,
            "event_results": event_results,
            "recall": float(np.mean([r["recalled"] for r in event_results if r["reachable"]]))
            if any(r["reachable"] for r in event_results) else None,
            "reachable_events": sum(1 for r in event_results if r["reachable"]),
            "total_events": len(event_results),
            "early_alarm_events": sum(1 for r in event_results if r["early"]),
            "median_delay": float(np.median([r["delay"] for r in event_results if r["delay"] is not None]))
            if any(r["delay"] is not None for r in event_results) else None,
            "normal_fpr": normal,
            "pr_auc": pr_auc,
        }
    if "baseline" in thresholds and "real" in thresholds:
        paired = bootstrap_diff(test, test_events, scores,
                                {"baseline": thresholds["baseline"], "real": thresholds["real"]})
        result["paired_baseline_vs_real"] = paired
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"probe_{event_type}_agent{agent}.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--cal-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    summary = []
    for i, eval_dir in enumerate(args.eval_dir):
        for event_type, agent in (("handoff_failure", 2), ("tool_wait_timeout", 0), ("tool_wait_timeout", 1)):
            try:
                summary.append(run_one(eval_dir, event_type, agent,
                                       args.train_count, args.cal_count,
                                       args.out_dir, seed=args.seed + i))
            except Exception as exc:
                summary.append({"eval_dir": eval_dir, "event_type": event_type, "agent": agent,
                                "error": repr(exc)})
    (Path(args.out_dir) / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True)[:4000])


if __name__ == "__main__":
    main()
