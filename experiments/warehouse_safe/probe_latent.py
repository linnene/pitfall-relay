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
            if t == 0:
                # At t=0 the Actor sees the reset observation, which is not in
                # the completed-step arrays; the next sample starts at t=1.
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
            # A sample at decision t uses the observations/actions completed
            # through step t-1 and the Actor latent produced by that same
            # decision.  This keeps the local-history baseline and the latent
            # arm on identical information (obs[t] would be one step newer
            # than h_out[t]).
            for s in range(t - window, t):
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
    """Deterministic cross-episode latent shuffle.

    Each episode receives a donor episode by a fixed cyclic mapping; each
    sample picks a donor row by a stable index.  The result is therefore
    identical every time a scorer is evaluated, which is required for a frozen
    threshold and for the paired bootstrap.
    """
    by_ep = episode_latent_map(samples)
    episodes = sorted(ep for ep, rows in by_ep.items() if rows)
    if not episodes:
        return np.zeros((0, 0), dtype=np.float32)
    donor = {ep: episodes[(i + 1 + int(seed)) % len(episodes)]
             for i, ep in enumerate(episodes)}
    out = []
    for s in samples:
        donors = by_ep[donor[s["episode"]]]
        idx = (int(s["episode"]) * 1000003 + int(s["step"]) * 97 + int(s["agent"]) * 13) % len(donors)
        out.append(donors[idx]["latent"])
    return np.stack(out)


def fit_predict(train, cal, test, latent_mode, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def stack(rows):
        X = np.stack([s["features"] for s in rows])
        if latent_mode == "real":
            X = np.concatenate([X, np.stack([s["latent"] for s in rows])], axis=1)
        elif latent_mode == "shuffled":
            X = np.concatenate([X, np.stack([s["latent_shuffled"] for s in rows])], axis=1)
        return X

    X_train = stack(train)
    y_train = np.array([s["label"] for s in train], dtype=np.int32)
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=seed)
    model.fit(scaler.transform(X_train), y_train)

    def score(rows):
        if not rows:
            return np.zeros((0,), dtype=np.float32)
        return model.predict_proba(scaler.transform(stack(rows)))[:, 1]

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


def effective_deadline(event, episode_events):
    """Window end for an event, truncated by the next event in the episode.

    Samples after a later failure describe that later event; attributing them
    to the earlier event would overstate its recall.
    """
    deadline = event.get("next_relevant_decision_step")
    if deadline is None:
        return None
    later = [int(e["step"]) for e in episode_events if int(e["step"]) > int(event["step"])]
    return min(int(deadline), min(later)) if later else int(deadline)


def window_rows_for_events(test, events):
    """Return (window_rows, pre_rows, effective_deadline) per event, once.

    Event windows are truncated at the next event in the same episode and use
    only that event agent's valid decision samples.
    """
    samples_by_ep = defaultdict(list)
    for s in test:
        samples_by_ep[s["episode"]].append(s)
    events_by_ep = defaultdict(list)
    for idx, e in enumerate(events):
        events_by_ep[e["_episode_index"]].append((idx, e))
    out = [None] * len(events)
    for ep, items in events_by_ep.items():
        items.sort(key=lambda pair: int(pair[1]["step"]))
        episode_events = [pair[1] for pair in items]
        prior_windows = []
        for pos, (idx, e) in enumerate(items):
            step = int(e["step"])
            end = effective_deadline(e, episode_events)
            rows = [s for s in samples_by_ep.get(ep, [])
                    if s["agent"] == e.get("agent") and step < s["step"]
                    and end is not None and s["step"] < end]
            # Early risk is measured only in normal stretches: samples inside
            # an earlier event's post-failure window are not "early" for this
            # event.
            pre = [s for s in samples_by_ep.get(ep, [])
                   if s["agent"] == e.get("agent") and s["step"] < step
                   and not any(start < s["step"] < prior_end
                               for start, prior_end in prior_windows if prior_end is not None)]
            out[idx] = (rows, pre, end)
            if end is not None:
                prior_windows.append((step, end))
    return out


def evaluate_events(test, events, score, threshold, event_windows=None):
    if event_windows is None:
        event_windows = window_rows_for_events(test, events)
    results = []
    for idx, e in enumerate(events):
        ep = e["_episode_index"]
        step = int(e["step"])
        deadline = e.get("next_relevant_decision_step")
        window_rows, pre_rows, window_end = event_windows[idx]
        if not window_rows:
            results.append({"episode": ep, "event_step": step, "deadline": deadline,
                            "effective_deadline": window_end,
                            "reachable": False, "recalled": False, "early": False,
                            "delay": None, "window_covered": False})
            continue
        scores = score(window_rows)
        alarm = scores >= threshold
        first_alarm = None
        if alarm.any():
            first_alarm = int(window_rows[int(np.argmax(alarm))]["step"])
        pre_scores = score(pre_rows) if pre_rows else np.zeros((0,))
        results.append({
            "episode": ep,
            "event_step": step,
            "deadline": deadline,
            "effective_deadline": window_end,
            "reachable": True,
            "window_steps": int(window_end) - step,
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


def timing_comparison(result):
    """Compare first local-history alarm, first real-latent alarm and deadline."""
    arms = result.get("arms", {})
    base = {(r["episode"], r["event_step"]): r
            for r in arms.get("baseline", {}).get("event_results", [])}
    real = {(r["episode"], r["event_step"]): r
            for r in arms.get("real", {}).get("event_results", [])}
    shuffled = {(r["episode"], r["event_step"]): r
                for r in arms.get("shuffled", {}).get("event_results", [])}
    rows = []
    for key, b in base.items():
        r = real.get(key)
        s = shuffled.get(key)
        end = b.get("effective_deadline")
        step = b["event_step"]
        rows.append({
            "episode": key[0],
            "event_step": step,
            "deadline": b.get("deadline"),
            "effective_deadline": end,
            "window_steps": (int(end) - step) if end is not None else None,
            "reachable": bool(b.get("reachable")),
            "local_alarm_step": b.get("first_alarm_step"),
            "latent_alarm_step": r.get("first_alarm_step") if r else None,
            "shuffled_alarm_step": s.get("first_alarm_step") if s else None,
            "local_delay": b.get("delay"),
            "latent_delay": r.get("delay") if r else None,
            "shuffled_delay": s.get("delay") if s else None,
            "local_early": bool(b.get("early")),
            "latent_early": bool(r.get("early")) if r else None,
        })

    def median(values):
        vals = [v for v in values if v is not None]
        return float(np.median(vals)) if vals else None

    reachable = [x for x in rows if x["reachable"]]
    both = [x for x in rows if x["local_alarm_step"] is not None and x["latent_alarm_step"] is not None]
    return {
        "events": len(rows),
        "reachable_events": len(reachable),
        "reachable_window_coverage": (len(reachable) / len(rows)) if rows else None,
        "median_decision_window_steps": median([x["window_steps"] for x in reachable]),
        "local_alarm_events": sum(1 for x in rows if x["local_alarm_step"] is not None),
        "latent_alarm_events": sum(1 for x in rows if x["latent_alarm_step"] is not None),
        "both_alarm_events": len(both),
        "local_only_alarm_events": sum(1 for x in rows if x["local_alarm_step"] is not None
                                       and x["latent_alarm_step"] is None),
        "latent_only_alarm_events": sum(1 for x in rows if x["local_alarm_step"] is None
                                        and x["latent_alarm_step"] is not None),
        "median_local_delay": median([x["local_delay"] for x in rows]),
        "median_latent_delay": median([x["latent_delay"] for x in rows]),
        "median_shuffled_delay": median([x["shuffled_delay"] for x in rows]),
        "median_local_minus_latent_delay": median(
            [x["local_delay"] - x["latent_delay"] for x in both]),
        "local_early_alarm_events": sum(1 for x in rows if x["local_early"]),
        "latent_early_alarm_events": sum(1 for x in rows if x["latent_early"]),
        "per_event": rows,
    }


def _event_hits(events, event_windows, scores, thresholds, baseline_name, latent_name):
    """Precompute per-event arm hits and group event indices by episode."""
    by_ep = defaultdict(list)
    base_hits = []
    lat_hits = []
    for idx, e in enumerate(events):
        rows, _, _ = event_windows[idx]
        if not rows:
            continue
        base = bool((scores[baseline_name](rows) >= thresholds[baseline_name]).any())
        lat = bool((scores[latent_name](rows) >= thresholds[latent_name]).any())
        by_ep[e["_episode_index"]].append(len(base_hits))
        base_hits.append(base)
        lat_hits.append(lat)
    return by_ep, np.asarray(base_hits, dtype=np.float32), np.asarray(lat_hits, dtype=np.float32)


def bootstrap_episode_diff(events, event_windows, scores, thresholds,
                           baseline_name="baseline", latent_name="real"):
    """Episode-level paired recall difference, resampling complete episodes."""
    by_ep, base_hits, lat_hits = _event_hits(
        events, event_windows, scores, thresholds, baseline_name, latent_name)
    rates = []
    for ep, idxs in by_ep.items():
        if not idxs:
            continue
        rates.append((ep, float(np.mean(base_hits[idxs])), float(np.mean(lat_hits[idxs]))))
    if not rates:
        return None
    rng = np.random.RandomState(4343)
    diffs = []
    for _ in range(1000):
        draw = rng.choice(len(rates), size=len(rates), replace=True)
        diffs.append(float(np.mean([rates[i][2] - rates[i][1] for i in draw])))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"mean": float(np.mean([r[2] - r[1] for r in rates])),
            "ci_low": float(lo), "ci_high": float(hi),
            "n_episodes": len(rates)}


def bootstrap_diff(events, event_windows, scores, thresholds,
                   baseline_name="baseline", latent_name="real"):
    """Event-level paired recall difference, resampling whole episodes."""
    by_ep, base_hits, lat_hits = _event_hits(
        events, event_windows, scores, thresholds, baseline_name, latent_name)
    ep_ids = sorted(by_ep)
    if not ep_ids:
        return None
    rng = np.random.RandomState(4242)
    diffs = []
    for _ in range(1000):
        draw = rng.choice(ep_ids, size=len(ep_ids), replace=True)
        idxs = [i for ep in draw for i in by_ep[ep]]
        if idxs:
            diffs.append(float(np.mean(lat_hits[idxs]) - np.mean(base_hits[idxs])))
    if not diffs:
        return None
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"mean": float(np.mean(lat_hits) - np.mean(base_hits)),
            "ci_low": float(lo), "ci_high": float(hi), "n_bootstrap": len(diffs)}


def run_one(eval_dir, event_type, agent, train_count, cal_count, out_dir, seed=0,
            normal_mode="episode"):
    episodes = load_eval(eval_dir)
    if len(episodes) < train_count + cal_count + 1:
        raise ValueError(f"not enough episodes in {eval_dir}: {len(episodes)}")
    train_eps = episodes[:train_count]
    cal_eps = episodes[train_count:train_count + cal_count]
    test_eps = episodes[train_count + cal_count:]
    train, train_events = build_samples(train_eps, agent, event_type, window=20)
    cal, cal_events = build_samples(cal_eps, agent, event_type, window=20)
    test, test_events = build_samples(test_eps, agent, event_type, window=20)
    for split in (train, cal, test):
        shuffled = shuffled_latents(split, seed)
        for sample, latent in zip(split, shuffled):
            sample["latent_shuffled"] = latent
    if normal_mode == "window":
        # Exploratory only: event episodes can supply normal stretches outside
        # event windows.  This is not the pre-registered normal-episode FPR.
        normal_cal = [s for s in cal if s["label"] == 0]
        normal_test = [s for s in test if s["label"] == 0]
    else:
        normal_cal = [s for s in cal if s["label"] == 0 and not any(
            e["_episode_index"] == s["episode"] for e in cal_events)]
        normal_test = [s for s in test if s["label"] == 0 and not any(
            e["_episode_index"] == s["episode"] for e in test_events)]
    event_eps = {e["_episode_index"] for e in test_events}
    normal_test_eps = {s["episode"] for s in normal_test}
    normal_cal_eps = {s["episode"] for s in normal_cal}
    if not train or not cal or not test:
        return {"eval_dir": str(eval_dir), "event_type": event_type, "agent": agent,
                "error": "empty train/cal/test sample set",
                "counts": {"train_samples": len(train), "cal_samples": len(cal), "test_samples": len(test)}}
    train_labels = np.array([s["label"] for s in train], dtype=np.int32)
    if train_labels.min() == train_labels.max():
        return {"eval_dir": str(eval_dir), "event_type": event_type, "agent": agent,
                "error": "training samples contain only one class",
                "counts": {"train_samples": len(train), "cal_samples": len(cal),
                           "test_samples": len(test), "train_events": len(train_events),
                           "cal_events": len(cal_events), "test_events": len(test_events)}}
    result = {
        "eval_dir": str(eval_dir),
        "event_type": event_type,
        "agent": agent,
        "normal_mode": normal_mode,
        "splits": {
            "train": [train_eps[0]["meta"]["seed"], train_eps[-1]["meta"]["seed"]],
            "cal": [cal_eps[0]["meta"]["seed"], cal_eps[-1]["meta"]["seed"]],
            "test": [test_eps[0]["meta"]["seed"], test_eps[-1]["meta"]["seed"]],
        },
        "counts": {"train_samples": len(train), "cal_samples": len(cal), "test_samples": len(test),
                   "train_events": len(train_events), "cal_events": len(cal_events),
                   "test_events": len(test_events), "normal_cal_samples": len(normal_cal),
                   "normal_test_samples": len(normal_test),
                   "train_episodes": len(train_eps), "cal_episodes": len(cal_eps),
                   "test_episodes": len(test_eps),
                   "test_event_episodes": len(event_eps),
                   "test_normal_episodes": len(normal_test_eps),
                   "cal_normal_episodes": len(normal_cal_eps)},
        "arms": {},
    }
    scores = {}
    thresholds = {}
    event_windows = window_rows_for_events(test, test_events)
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
        event_results = evaluate_events(test, test_events, score, threshold, event_windows)
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
            "recall_all_events": float(np.mean([r["recalled"] for r in event_results]))
            if event_results else None,
            "reachable_events": sum(1 for r in event_results if r["reachable"]),
            "reachable_window_coverage": (sum(1 for r in event_results if r["reachable"])
                                          / len(event_results)) if event_results else None,
            "total_events": len(event_results),
            "early_alarm_events": sum(1 for r in event_results if r["early"]),
            "median_delay": float(np.median([r["delay"] for r in event_results if r["delay"] is not None]))
            if any(r["delay"] is not None for r in event_results) else None,
            "normal_fpr": normal,
            "pr_auc": pr_auc,
        }
    if "baseline" in thresholds and "real" in thresholds:
        paired = bootstrap_diff(test_events, event_windows, scores, thresholds,
                                baseline_name="baseline", latent_name="real")
        result["paired_baseline_vs_real"] = paired
        result["paired_episode_baseline_vs_real"] = bootstrap_episode_diff(
            test_events, event_windows, scores, thresholds,
            baseline_name="baseline", latent_name="real")
    if "shuffled" in thresholds and "real" in thresholds:
        result["paired_shuffled_vs_real"] = bootstrap_diff(
            test_events, event_windows, scores, thresholds,
            baseline_name="shuffled", latent_name="real")
        result["paired_episode_shuffled_vs_real"] = bootstrap_episode_diff(
            test_events, event_windows, scores, thresholds,
            baseline_name="shuffled", latent_name="real")
    result["timing"] = timing_comparison(result)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    arrays = {
        "feature": np.stack([s["features"] for s in test]) if test else np.zeros((0, 0), dtype=np.float32),
        "latent": np.stack([s["latent"] for s in test]) if test else np.zeros((0, 0), dtype=np.float32),
        "label": np.array([s["label"] for s in test], dtype=np.int32),
        "episode": np.array([s["episode"] for s in test], dtype=np.int32),
        "step": np.array([s["step"] for s in test], dtype=np.int32),
        "agent": np.array([s["agent"] for s in test], dtype=np.int32),
    }
    for name, scorer in scores.items():
        arrays[f"score_{name}"] = scorer(test) if test else np.zeros((0,), dtype=np.float32)
    raw_path = Path(out_dir) / f"probe_{event_type}_agent{agent}_test.npz"
    np.savez_compressed(str(raw_path), **arrays)
    result["test_arrays"] = str(raw_path)
    (Path(out_dir) / f"probe_{event_type}_agent{agent}.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--cal-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normal-mode", choices=["episode", "window"], default="episode",
                        help="episode: pre-registered event-free episodes; window: exploratory "
                             "normal stretches outside event windows")
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    summary = []
    for i, eval_dir in enumerate(args.eval_dir):
        for event_type, agent in (("handoff_failure", 2), ("tool_wait_timeout", 0), ("tool_wait_timeout", 1)):
            try:
                summary.append(run_one(eval_dir, event_type, agent,
                                       args.train_count, args.cal_count,
                                       args.out_dir, seed=args.seed + i,
                                       normal_mode=args.normal_mode))
            except Exception as exc:
                summary.append({"eval_dir": eval_dir, "event_type": event_type, "agent": agent,
                                "error": repr(exc)})
    (Path(args.out_dir) / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True)[:4000])


if __name__ == "__main__":
    main()
