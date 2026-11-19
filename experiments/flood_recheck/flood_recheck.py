#!/usr/bin/env python3
"""Independent flood-task recheck: four arms, four parameter categories.

Reuses the frozen water/damage/action mechanics of flood_pilot.py (commit
5c4949c) but generates new scenarios and isolates the information value of
A's sensor history versus A's execution context.

Pure stdlib, deterministic per seed.  Policies never receive scenario truth.
"""

import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "flood_pilot"))
import flood_pilot as base  # noqa: E402


CATEGORIES = ("relevant_pump", "relevant_delay",
              "nonapp_local", "nonapp_handled")
CAT_LABEL = {"relevant_pump": "relevant", "relevant_delay": "relevant",
             "nonapp_local": "nonapp", "nonapp_handled": "nonapp"}
ARMS = ("state", "notice", "sensor", "exec")


def make_scenario(seed, category):
    rng = random.Random(seed * 100057 + CATEGORIES.index(category) * 17 + 7)
    h = 24
    # Independent inflow/local perturbation ranges, varied by seed.
    amp = rng.uniform(0.04, 0.10)
    noise = rng.uniform(0.01, 0.03)
    if category in ("relevant_pump", "relevant_delay"):
        step_t = 4 if category == "relevant_pump" else 3
        step_add = (rng.uniform(1.8, 2.2) if category == "relevant_pump"
                    else rng.uniform(5.0, 6.0))
        shared = base._series(rng, h, rng.uniform(0.23, 0.28), amp, noise,
                              step_t, step_add)
        la = base._series(rng, h, rng.uniform(0.15, 0.25), amp, noise)
        lb = base._series(rng, h, rng.uniform(0.08, 0.12), amp, noise)
        lag = 4 if category == "relevant_pump" else 5
    elif category == "nonapp_local":
        shared = base._series(rng, h, rng.uniform(0.03, 0.06), amp, noise)
        la = base._series(rng, h, rng.uniform(0.75, 0.95), amp, noise,
                          4, rng.uniform(0.15, 0.25))
        lb = base._series(rng, h, rng.uniform(0.08, 0.12), amp, noise,
                          4, rng.uniform(0.04, 0.08))
        lag = 0
    else:  # nonapp_handled: same transferable water, B already safe
        shared = base._series(rng, h, rng.uniform(0.23, 0.28), amp, noise,
                              4, rng.uniform(1.05, 1.35))
        la = base._series(rng, h, rng.uniform(0.15, 0.25), amp, noise)
        lb = base._series(rng, h, rng.uniform(0.08, 0.12), amp, noise)
        lag = 4
    a = base.AreaCfg("A", rng.uniform(95.0, 110.0), rng.uniform(28.8, 29.4),
                     rng.uniform(0.139, 0.141), rng.uniform(1.65, 1.75),
                     rng.uniform(2.8, 3.3), 2, rng.randint(6, 8), 0.5)
    if category == "relevant_pump":
        bdamage = rng.uniform(1.28, 1.38)
    elif category == "relevant_delay":
        bdamage = rng.uniform(1.72, 1.92)
    else:
        bdamage = rng.uniform(0.74, 0.80)
    b = base.AreaCfg("B", rng.uniform(75.0, 85.0),
                     10.0 if category in ("relevant_pump", "relevant_delay")
                     else rng.uniform(9.0, 11.0),
                     rng.uniform(0.089, 0.091), bdamage,
                     rng.uniform(0.4, 0.6), 3, rng.randint(5, 7), 0.5)
    iso_travel = 0 if category in ("relevant_pump", "relevant_delay") else 1
    iso_work = 0 if category in ("relevant_pump", "relevant_delay") else 1
    iso_effect = 0.0
    return base.Scenario(seed, category, h, shared, la, lb, a, b,
                         iso_travel, iso_work, iso_effect, lag)


def _b_state(scn, t0):
    b = scn.b
    w = b.w0
    d = 0.0
    for t in range(t0):
        idx = max(0, t - scn.b_lag)
        w += scn.shared[idx] * b.share + scn.local_b[t]
        w = max(0.0, w)
        d += b.damage_rate * max(0.0, w / b.size - b.threshold)
    return w, d


def _b_levels(scn, t0):
    b = scn.b
    w = b.w0
    out = []
    for t in range(t0):
        idx = max(0, t - scn.b_lag)
        w += scn.shared[idx] * b.share + scn.local_b[t]
        w = max(0.0, w)
        out.append(w / b.size)
    return out


def _slope(xs):
    if len(xs) < 4:
        return 0.0
    return (xs[-1] - xs[-4]) / 3.0


def _choose(arm, scn, t0, wb, db, b_levels, a_ctx, b_preserved):
    b = scn.b
    if b_preserved:
        return "wait", 0.0, {}
    bs = _slope(b_levels)
    bq = max(0.02, bs * b.size)
    aq_raw = a_ctx["raw_q"]
    aq_exec = a_ctx["exec_q"]
    def guard(q):
        return bs > 0.0002 and q <= 5.0 * max(bq, 0.05)
    if arm == "state":
        q = bq
    elif arm == "notice":
        q = bq * 1.15
    elif arm == "sensor":
        q = max(bq, 0.9 * aq_raw) if guard(aq_raw) else bq
    else:
        q = (max(bq, 0.9 * aq_exec) if guard(aq_exec)
             else (max(bq, 0.9 * aq_raw) if guard(aq_raw) else bq))
    pred = {a: base._predict_action(scn, wb, db, a, q)[0]
            for a in ("transport", "pump", "isolate")}
    for action in ("transport", "pump", "isolate"):
        if pred[action] < 1.0:
            return action, q, pred
    return min(pred, key=lambda a: (pred[a], a != "isolate")), q, pred


def _bootstrap_ci(diffs, rng, n=2000):
    if not diffs:
        return [0.0, 0.0]
    vals = []
    for _ in range(n):
        vals.append(sum(rng.choice(diffs) for _ in diffs) / len(diffs))
    vals.sort()
    return [vals[int(0.025 * n)], vals[int(0.975 * n)]]


def one(seed, category):
    scn = make_scenario(seed, category)
    ra = base._sim_area(scn.a, scn.shared, scn.local_a, "pump_first",
                        scn.horizon)
    if not ra["lost"]:
        raise RuntimeError(f"A did not fail: {category} {seed}")
    t0 = ra["loss_time"]
    if t0 < 2:
        raise RuntimeError(f"A failed too early: {category} {seed}")
    wb, db = _b_state(scn, t0)
    b_levels = _b_levels(scn, t0)
    b_preserved = category == "nonapp_handled"
    if not b_preserved and db >= 1.0:
        raise RuntimeError(f"B lost before decision: {category} {seed}")
    # A's observable sensor history and execution context.
    a_levels = ra["level"]
    raw_q = max(0.0, _slope(a_levels[-4:]) * scn.a.size)
    pump_from = scn.a.pump_setup
    pump_active_min = max(0, t0 - pump_from)
    exec_q = raw_q + (0.25 * scn.a.pump_rate if pump_active_min > 0 else 0.0)
    a_ctx = {"raw_q": raw_q, "exec_q": exec_q,
             "pump_on": pump_active_min > 0,
             "pump_work": pump_active_min,
             "order": "pump_first", "outcome": "lost"}
    row = {"seed": seed, "category": category, "t0": t0,
           "a_lost": True, "a_preserved": bool(ra["preserved"]),
           "a_confirmed": True, "message_late": False,
           "executable_choice": True,
           "b_preserved_at_decision": b_preserved,
           "a_sensor": {"levels": a_levels, "pump_from": pump_from,
                        "pump_work": pump_active_min},
           "a_exec": {"order": "pump_first", "pump_work": pump_active_min,
                      "outcome": "lost", "raw_q": raw_q, "exec_q": exec_q}}
    if b_preserved:
        for arm in ARMS:
            row[arm] = {"action": "wait", "q_est": 0.0, "pred": {},
                        "preserved": True, "lost": False, "done": t0,
                        "wrong_isolation": False}
    else:
        for arm in ARMS:
            action, q, pred = _choose(arm, scn, t0, wb, db, b_levels,
                                      a_ctx, b_preserved)
            res = base._run_b_action(scn, t0, wb, db, action)
            row[arm] = {"action": action, "q_est": q, "pred": pred,
                        "preserved": res["preserved"], "lost": res["lost"],
                        "done": res["done"], "wrong_isolation": False}
        forced = base._run_b_action(scn, t0, wb, db, "transport")
        for arm in ARMS:
            row[arm]["wrong_isolation"] = bool(
                row[arm]["action"] == "isolate" and forced["preserved"])
    row["team_preserved"] = {
        arm: int(row[arm]["preserved"]) + int(row["a_preserved"])
        for arm in ARMS}
    row["all_preserved"] = {
        arm: bool(row[arm]["preserved"] and row["a_preserved"])
        for arm in ARMS}
    return row


def run_groups(seed_map, out_path=None):
    rows = []
    for category, seeds in seed_map.items():
        for seed in seeds:
            try:
                rows.append(one(seed, category))
            except Exception as exc:
                rows.append({"seed": seed, "category": category,
                             "error": repr(exc)})
    summary = {}
    for category in CATEGORIES:
        rr = [r for r in rows if r.get("category") == category
              and "error" not in r]
        errs = [r for r in rows if r.get("category") == category
                and "error" in r]
        summary[category] = {
            "n": len(rr), "errors": len(errs),
            "b_loss": {a: sum(1 for r in rr if r[a]["lost"]) for a in ARMS},
            "a_preserved": sum(1 for r in rr if r["a_preserved"]),
            "team_preserved": {a: sum(r["team_preserved"][a] for r in rr) for a in ARMS},
            "all_preserved": {a: sum(r["all_preserved"][a] for r in rr) for a in ARMS},
            "wrong_isolation": {a: sum(1 for r in rr if r[a]["wrong_isolation"]) for a in ARMS},
            "decisions": {a: {x: sum(1 for r in rr if r[a]["action"] == x) for x in ("transport", "pump", "isolate", "wait")} for a in ARMS},
            "mean_done": {a: (statistics.mean(r[a]["done"] for r in rr if r[a]["done"] is not None)
                              if any(r[a]["done"] is not None for r in rr) else None)
                          for a in ARMS},
            "a_unconfirmed": sum(1 for r in rr if not r["a_confirmed"]),
            "message_late": sum(1 for r in rr if r["message_late"]),
            "no_executable_choice": sum(1 for r in rr if not r["executable_choice"]),
        }
        diffs = [int(r["exec"]["lost"]) - int(r["sensor"]["lost"]) for r in rr]
        summary[category]["loss_diff_exec_minus_sensor"] = {
            "sum": sum(diffs),
            "mean": sum(diffs) / len(diffs) if diffs else 0.0,
        }
        diffs2 = [int(r["sensor"]["lost"]) - int(r["notice"]["lost"]) for r in rr]
        summary[category]["loss_diff_sensor_minus_notice"] = {
            "sum": sum(diffs2),
            "mean": sum(diffs2) / len(diffs2) if diffs2 else 0.0,
        }
    result = {"rows": rows, "summary": summary}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1, sort_keys=True)
    return result


def dev_seeds():
    return {c: list(range(5001 + 100 * i, 5005 + 100 * i))
            for i, c in enumerate(CATEGORIES)}


def holdout_seeds():
    return {c: list(range(7001 + 100 * i, 7025 + 100 * i))
            for i, c in enumerate(CATEGORIES)}


def holdout(out_path=None):
    r = run_groups(holdout_seeds(), out_path)
    summary = r["summary"]
    rel = ["relevant_pump", "relevant_delay"]
    non = ["nonapp_local", "nonapp_handled"]
    rel_b_loss = {a: sum(summary[c]["b_loss"][a] for c in rel) for a in ARMS}
    non_b_loss = {a: sum(summary[c]["b_loss"][a] for c in non) for a in ARMS}
    rel_team = {a: sum(summary[c]["team_preserved"][a] for c in rel) for a in ARMS}
    non_team = {a: sum(summary[c]["team_preserved"][a] for c in non) for a in ARMS}
    non_wrong = {a: sum(summary[c]["wrong_isolation"][a] for c in non) for a in ARMS}
    checks = {
        "relevant_gain_exec_vs_sensor": rel_b_loss["sensor"] - rel_b_loss["exec"],
        "relevant_team_not_lower": rel_team["exec"] >= rel_team["sensor"],
        "relevant_subclasses_not_worse": all(
            summary[c]["b_loss"]["exec"] <= summary[c]["b_loss"]["sensor"]
            for c in rel),
        "nonapp_extra_wrong": non_wrong["exec"] - non_wrong["sensor"],
        "nonapp_team_not_lower": non_team["exec"] >= non_team["sensor"],
    }
    rel_diffs = [int(r["exec"]["lost"]) - int(r["sensor"]["lost"])
                 for r in r["rows"]
                 if r.get("category") in rel and "error" not in r]
    checks["relevant_loss_diff_ci95"] = _bootstrap_ci(rel_diffs, random.Random(7))
    checks["relevant_ci_excludes_zero"] = checks["relevant_loss_diff_ci95"][1] < 0
    checks["thresholds_met"] = (
        checks["relevant_gain_exec_vs_sensor"] >= 8
        and checks["relevant_team_not_lower"]
        and checks["relevant_subclasses_not_worse"]
        and checks["relevant_ci_excludes_zero"]
        and checks["nonapp_extra_wrong"] <= 4
        and checks["nonapp_team_not_lower"])
    r["aggregate"] = {
        "relevant_b_loss": rel_b_loss, "nonapp_b_loss": non_b_loss,
        "relevant_team": rel_team, "nonapp_team": non_team,
        "nonapp_wrong_isolation": non_wrong, "checks": checks,
        "protocol": {
            "n_total": len(r["rows"]),
            "n_valid": sum(1 for row in r["rows"] if "error" not in row),
            "errors": sum(1 for row in r["rows"] if "error" in row),
            "a_unconfirmed": sum(1 for row in r["rows"]
                                 if "error" not in row and not row["a_confirmed"]),
            "message_late": sum(1 for row in r["rows"]
                                if "error" not in row and row["message_late"]),
            "no_executable_choice": sum(1 for row in r["rows"]
                                        if "error" not in row and not row["executable_choice"]),
        },
        "attributable": {
            "n": sum(1 for row in r["rows"]
                     if row.get("category") in rel and "error" not in row
                     and row.get("executable_choice")),
            "b_loss": {a: sum(1 for row in r["rows"]
                              if row.get("category") in rel
                              and "error" not in row and row.get("executable_choice")
                              and row[a]["lost"]) for a in ARMS},
        },
    }
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1, sort_keys=True)
    return r


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["dev", "holdout"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.command == "dev":
        r = run_groups(dev_seeds(), args.out)
    else:
        r = holdout(args.out)
    print(json.dumps(r.get("aggregate", r["summary"]), indent=1,
                     sort_keys=True))


if __name__ == "__main__":
    main()
