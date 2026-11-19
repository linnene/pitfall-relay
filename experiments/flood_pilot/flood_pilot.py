#!/usr/bin/env python3
"""Abstract two-area flood task: order reversal and information-value pilot.

Pure stdlib.  All times are integer minutes.  No truth is given to a policy:
the scripted policies see only their own area observations plus the declared
information arm.  Scenario truth is used only by the simulator and scorer.
"""

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class AreaCfg:
    name: str
    size: float          # m2
    w0: float            # m3
    threshold: float     # m water level
    damage_rate: float   # damage per minute per m above threshold
    pump_rate: float     # m3/min
    pump_setup: int      # minutes
    transport: int       # minutes
    share: float         # share of the common pipe


@dataclass
class Scenario:
    seed: int
    regime: str
    horizon: int
    shared: list
    local_a: list
    local_b: list
    a: AreaCfg
    b: AreaCfg
    iso_travel: int
    iso_work: int
    iso_effect: float
    b_lag: int = 0


def _series(rng, n, base, amp, noise, step_t=None, step_add=0.0):
    out = []
    for t in range(n):
        v = base * (1.0 + amp * math.sin(2.0 * math.pi * t / 13.0))
        v += rng.gauss(0.0, noise)
        if step_t is not None and t >= step_t:
            v += step_add
        out.append(max(0.0, v))
    return out


def make_gate1(seed, regime):
    rng = random.Random(seed * 100003 + (1 if regime == "high_a" else 2))
    h = 20
    if regime == "high_a":
        shared = _series(rng, h, 0.35, 0.08, 0.02, 6, 0.15)
        la = _series(rng, h, 0.80, 0.08, 0.04)
        lb = _series(rng, h, 0.20, 0.10, 0.02)
        j = rng.uniform(-0.03, 0.03)
        a = AreaCfg("A", 100.0, 30.0 * (1 + j), 0.18, 1.2, 3.0, 6, 5, 0.5)
        b = AreaCfg("B", 80.0, 10.0 * (1 + j), 0.18, 1.0, 2.5, 4, 5, 0.5)
    else:
        shared = _series(rng, h, 0.30, 0.08, 0.02)
        la = _series(rng, h, 0.10, 0.10, 0.02)
        lb = _series(rng, h, 0.50, 0.08, 0.03)
        j = rng.uniform(-0.03, 0.03)
        a = AreaCfg("A", 100.0, 6.0 * (1 + j), 0.18, 1.0, 3.0, 5, 5, 0.5)
        b = AreaCfg("B", 80.0, 22.0 * (1 + j), 0.10, 1.5, 6.0, 2, 6, 0.5)
    return Scenario(seed, regime, h, shared, la, lb, a, b, 2, 3, 0.2)


def _sim_area(cfg, shared, local, policy, horizon, iso_done=None,
              include_iso=True, lag=0):
    """Run one area under pump-first or transport-first local schedule.

    Iso is applied to this area's total inflow at iso_done if include_iso.
    Returns outcome and minute-level observed levels.
    """
    if policy == "pump_first":
        pump_start, transport_start = 0, cfg.pump_setup
    elif policy == "transport_first":
        transport_start, pump_start = 0, cfg.transport
    else:
        raise ValueError(policy)
    pump_active_from = pump_start + cfg.pump_setup
    transport_done = transport_start + cfg.transport
    w = cfg.w0
    damage = 0.0
    preserved = False
    lost = False
    loss_time = None
    preserved_time = None
    levels = []
    isolated = False
    for t in range(horizon):
        idx = max(0, t - lag)
        inflow = shared[idx] * cfg.share + local[t]
        if include_iso and isolated:
            inflow *= 0.2
        w += inflow
        if t >= pump_active_from:
            w -= min(cfg.pump_rate, w)
        w = max(0.0, w)
        level = w / cfg.size
        if not preserved and not lost:
            damage += cfg.damage_rate * max(0.0, level - cfg.threshold)
            if damage >= 1.0:
                damage = 1.0
                lost = True
                loss_time = t + 1
        levels.append(level)
        if iso_done is not None and t + 1 == iso_done:
            isolated = True
        if t + 1 == transport_done and not lost and not preserved:
            preserved = True
            preserved_time = t + 1
    return {
        "level": levels,
        "damage": damage,
        "preserved": preserved,
        "lost": lost,
        "loss_time": loss_time,
        "preserved_time": preserved_time,
    }


def simulate_gate1(scn, policy):
    if policy == "pump_first":
        iso_start = max(scn.a.pump_setup, scn.b.pump_setup)
    else:
        iso_start = 0
    iso_done = iso_start + scn.iso_travel + scn.iso_work
    # The shared robot isolates A first in both arms; the arms differ in when
    # they request it and in local action order.
    ra = _sim_area(scn.a, scn.shared, scn.local_a, policy, scn.horizon,
                   iso_done=iso_done, include_iso=True)
    rb = _sim_area(scn.b, scn.shared, scn.local_b, policy, scn.horizon,
                   iso_done=None, include_iso=False, lag=scn.b_lag)
    return {"A": ra, "B": rb, "iso_start": iso_start, "iso_done": iso_done}


def gate1(seeds=8, out_path=None):
    rows = []
    for regime in ("high_a", "urgent_b"):
        for seed in range(1, seeds + 1):
            scn = make_gate1(seed, regime)
            p = simulate_gate1(scn, "pump_first")
            i = simulate_gate1(scn, "transport_first")
            p2 = simulate_gate1(scn, "pump_first")
            i2 = simulate_gate1(scn, "transport_first")
            repro = (p == p2 and i == i2)
            rows.append({
                "regime": regime, "seed": seed, "repro": repro,
                "pump_first": {
                    "A": p["A"], "B": p["B"],
                    "preserved": int(p["A"]["preserved"]) + int(p["B"]["preserved"]),
                },
                "transport_first": {
                    "A": i["A"], "B": i["B"],
                    "preserved": int(i["A"]["preserved"]) + int(i["B"]["preserved"]),
                },
                "a_flows": {"shared": scn.shared, "local": scn.local_a,
                            "w0": scn.a.w0, "cfg": asdict(scn.a)},
                "b_cfg": asdict(scn.b),
            })
    summary = {}
    for regime in ("high_a", "urgent_b"):
        rr = [r for r in rows if r["regime"] == regime]
        summary[regime] = {
            "pump_wins": sum(r["pump_first"]["preserved"] > r["transport_first"]["preserved"] for r in rr),
            "transport_wins": sum(r["transport_first"]["preserved"] > r["pump_first"]["preserved"] for r in rr),
            "ties": sum(r["pump_first"]["preserved"] == r["transport_first"]["preserved"] for r in rr),
            "pump_total": sum(r["pump_first"]["preserved"] for r in rr),
            "transport_total": sum(r["transport_first"]["preserved"] for r in rr),
            "pump_a": sum(r["pump_first"]["A"]["preserved"] for r in rr),
            "pump_b": sum(r["pump_first"]["B"]["preserved"] for r in rr),
            "transport_a": sum(r["transport_first"]["A"]["preserved"] for r in rr),
            "transport_b": sum(r["transport_first"]["B"]["preserved"] for r in rr),
            "all_repro": all(r["repro"] for r in rr),
        }
    result = {"rows": rows, "summary": summary}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1, sort_keys=True)
    return result


# ---------------------------------------------------------------------------
# Gate 2: B decision after A failure.
# ---------------------------------------------------------------------------

def _make_gate2(seed, group_type):
    """Build A failure history and B's decision-time state.

    relevant: A sees a shared inflow step earlier than B (B branch lag).
    nonapplicable: A's step is local; B's branch is flat, so A history
    misleads and the guard must reject it.
    """
    rng = random.Random(seed * 100019 + (11 if group_type == "relevant" else 12))
    h = 24
    if group_type == "relevant":
        shared = _series(rng, h, 0.25, 0.06, 0.02, 4, 1.20)
        la = _series(rng, h, 0.20, 0.08, 0.03)
        lb = _series(rng, h, 0.10, 0.08, 0.02)
        a = AreaCfg("A", 100.0, 25.0, 0.15, 1.4, 3.0, 6, 5, 0.5)
        b = AreaCfg("B", 80.0, 13.5, 0.10, 0.8, 0.5, 3, 6, 0.5)
        lag = 4
    else:
        shared = _series(rng, h, 0.02, 0.01, 0.005)
        la = _series(rng, h, 0.85, 0.06, 0.03, 4, 0.25)
        lb = _series(rng, h, 0.00, 0.00, 0.002)
        a = AreaCfg("A", 100.0, 25.0, 0.15, 1.4, 3.0, 6, 5, 0.5)
        b = AreaCfg("B", 80.0, 10.0, 0.12, 0.8, 0.5, 3, 6, 0.5)
        lag = 0
    scn = Scenario(seed, group_type, h, shared, la, lb, a, b, 0, 1, 0.05, lag)
    # A follows pump-first and fails.  Find its first loss minute.
    ra = _sim_area(a, shared, la, "pump_first", h)
    if not ra["lost"]:
        raise RuntimeError(f"A did not fail for {group_type} seed {seed}")
    t0 = ra["loss_time"]
    # B is observed with no action before the decision.
    wb = b.w0
    db = 0.0
    for t in range(t0):
        idx = max(0, t - scn.b_lag)
        wb += shared[idx] * b.share + lb[t]
        wb = max(0.0, wb)
        db += b.damage_rate * max(0.0, wb / b.size - b.threshold)
    if db >= 1.0:
        raise RuntimeError(f"B lost before A decision for {group_type} seed {seed}")
    return scn, t0, wb, db, ra


def _b_level_series(scn, t0, wb0):
    w = scn.b.w0
    levels = []
    for t in range(t0):
        idx = max(0, t - scn.b_lag)
        w += scn.shared[idx] * scn.b.share + scn.local_b[t]
        w = max(0.0, w)
        levels.append(w / scn.b.size)
    return levels


def _predict_action(scn, wb, db, action, q_est):
    b = scn.b
    w = wb
    d = db
    if action == "transport":
        total = b.transport
        iso_at = None
        pump_from = None
    elif action == "pump":
        total = b.pump_setup + b.transport
        iso_at = None
        pump_from = b.pump_setup
    elif action == "isolate":
        total = scn.iso_travel + scn.iso_work + b.transport
        iso_at = scn.iso_travel + scn.iso_work
        pump_from = None
    else:
        raise ValueError(action)
    for t in range(total):
        q = q_est * (scn.iso_effect if iso_at is not None and t >= iso_at else 1.0)
        w += q
        if pump_from is not None and t >= pump_from:
            w -= min(b.pump_rate, w)
        w = max(0.0, w)
        d += b.damage_rate * max(0.0, w / b.size - b.threshold)
        if d >= 1.0:
            return 2.0, w / b.size
    return d, w / b.size


def _estimate_q(arm, scn, t0, b_levels, a_context):
    b = scn.b
    slope = 0.0
    if len(b_levels) >= 4:
        slope = (b_levels[-1] - b_levels[-4]) / 3.0
    q = max(0.02, slope * b.size)
    if arm == "notice":
        q *= 1.15
    if arm == "history" and a_context is not None:
        # B only uses A's observable pre-pump water curve.  Guard: only
        # transfer if B's own recent trend is also rising; otherwise the
        # instance is treated as non-transferable.
        if (slope > 0.0003 and a_context["a_pre_slope"] > 0.001
                and a_context["a_pre_slope"] > 2.0 * slope):
            q = max(q, 0.9 * a_context["a_pre_slope"] * scn.a.size)
    return min(q, 2.0)


def _choose_b_action(arm, scn, t0, wb, db, b_levels, a_context):
    q = _estimate_q(arm, scn, t0, b_levels, a_context)
    pred = {a: _predict_action(scn, wb, db, a, q)[0]
            for a in ("transport", "pump", "isolate")}
    for action in ("transport", "pump", "isolate"):
        if pred[action] < 1.0:
            return action, q, pred
    action = min(pred, key=lambda a: (pred[a], a != "isolate"))
    return action, q, pred


def _run_b_action(scn, t0, wb, db, action):
    b = scn.b
    w = wb
    d = db
    lost = False
    preserved = False
    done = None
    if action == "transport":
        total = b.transport
        iso_at = None
        pump_from = None
    elif action == "pump":
        total = b.pump_setup + b.transport
        iso_at = None
        pump_from = b.pump_setup
    elif action == "isolate":
        total = scn.iso_travel + scn.iso_work + b.transport
        iso_at = scn.iso_travel + scn.iso_work
        pump_from = None
    else:
        raise ValueError(action)
    for k in range(total):
        t = t0 + k
        idx = max(0, t - scn.b_lag)
        q = scn.shared[idx] * b.share + scn.local_b[t]
        if iso_at is not None and k >= iso_at:
            q *= scn.iso_effect
        w += q
        if pump_from is not None and k >= pump_from:
            w -= min(b.pump_rate, w)
        w = max(0.0, w)
        if not preserved and not lost:
            d += b.damage_rate * max(0.0, w / b.size - b.threshold)
            if d >= 1.0:
                lost = True
                done = t + 1
        if k + 1 == total and not lost and not preserved:
            preserved = True
            done = t + 1
            break
    return {"preserved": preserved, "lost": lost, "done": done,
            "action": action}


def _gate2_one(seed, group_type):
    scn, t0, wb, db, ra = _make_gate2(seed, group_type)
    b_levels = _b_level_series(scn, t0, wb)
    # A's pre-pump curve is locally observable; take the largest positive
    # one-minute rise before its pump starts.
    pre = ra["level"][:max(1, scn.a.pump_setup)]
    a_pre_slope = 0.0
    for x, y in zip(pre, pre[1:]):
        a_pre_slope = max(a_pre_slope, y - x)
    a_context = {"a_pre_slope": a_pre_slope}
    out = {"seed": seed, "group": group_type, "t0": t0,
           "a_lost": True, "a_preserved": bool(ra["preserved"]),
           "a_history": {
               "levels": ra["level"], "order": "pump_first",
               "pump_work": min(scn.a.pump_setup, t0), "outcome": "lost"}}
    # Counterfactual: would B have survived if transported immediately?
    forced = _run_b_action(scn, t0, wb, db, "transport")
    out["transport_preserves"] = forced["preserved"]
    for arm in ("shared", "notice", "history"):
        action, q, pred = _choose_b_action(
            arm, scn, t0, wb, db, b_levels,
            a_context if arm == "history" else None)
        res = _run_b_action(scn, t0, wb, db, action)
        out[arm] = {"action": action, "q_est": q, "pred": pred,
                    "preserved": res["preserved"], "lost": res["lost"],
                    "done": res["done"]}
    # wrong isolation: isolation chosen although immediate transport would
    # have preserved B under the same actual flows.
    for arm in ("shared", "notice", "history"):
        out[arm]["wrong_isolation"] = bool(
            out[arm]["action"] == "isolate" and forced["preserved"])
    out["team_preserved"] = {
        a: int(out[a]["preserved"]) + int(out["a_preserved"])
        for a in ("shared", "notice", "history")
    }
    return out


def _bootstrap_ci(diffs, rng, n=2000):
    if not diffs:
        return [0.0, 0.0]
    vals = []
    for _ in range(n):
        vals.append(sum(rng.choice(diffs) for _ in diffs) / len(diffs))
    vals.sort()
    return [vals[int(0.025 * n)], vals[int(0.975 * n)]]


def gate2_holdout(out_path=None):
    groups = [("relevant", s) for s in range(2001, 2025)]
    groups += [("nonapplicable", s) for s in range(3001, 3025)]
    rows = [_gate2_one(seed, g) for g, seed in groups]
    summary = {}
    for g in ("relevant", "nonapplicable"):
        rr = [r for r in rows if r["group"] == g]
        summary[g] = {
            "n": len(rr),
            "b_loss": {a: sum(1 for r in rr if r[a]["lost"]) for a in ("shared", "notice", "history")},
            "b_loss_rate": {a: sum(1 for r in rr if r[a]["lost"]) / len(rr) for a in ("shared", "notice", "history")},
            "team_preserved": {a: sum(r["team_preserved"][a] for r in rr) for a in ("shared", "notice", "history")},
            "wrong_isolation": {a: sum(1 for r in rr if r[a]["wrong_isolation"]) for a in ("shared", "notice", "history")},
            "decisions": {a: {x: sum(1 for r in rr if r[a]["action"] == x) for x in ("transport", "pump", "isolate")} for a in ("shared", "notice", "history")},
            "mean_done": {a: statistics.mean(r[a]["done"] for r in rr if r[a]["done"] is not None) for a in ("shared", "notice", "history")},
            "decision_changes": {
                "notice_vs_shared": sum(r["notice"]["action"] != r["shared"]["action"] for r in rr),
                "history_vs_notice": sum(r["history"]["action"] != r["notice"]["action"] for r in rr),
            },
        }
        diffs = [int(r["history"]["lost"]) - int(r["notice"]["lost"]) for r in rr]
        summary[g]["loss_diff_history_minus_notice"] = {
            "sum": sum(diffs), "mean": sum(diffs) / len(diffs),
            "ci95": _bootstrap_ci(diffs, random.Random(7)),
        }
    checks = {
        "relevant_loss_gain": summary["relevant"]["b_loss"]["notice"] - summary["relevant"]["b_loss"]["history"],
        "relevant_team_not_lower": summary["relevant"]["team_preserved"]["history"] >= summary["relevant"]["team_preserved"]["notice"],
        "nonapp_wrong_extra": summary["nonapplicable"]["wrong_isolation"]["history"] - summary["nonapplicable"]["wrong_isolation"]["notice"],
        "nonapp_team_not_lower": summary["nonapplicable"]["team_preserved"]["history"] >= summary["nonapplicable"]["team_preserved"]["notice"],
    }
    checks["thresholds_met"] = (
        checks["relevant_loss_gain"] >= 4
        and checks["relevant_team_not_lower"]
        and checks["nonapp_wrong_extra"] <= 2
        and checks["nonapp_team_not_lower"])
    result = {"rows": rows, "summary": summary, "checks": checks}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1, sort_keys=True)
    return result


def gate2_smoke(seeds=(1, 2, 3, 4)):
    rows = []
    for seed in seeds:
        for g in ("relevant", "nonapplicable"):
            try:
                rows.append(_gate2_one(seed, g))
            except RuntimeError as exc:
                rows.append({"seed": seed, "group": g, "error": str(exc)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["gate1", "gate2-smoke", "holdout"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.command == "gate1":
        print(json.dumps(gate1(8, args.out), indent=1, sort_keys=True))
    elif args.command == "gate2-smoke":
        print(json.dumps(gate2_smoke(), indent=1, sort_keys=True))
    else:
        print(json.dumps(gate2_holdout(args.out), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
