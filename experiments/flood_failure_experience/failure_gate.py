#!/usr/bin/env python3
"""Failure-experience gate: can a sequence-failure experience add anything
over full normal telemetry with actual pump/isolation measurements?

A/B water and damage are evolved every minute.  A has two legal orders:
pump-first (transport delayed) and transport-first.  Both skills execute
normally; only the order can miss the preservation window.  B decides at its
own node.  The normal control sees A's full level trace, pump/isolation
timeline, actual pump measurement and the ordinary failure notice; the
experience arm sees a Condition/Attempt/Outcome summary.  The probe asks
whether the summary adds a decision the normal control cannot reconstruct.
"""

import argparse
import json
import random

ARMS = ("normal", "experience", "permuted")
T_END = 20


def _evolve(initial, inflow, pump_rate, pump_start, pump_end, transport_start,
            transport_time, threshold, damage_rate):
    w = initial
    damage = 0.0
    preserved = False
    lost = False
    levels = []
    timeline = []
    for t in range(T_END):
        w += inflow
        pump_on = pump_start <= t < pump_end
        if pump_on:
            w -= min(pump_rate, w)
        w = max(0.0, w)
        damage += damage_rate * max(0.0, w - threshold)
        if damage >= 1.0:
            lost = True
        levels.append(w)
        timeline.append({"t": t, "pump_on": pump_on,
                         "transport_on": transport_start <= t < transport_start + transport_time})
        if t + 1 == transport_start + transport_time and not lost:
            preserved = True
            break
        if lost:
            break
    return {"levels": levels, "damage": damage, "preserved": preserved,
            "lost": lost, "timeline": timeline}


def run_order(initial, inflow, pump_rate, order, threshold, damage_rate,
              transport_time=5):
    if order == "pump_first":
        # 2-min pump set-up, pump 2..10, transport 10..15.
        return _evolve(initial, inflow, pump_rate, 2, 10, 10, transport_time,
                       threshold, damage_rate)
    # transport 0..5, then pump 5..13.
    return _evolve(initial, inflow, pump_rate, 5, 13, 0, transport_time,
                   threshold, damage_rate)


def make_pair(seed, kind):
    rng = random.Random(seed * 100019 + (1 if kind == "similar" else 2))
    # A's order comparison: same initial/exogenous state, both skills work.
    a0 = 0.10 + rng.uniform(-0.005, 0.005)
    a_in = 0.060 + rng.uniform(-0.003, 0.003)
    a_p = 0.060 + rng.uniform(-0.003, 0.003)
    a_thr = 0.12
    a_dr = 0.50 + rng.uniform(-0.03, 0.03)
    p = run_order(a0, a_in, a_p, "pump_first", a_thr, a_dr)
    t = run_order(a0, a_in, a_p, "transport_first", a_thr, a_dr)
    # B is similar in condition or not; its own node is fixed at 10.
    b0 = a0 if kind == "similar" else a0 - 0.04
    b_in = a_in
    b_p = a_p
    b_thr = a_thr
    b_dr = a_dr
    b_pf = run_order(b0, b_in, b_p, "pump_first", b_thr, b_dr)
    b_tf = run_order(b0, b_in, b_p, "transport_first", b_thr, b_dr)
    return {
        "seed": seed, "kind": kind,
        "a": {"pump_first": p, "transport_first": t,
              "initial": a0, "inflow": a_in, "pump_rate": a_p,
              "threshold": a_thr, "damage_rate": a_dr},
        "b": {"initial": b0, "inflow": b_in, "pump_rate": b_p,
              "threshold": b_thr, "damage_rate": b_dr,
              "pump_first": b_pf, "transport_first": b_tf},
    }


def normal_view(pair):
    """Full legal normal telemetry: A levels, pump/isolation timeline,
    actual pump measurement, failure notice, B levels/damage."""
    a = pair["a"]
    pf = a["pump_first"]
    tf = a["transport_first"]
    # A's actually attempted order is visible from its pump start; the
    # normal view includes the full pump timeline and outcome notice.
    pump_start = next(x["t"] for x in pf["timeline"] if x["pump_on"])
    return {
        "a_levels": pf["levels"],
        "a_pump_on": [x["pump_on"] for x in pf["timeline"]],
        "a_transport_on": [x["transport_on"] for x in pf["timeline"]],
        "a_pump_rate": a["pump_rate"],
        "a_failure_notice": pf["lost"],
        "b_levels": pair["b"]["pump_first"]["levels"][:10],
        "b_damage": pair["b"]["pump_first"]["damage"],
        "observed_order": "pump_first" if pump_start == 2 else "transport_first",
    }


def experience_view(pair):
    a = pair["a"]
    pf = a["pump_first"]
    return {
        "condition": {"initial": a["initial"], "inflow": a["inflow"],
                      "threshold": a["threshold"]},
        "attempt": {"order": "pump_first", "pump_start": 2, "pump_end": 10,
                    "transport_start": 10, "transport_time": 5},
        "outcome": {"preserved": pf["preserved"], "lost": pf["lost"]},
    }


def _normal_choice(normal, b):
    # Reconstruct the order from the legal pump timeline and use the same
    # rule as the experience arm: avoid pump-first when A's order failed.
    if normal["a_failure_notice"] and normal["observed_order"] == "pump_first":
        return "transport_first"
    if b["pump_first"]["preserved"]:
        return "pump_first"
    return "transport_first"


def _experience_choice(exp, b):
    if exp["outcome"]["lost"] and exp["attempt"]["order"] == "pump_first":
        return "transport_first"
    if b["pump_first"]["preserved"]:
        return "pump_first"
    return "transport_first"


def _choice_outcome(b, order):
    return b[order]


def run_dev(pairs=8):
    rows = []
    for kind in ("similar", "nonapp"):
        for seed in range(6001 if kind == "similar" else 6101,
                          6001 + pairs if kind == "similar" else 6101 + pairs):
            pair = make_pair(seed, kind)
            nv = normal_view(pair)
            ev = experience_view(pair)
            for arm in ARMS:
                if arm == "normal":
                    choice = _normal_choice(nv, pair["b"])
                    view = nv
                elif arm == "experience":
                    choice = _experience_choice(ev, pair["b"])
                    view = ev
                else:
                    # Fixed permutation: use the sibling seed's experience.
                    other = make_pair(seed + 100, kind)
                    choice = _experience_choice(experience_view(other), pair["b"])
                    view = experience_view(other)
                res = _choice_outcome(pair["b"], choice)
                rows.append({
                    "seed": seed, "kind": kind, "arm": arm,
                    "choice": choice, "b_preserved": res["preserved"],
                    "b_lost": res["lost"],
                    "a_pump_first_lost": pair["a"]["pump_first"]["lost"],
                    "a_transport_preserved": pair["a"]["transport_first"]["preserved"],
                    "normal_order": nv["observed_order"],
                    "normal_failure": nv["a_failure_notice"],
                    "view": view,
                })
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    r = run_dev()
    print(json.dumps({"n": len(r["rows"])}, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
