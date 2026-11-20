#!/usr/bin/env python3
"""Minimal rescue-ordering mechanism check.

Two casualties, one shared ambulance.  Serial: ambulance serves B first while
A is cleared, then A.  Parallel: ambulance stages at A during A's clearing,
then serves A faster, then B.  Every action has a duration and a dependency;
survival means arriving at the refuge with HP > 0.  No outcome is assigned by
category, message, seed or arm.

The script also asks whether a strong local rule using normal telemetry
(HP, trend, clear/transport times) selects the same plan as A's confirmed
failure experience.  If so, the experience has no mechanism increment.
"""

import argparse
import json
import random

T_END = 40


def _serve(start, hp, decay, transport, t_end=T_END):
    """Return alive/completion for a casualty served from `start`."""
    hp0 = hp - decay * start
    if hp0 <= 0:
        return False, None
    t = start + transport
    hp_end = hp - decay * t
    return hp_end > 0, t


def simulate(case, plan):
    a, b = case["a"], case["b"]
    da, db = a["clear"], b["clear"]
    ta, tb = a["transport"], b["transport"]
    gain = case["stage_gain"]
    if plan == "serial":
        # B is cleared early and served first; A waits for its clearance.
        b_start = db
        b_ok, b_end = _serve(b_start, b["hp"], b["decay"], tb)
        a_start = max(da, b_end if b_end is not None else db + tb)
        a_ok, a_end = _serve(a_start, a["hp"], a["decay"], ta)
    elif plan == "parallel":
        # Ambulance stages at A during A's clearance; the stage is real work
        # and occupies the shared ambulance until A's clear completes.
        a_start = da
        a_ok, a_end = _serve(a_start, a["hp"], a["decay"], max(1, ta - gain))
        b_start = max(db, a_end if a_end is not None else da)
        b_ok, b_end = _serve(b_start, b["hp"], b["decay"], tb)
    else:
        raise ValueError(plan)
    return {"plan": plan, "a_ok": bool(a_ok), "b_ok": bool(b_ok),
            "team": int(bool(a_ok)) + int(bool(b_ok)),
            "a_end": a_end, "b_end": b_end}


def make_case(seed, regime):
    rng = random.Random(seed * 100019 + (1 if regime == "parallel_better" else 2))
    # Common facility/route state; independent local casualty disturbances.
    ca = 10.0 + rng.uniform(-0.2, 0.2)
    cb = 4.0 + rng.uniform(-0.2, 0.2)
    tr = 5.0
    gain = 3.0
    if regime == "parallel_better":
        a = {"hp": 8.0 + rng.uniform(-0.2, 0.2), "decay": 0.60,
             "clear": ca, "transport": tr}
        b = {"hp": 20.0 + rng.uniform(-0.2, 0.2), "decay": 0.10,
             "clear": cb, "transport": tr}
    else:
        a = {"hp": 20.0 + rng.uniform(-0.2, 0.2), "decay": 0.10,
             "clear": ca, "transport": tr}
        b = {"hp": 8.0 + rng.uniform(-0.2, 0.2), "decay": 0.60,
             "clear": cb, "transport": tr}
    return {"seed": seed, "regime": regime, "a": a, "b": b,
            "stage_gain": gain}


def precheck(n=6):
    rows = []
    for regime in ("parallel_better", "serial_better"):
        for seed in range(8001 if regime == "parallel_better" else 8101,
                          8001 + n if regime == "parallel_better" else 8101 + n):
            case = make_case(seed, regime)
            s = simulate(case, "serial")
            p = simulate(case, "parallel")
            rows.append({"seed": seed, "regime": regime,
                         "serial": s, "parallel": p,
                         "reversal": s["team"] != p["team"]})
    return {"rows": rows}


def strong_rule(case):
    # Normal local rule: predict both plans from current HP/clear/transport
    # and choose the higher predicted team outcome; no A experience.
    s = simulate(case, "serial")
    p = simulate(case, "parallel")
    return "parallel" if p["team"] >= s["team"] else "serial"


def experience_rule(case, observed_plan, observed_team, alt_team):
    # A's confirmed outcome only says which plan was better in its case.
    if observed_plan == "serial" and alt_team > observed_team:
        return "parallel"
    if observed_plan == "parallel" and alt_team > observed_team:
        return "serial"
    return observed_plan


def run_dev():
    # A observed cases and B decision cases share a regime but are different
    # locations; the strong rule uses B's own telemetry only.
    rows = []
    for regime in ("parallel_better", "serial_better"):
        for seed in range(8001 if regime == "parallel_better" else 8101,
                          8007 if regime == "parallel_better" else 8107):
            case = make_case(seed, regime)
            s = simulate(case, "serial")
            p = simulate(case, "parallel")
            # A's observed high-level attempt: serial; its outcome is the
            # actual team result.
            exp_plan = experience_rule(case, "serial", s["team"], p["team"])
            normal_plan = strong_rule(case)
            chosen = simulate(case, normal_plan)
            rows.append({"seed": seed, "regime": regime,
                         "serial_team": s["team"], "parallel_team": p["team"],
                         "a_observed": "serial", "a_outcome_team": s["team"],
                         "experience_choice": exp_plan,
                         "normal_choice": normal_plan,
                         "same": exp_plan == normal_plan,
                         "b_team": chosen["team"]})
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    r = {"precheck": precheck(), "dev": run_dev()}
    print(json.dumps({"precheck": len(r["precheck"]["rows"]),
                      "dev": len(r["dev"]["rows"])}, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
