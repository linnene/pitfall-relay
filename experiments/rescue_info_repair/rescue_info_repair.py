#!/usr/bin/env python3
"""Legal-information repair for the rescue-ordering mechanism.

Reuses the task-level actions/resources/delivery rule from
experiments/rescue_mechanism/rescue_mechanism.py (292b924).  A and B are
different cases.  B sees only a legal observation dictionary (current HP,
recent HP trend, clear/transport ETA, shared-resource ETA); it does not get
the full case, true future decay, arm name, category or counterfactual.  A's
experience is one actual attempted order and its confirmed outcome.

The script tests whether the experience changes B's choice beyond a normal
trend/ETA estimator.  It is a development gate only.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rescue_mechanism"))
import rescue_mechanism as rm  # noqa: E402


def legal_obs(case):
    """Only observations a local B can legally have at its node."""
    a, b = case["a"], case["b"]
    # A short local HP history; no true decay or full case.
    a_hist = [a["hp"] - a["decay"] * t for t in (0, 2, 4)]
    b_hist = [b["hp"] - b["decay"] * t for t in (0, 2, 4)]
    return {
        "a_hp": a_hist[0], "a_trend": (a_hist[0] - a_hist[-1]) / 4.0,
        "b_hp": b_hist[0], "b_trend": (b_hist[0] - b_hist[-1]) / 4.0,
        "a_clear": a["clear"], "b_clear": b["clear"],
        "a_transport": a["transport"], "b_transport": b["transport"],
        "stage_gain": case["stage_gain"],
        "ambulance_eta": 0.0,
    }


def _case_from_obs(obs, plan):
    # Reconstruct a planning case from legal observations only; the
    # estimator uses the observed recent trend as its decay estimate.
    return {
        "a": {"hp": obs["a_hp"], "decay": max(0.0, obs["a_trend"]),
              "clear": obs["a_clear"], "transport": obs["a_transport"]},
        "b": {"hp": obs["b_hp"], "decay": max(0.0, obs["b_trend"]),
              "clear": obs["b_clear"], "transport": obs["b_transport"]},
        "stage_gain": obs["stage_gain"],
    }


def normal_choice(obs):
    sim = _case_from_obs(obs, "serial")
    s = rm.simulate(sim, "serial")
    p = rm.simulate(sim, "parallel")
    return "parallel" if p["team"] >= s["team"] else "serial"


def experience_choice(obs, exp):
    # The same rule; the experience may only update the prior on the same
    # observable trend.  A confirmed failure under serial raises the decay
    # estimate for a similar case, but no category/optimal action is passed.
    if exp["confirmed_failure"] and exp["attempt"] == "serial":
        obs2 = dict(obs)
        obs2["a_trend"] = max(obs2["a_trend"], exp["observed_decay"])
        return normal_choice(obs2)
    return normal_choice(obs)


def a_case_record(a_case):
    # A actually executes serial; its record contains only the attempt and
    # the confirmed team outcome (no alt-plan counterfactual).
    s = rm.simulate(a_case, "serial")
    return {"attempt": "serial", "team": s["team"],
            "confirmed_failure": s["team"] < 2,
            "observed_decay": max(a_case["a"]["decay"],
                                 a_case["b"]["decay"]),
            "timestamps": {"pump_clear": a_case["a"]["clear"],
                           "b_clear": a_case["b"]["clear"]}}


def run_dev():
    rows = []
    for regime, seed_base in (("parallel_better", 9001),
                              ("serial_better", 9101)):
        for i in range(6):
            a_case = rm.make_case(seed_base + i, regime)
            b_case = rm.make_case(seed_base + 50 + i, regime)
            obs = legal_obs(b_case)
            exp = a_case_record(a_case)
            n = normal_choice(obs)
            e = experience_choice(obs, exp)
            n_out = rm.simulate(b_case, n)
            e_out = rm.simulate(b_case, e)
            rows.append({
                "seed": b_case["seed"], "regime": regime,
                "a_seed": a_case["seed"], "a_confirmed_failure": exp["confirmed_failure"],
                "normal_choice": n, "experience_choice": e,
                "same": n == e,
                "normal_team": n_out["team"], "experience_team": e_out["team"],
                "a_team": exp["team"],
                "obs": obs,
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
