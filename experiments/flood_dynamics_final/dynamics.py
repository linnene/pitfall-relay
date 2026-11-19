#!/usr/bin/env python3
"""Final flood fact test with native water evolution.

No outcome tables: A and B water volumes, pump drainage, isolation inflow
reduction and equipment damage are evolved step by step.  The only new
logic is connecting the frozen external inflow process and the executed
pump/isolation actions to that evolution.
"""

import argparse
import json
import random

ARMS = ("baseline", "fact", "permuted")
T_B = 10
T_END = 18


def _evolve(levels, damage, shared, local, pump_rate, pump_on,
            isolation_at, closure, threshold, damage_rate, start=0,
            isolate_shared=True, steps=10):
    """One-minute nonnegative volume/damage evolution.

    Isolation reduces only the declared scope: the shared inlet for B, the
    local inlet for A.  The other component continues.
    """
    w = levels[-1]
    d = damage
    n = (len(shared) if isinstance(shared, list)
         else (len(local) if isinstance(local, list) else steps))
    for k in range(n):
        t = start + k
        qs = shared[k] if isinstance(shared, list) else shared
        ql = local[k] if isinstance(local, list) else local
        isolated = isolation_at is not None and t >= isolation_at
        if isolated:
            if isolate_shared:
                qs *= (1.0 - closure)
            else:
                ql *= (1.0 - closure)
        w += qs + ql
        if pump_on:
            w -= min(pump_rate, w)
        w = max(0.0, w)
        d += damage_rate * max(0.0, w - threshold)
        levels.append(w)
        if d >= 1.0:
            return levels, 1.0, True
    return levels, d, False


def scenario(seed, kind):
    rng = random.Random(seed * 100019 + {"relevant": 1, "nonapp": 2,
                                         "notimely": 3}[kind])
    # A: two parameterizations with the same normal water trace.
    if kind == "relevant":
        qs, ql, p, c = 0.10, 0.02, 0.06, 0.80
    elif kind == "nonapp":
        qs, ql, p, c = 0.03, 0.03, 0.00, 0.5333333333333333
    else:
        qs, ql, p, c = rng.choice([(0.10, 0.02, 0.06, 0.80),
                                   (0.03, 0.03, 0.00, 0.5333333333333333)])
    # A pump 0..8, isolation 4..10, common external trace.
    a_levels = [0.10]
    a_damage = 0.0
    a_levels, a_damage, a_lost = _evolve(
        a_levels, a_damage, qs, ql, p, True, 4, c, 0.20, 0.12,
        isolate_shared=False)
    # Keep the A outcome varied but paired by seed, using a second damage
    # integration on the same observable trace (not a table).
    a_dr = random.Random((seed % 100) * 7919 + 99).uniform(0.25, 0.55)
    a_damage = 0.0
    for x in a_levels:
        a_damage += a_dr * max(0.0, x - 0.20)
    a_preserved = a_damage < 1.0
    # B normal trace: same total normal rate in both kinds.
    b_q = {"relevant": 0.10, "nonapp": 0.03,
           "notimely": qs}[kind] if kind != "notimely" else qs
    b_ql = {"relevant": 0.00, "nonapp": 0.07, "notimely": 0.0}[kind]
    b_p = 0.09
    b_levels = [0.15]
    b_damage = 0.0
    b_levels, b_damage, b_lost0 = _evolve(
        b_levels, b_damage, b_q, b_ql, b_p, True, None, 0.0, 0.18, 0.5)
    # Future: shared differs; local after t=10 is small.
    future_q = {"relevant": 0.20, "nonapp": 0.03,
                "notimely": qs}[kind]
    future_local = 0.02
    a_levels = [round(x, 9) for x in a_levels]
    b_levels = [round(x, 9) for x in b_levels]
    return {
        "seed": seed, "kind": kind,
        "a_levels": a_levels, "a_p": p, "a_closure": c,
        "a_damage": a_damage, "a_preserved": a_preserved,
        "b_levels": b_levels, "b_damage": b_damage,
        "b_q": b_q, "b_ql": b_ql, "b_p": b_p,
        "future_q": future_q, "future_local": future_local,
        "threshold": 0.18, "b_damage_rate": 0.5,
    }


def normal_input(scn):
    return {
        "a_levels": list(scn["a_levels"]),
        "a_pump_status": "on",
        "a_isolation_status": "on",
        "a_failure_notice": not scn["a_preserved"],
        "b_levels": list(scn["b_levels"]),
        "b_damage": scn["b_damage"],
    }


def fact_input(scn):
    return {"pump_outflow": scn["a_p"], "isolation_closure": scn["a_closure"],
            "levels": list(scn["a_levels"])}


def estimate_qs(fact):
    lv = fact["levels"]
    r0 = (lv[4] - lv[0]) / 4.0
    r1 = (lv[9] - lv[4]) / 5.0
    c = max(0.05, fact["isolation_closure"])
    ql = max(0.0, (r0 - r1) / c)
    return max(0.0, r0 + fact["pump_outflow"] - ql)


def decide(arm, normal, fact):
    # Baseline's legal local forecast uses only normal telemetry.  It sees a
    # flat B trace, so it chooses transport; the fact arm can see the applied
    # work and estimate residual shared inflow.
    b = normal["b_levels"]
    trend = (b[-1] - b[-4]) / 3.0
    q = max(0.0, trend)
    if arm == "fact" and fact is not None:
        q = estimate_qs(fact)
    elif arm == "permuted" and fact is not None:
        q = estimate_qs(fact)
    return ("isolate" if q >= 0.06 else "transport"), q


def run_b(scn, action):
    # Water at decision time; action-specific evolution.
    levels = list(scn["b_levels"])
    damage = scn["b_damage"]
    if action == "transport":
        total = 5
        iso_at = None
        closure = 0.0
    else:
        total = 7  # 2-min request/travel + 5-min transport
        iso_at = T_B + 2
        closure = 1.0
    # Re-run with B pump on and isolation on the shared inlet.
    levels, damage, lost = _evolve(
        levels, damage, scn["future_q"], scn["future_local"],
        scn["b_p"], True, iso_at, closure,
        scn["threshold"], scn["b_damage_rate"], start=T_B,
        isolate_shared=True, steps=total)
    preserved = (not lost) and damage < 1.0
    return {"action": action, "damage": damage, "preserved": preserved,
            "lost": lost, "done": T_B + total,
            "isolation_request": action == "isolate"}


def row(scn, arm, fact, perm_fact):
    normal = normal_input(scn)
    if scn["kind"] == "notimely" or arm == "baseline":
        f = None
    elif arm == "permuted":
        f = perm_fact
    else:
        f = fact
    action, q = decide(arm, normal, f)
    res = run_b(scn, action)
    # Counterfactual transport for wrong-isolation accounting.
    alt = run_b(scn, "transport")
    return {
        "seed": scn["seed"], "kind": scn["kind"], "arm": arm,
        "a_preserved": scn["a_preserved"], "a_damage": scn["a_damage"],
        "q_est": q, "action": action, "b_preserved": res["preserved"],
        "b_lost": res["lost"], "b_damage": res["damage"],
        "done": res["done"], "isolation_request": res["isolation_request"],
        "wrong_isolation": bool(action == "isolate" and alt["preserved"]),
        "fact_visible": f is not None,
    }


def run(kind_seeds, out_path=None):
    rows = []
    kinds = list(kind_seeds)
    for ki, kind in enumerate(kinds):
        other = kinds[(ki + 1) % len(kinds)]
        for i, seed in enumerate(kind_seeds[kind]):
            scn = scenario(seed, kind)
            real_fact = fact_input(scn)
            pseed = kind_seeds[other][i % len(kind_seeds[other])]
            perm_fact = fact_input(scenario(pseed, other))
            for arm in ARMS:
                rows.append(row(scn, arm, real_fact, perm_fact))
    summary = {}
    for kind in kinds:
        rr = [x for x in rows if x["kind"] == kind]
        summary[kind] = {}
        for arm in ARMS:
            aa = [x for x in rr if x["arm"] == arm]
            summary[kind][arm] = {
                "n": len(aa),
                "b_loss": sum(x["b_lost"] for x in aa),
                "b_preserved": sum(x["b_preserved"] for x in aa),
                "a_preserved": sum(x["a_preserved"] for x in aa),
                "team_preserved": sum(x["a_preserved"] + x["b_preserved"] for x in aa),
                "all_preserved": sum(x["a_preserved"] and x["b_preserved"] for x in aa),
                "wrong_isolation": sum(x["wrong_isolation"] for x in aa),
                "mean_done": sum(x["done"] for x in aa) / len(aa),
            }
    result = {"rows": rows, "summary": summary}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1, sort_keys=True)
    return result


def _bootstrap_ci(diffs, rng, n=2000):
    if not diffs:
        return [0.0, 0.0]
    vals = []
    for _ in range(n):
        vals.append(sum(rng.choice(diffs) for _ in diffs) / len(diffs))
    vals.sort()
    return [vals[int(0.025 * n)], vals[int(0.975 * n)]]


def dev():
    return run({"relevant": list(range(4001, 4009)),
                "nonapp": list(range(4101, 4109))})


def holdout():
    r = run({"relevant": list(range(9001, 9025)),
             "nonapp": list(range(9101, 9125)),
             "notimely": list(range(9201, 9225))})
    s = r["summary"]
    rel = {a: s["relevant"][a]["b_loss"] for a in ARMS}
    non = {a: s["nonapp"][a]["b_loss"] for a in ARMS}
    rel_team = {a: s["relevant"][a]["team_preserved"] for a in ARMS}
    non_team = {a: s["nonapp"][a]["team_preserved"] for a in ARMS}
    non_wrong = {a: s["nonapp"][a]["wrong_isolation"] for a in ARMS}
    diffs = []
    for x in r["rows"]:
        if x["kind"] == "relevant" and x["arm"] == "fact":
            b = [y for y in r["rows"] if y["kind"] == "relevant"
                 and y["seed"] == x["seed"] and y["arm"] == "baseline"][0]
            diffs.append(int(x["b_lost"]) - int(b["b_lost"]))
    by_seed = {}
    for x in r["rows"]:
        if x["kind"] == "notimely":
            by_seed.setdefault(x["seed"], set()).add((x["action"], x["b_lost"]))
    checks = {
        "relevant_gain_fact_vs_baseline": rel["baseline"] - rel["fact"],
        "relevant_ci95": _bootstrap_ci(diffs, random.Random(13)),
        "relevant_ci_excludes_zero": _bootstrap_ci(diffs, random.Random(13))[1] < 0,
        "relevant_fact_better_than_permuted": rel["fact"] <= rel["permuted"],
        "relevant_team_not_lower": rel_team["fact"] >= rel_team["baseline"],
        "nonapp_extra_wrong": non_wrong["fact"] - non_wrong["baseline"],
        "nonapp_team_not_lower": non_team["fact"] >= non_team["baseline"],
        "notimely_equal": all(len(v) == 1 for v in by_seed.values()),
    }
    checks["thresholds_met"] = (
        checks["relevant_gain_fact_vs_baseline"] >= 4
        and checks["relevant_ci_excludes_zero"]
        and checks["relevant_fact_better_than_permuted"]
        and checks["relevant_team_not_lower"]
        and checks["nonapp_extra_wrong"] <= 2
        and checks["nonapp_team_not_lower"]
        and checks["notimely_equal"])
    r["checks"] = checks
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["dev", "holdout"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    r = dev() if args.command == "dev" else holdout()
    print(json.dumps(r["summary"], indent=1, sort_keys=True))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
