#!/usr/bin/env python3
"""Fact gate for the flood task: independent B node and a local work fact.

A's locally measured pump outflow and isolation closure are not in normal
telemetry.  B decides at its own fixed inspection node, independently of A.
The same decision function is used with different message payloads.

Pure stdlib.  Abstract water/damage mechanics; no simulator truth is passed to
a policy.
"""

import argparse
import json
import math
import random
import statistics

ARMS = ("baseline", "fact", "permuted")
T_B = 10
T_END = 16


def _levels(initial, r0, r1, t0=0, split=4, n=10):
    out = []
    x = initial
    for t in range(n):
        out.append(x)
        x += r0 if t < split else r1
    return out


def _damage(levels, threshold, rate):
    d = 0.0
    for x in levels:
        d += rate * max(0.0, x - threshold)
    return d


def scenario(seed, kind):
    """kind: relevant, nonapp, notimely."""
    rng = random.Random(seed * 100003 + {"relevant": 1, "nonapp": 2,
                                         "notimely": 3}[kind])
    # A's normal water trace is identical for relevant and nonapp.
    a_levels = _levels(0.10, 0.60, 0.36)
    if kind == "relevant":
        a_pump, a_closure = 0.60, 0.80
        b_future = "shared_high"
    elif kind == "nonapp":
        a_pump, a_closure = 0.00, 0.80
        b_future = "local_only"
    else:
        a_pump, a_closure = rng.choice([(0.60, 0.80), (0.00, 0.80)])
        b_future = rng.choice(["shared_high", "local_only"])
    # A equipment outcome is hidden behind a random damage threshold/rate;
    # both success and failure occur in the frozen set.
    # Same A outcome for a paired relevant/nonapp seed (pair id = seed % 100).
    a_dr = random.Random((seed % 100) * 7919 + 99).uniform(0.035, 0.075)
    a_thr = 0.12
    a_damage = _damage(a_levels, a_thr, a_dr)
    a_preserved = a_damage < 1.0
    # B's normal trace and decision-time state are the same across kinds.
    b_levels = _levels(0.15, 0.04, 0.04)
    b_damage0 = 0.45
    b_future_damage = {
        "shared_high": {"transport": 0.62, "isolate": 0.08},
        "local_only": {"transport": 0.10, "isolate": 0.14},
    }[b_future]
    return {
        "seed": seed, "kind": kind, "b_future": b_future,
        "a_levels": a_levels, "a_pump": a_pump, "a_closure": a_closure,
        "a_damage": a_damage, "a_preserved": a_preserved,
        "b_levels": b_levels, "b_damage0": b_damage0,
        "b_future_damage": b_future_damage,
    }


def _normal_input(scn, with_notice):
    return {
        "a_levels": list(scn["a_levels"]),
        "a_pump_status": "on",
        "a_isolation_status": "on",
        "b_levels": list(scn["b_levels"]),
        "b_damage": scn["b_damage0"],
        "a_failure_notice": bool(with_notice and not scn["a_preserved"]),
    }


def _fact_input(scn):
    return {"pump_outflow": scn["a_pump"],
            "isolation_closure": scn["a_closure"],
            "before_levels": scn["a_levels"][:5],
            "after_levels": scn["a_levels"][4:9]}


def _estimate_qs(fact):
    before = fact["before_levels"]
    after = fact["after_levels"]
    r0 = (before[-1] - before[0]) / max(1, len(before) - 1)
    r1 = (after[-1] - after[0]) / max(1, len(after) - 1)
    closure = max(0.05, fact["isolation_closure"])
    ql = max(0.0, (r0 - r1) / closure)
    return max(0.0, r0 + fact["pump_outflow"] - ql)


def decide(arm, normal, fact, base_qs, own_estimate=True):
    """Same decision function for all arms; only payload differs."""
    # Baseline uses its own normal-state estimate: B's recent trend and the
    # visible A trace, capped as in the prior task's strong rule.
    b_levels = normal["b_levels"]
    recent = (b_levels[-1] - b_levels[-4]) / 3.0
    q_b = max(0.02, recent * 1.0)
    qs = q_b
    if arm == "fact" and fact is not None:
        qs = _estimate_qs(fact)
    elif arm == "permuted" and fact is not None:
        qs = _estimate_qs(fact)  # caller passes already-permuted fact
    # Common action rule: if estimated shared flow is high, isolate; else
    # transport.  Prediction is explicit and identical across arms.
    if qs >= 0.60:
        return "isolate", qs
    return "transport", qs


def outcome(scn, action):
    d = scn["b_damage0"] + scn["b_future_damage"][action]
    lost = d >= 1.0
    return {"action": action, "damage": d, "preserved": not lost,
            "lost": lost}


def row(seed, kind, arm, fact, permuted_fact):
    scn = scenario(seed, kind)
    normal = _normal_input(scn, with_notice=True)
    if kind == "notimely" or arm == "baseline":
        f = None
    elif arm == "permuted":
        f = permuted_fact
    else:
        f = fact
    action, qs = decide(arm, normal, f, None)
    res = outcome(scn, action)
    return {
        "seed": seed, "kind": kind, "arm": arm,
        "a_preserved": scn["a_preserved"],
        "a_damage": scn["a_damage"],
        "q_est": qs, "action": action, "b_preserved": res["preserved"],
        "b_lost": res["lost"], "b_damage": res["damage"],
        "normal": normal,
        "fact_visible": f is not None,
    }


def _permutation_map(seeds):
    """Fixed cross-scenario permutation: adjacent seeds swap."""
    out = {}
    for i, s in enumerate(seeds):
        out[s] = seeds[(i + 1) % len(seeds)]
    return out


def run(kind_seeds, out_path=None):
    # Fact payloads for the fixed permutation use the mapped seed's scenario.
    all_rows = []
    kinds = list(kind_seeds)
    for ki, kind in enumerate(kinds):
        other = kinds[(ki + 1) % len(kinds)]
        other_seeds = kind_seeds[other]
        seeds = kind_seeds[kind]
        for i, seed in enumerate(seeds):
            # Real fact from this scenario; permuted fact from a fixed
            # cross-kind scenario, never from B truth.
            real_fact = _fact_input(scenario(seed, kind))
            perm_seed = other_seeds[i % len(other_seeds)]
            perm_fact = _fact_input(scenario(perm_seed, other))
            for arm in ARMS:
                all_rows.append(row(seed, kind, arm, real_fact, perm_fact))
    summary = {}
    for kind in kind_seeds:
        rr = [r for r in all_rows if r["kind"] == kind]
        summary[kind] = {}
        for arm in ARMS:
            aa = [r for r in rr if r["arm"] == arm]
            summary[kind][arm] = {
                "n": len(aa),
                "b_loss": sum(r["b_lost"] for r in aa),
                "b_preserved": sum(r["b_preserved"] for r in aa),
                "wrong_isolate": sum(1 for r in aa
                                     if r["action"] == "isolate"
                                     and r["kind"] != "relevant"),
                "a_preserved": sum(r["a_preserved"] for r in aa),
                "team_preserved": sum(r["a_preserved"] + r["b_preserved"] for r in aa),
                "all_preserved": sum(r["a_preserved"] and r["b_preserved"] for r in aa),
            }
    result = {"rows": all_rows, "summary": summary}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1, sort_keys=True)
    return result


def dev():
    seeds = {
        "relevant": list(range(4001, 4009)),
        "nonapp": list(range(4101, 4109)),
    }
    return run(seeds)


def _bootstrap_ci(diffs, rng, n=2000):
    if not diffs:
        return [0.0, 0.0]
    vals = []
    for _ in range(n):
        vals.append(sum(rng.choice(diffs) for _ in diffs) / len(diffs))
    vals.sort()
    return [vals[int(0.025 * n)], vals[int(0.975 * n)]]


def holdout():
    seeds = {
        "relevant": list(range(9001, 9025)),
        "nonapp": list(range(9101, 9125)),
        "notimely": list(range(9201, 9225)),
    }
    r = run(seeds)
    s = r["summary"]
    rel = {a: s["relevant"][a]["b_loss"] for a in ARMS}
    non = {a: s["nonapp"][a]["b_loss"] for a in ARMS}
    rel_team = {a: s["relevant"][a]["team_preserved"] for a in ARMS}
    non_team = {a: s["nonapp"][a]["team_preserved"] for a in ARMS}
    non_wrong = {a: s["nonapp"][a]["wrong_isolate"] for a in ARMS}
    relevant_rows = [x for x in r["rows"]
                     if x["kind"] == "relevant" and x["arm"] == "fact"]
    diffs = []
    for x in relevant_rows:
        seed = x["seed"]
        b = [y for y in r["rows"] if y["kind"] == "relevant"
             and y["seed"] == seed and y["arm"] == "baseline"][0]
        diffs.append(int(x["b_lost"]) - int(b["b_lost"]))
    by_seed = {}
    for x in r["rows"]:
        if x["kind"] == "notimely":
            by_seed.setdefault(x["seed"], set()).add((x["action"], x["b_lost"]))
    notimely_equal = all(len(v) == 1 for v in by_seed.values())
    checks = {
        "relevant_gain_fact_vs_baseline": rel["baseline"] - rel["fact"],
        "relevant_ci95_fact_minus_baseline": _bootstrap_ci(diffs, random.Random(11)),
        "relevant_ci_excludes_zero": _bootstrap_ci(diffs, random.Random(11))[1] < 0,
        "relevant_fact_better_than_permuted": rel["fact"] <= rel["permuted"],
        "relevant_team_not_lower": rel_team["fact"] >= rel_team["baseline"],
        "nonapp_extra_wrong": non_wrong["fact"] - non_wrong["baseline"],
        "nonapp_team_not_lower": non_team["fact"] >= non_team["baseline"],
        "notimely_arms_identical": notimely_equal,
    }
    checks["thresholds_met"] = (
        checks["relevant_gain_fact_vs_baseline"] >= 4
        and checks["relevant_ci_excludes_zero"]
        and checks["relevant_fact_better_than_permuted"]
        and checks["relevant_team_not_lower"]
        and checks["nonapp_extra_wrong"] <= 2
        and checks["nonapp_team_not_lower"]
        and checks["notimely_arms_identical"])
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
