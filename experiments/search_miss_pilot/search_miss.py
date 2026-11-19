#!/usr/bin/env python3
"""Search-miss pilot: minimal 2D coverage/visibility task.

A and B search different chambers.  A fast boundary scan can complete its
coverage yet miss a target hidden by an occluder; C's legal local probe
confirms the miss.  A slow segmented inspection finds the target.  The normal
control has the map, coverage/visibility map, C status and a strong coverage
rule; the experience arm gets A's Condition/Attempt/Outcome.  The probe asks
whether the experience adds a decision the strong coverage summary cannot
already make.

Stdlib only; abstract chambers, not a robot controller.
"""

import argparse
import json
from collections import deque

N = 7
ENTRY = 0  # fast scan enters from the west boundary


def line(x0, y0, x1, y1):
    """Bresenham line cells for line-of-sight visibility."""
    cells = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    while True:
        cells.append((x0, y0))
        if (x0, y0) == (x1, y1):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return cells


def visible(chamber, cell):
    # C's or A's local probe: line of sight from the entry side to the cell.
    x, y = cell
    occ = chamber["occluders"]
    for c in line(ENTRY, y, x, y):
        if c in occ and c != cell:
            return False
    return True


def fast_visible(chamber):
    return {cell for cell in chamber["cells"]
            if visible(chamber, cell)}


def fast_found(chamber):
    return chamber["target"] in fast_visible(chamber)


def slow_found(chamber):
    return chamber["target"] in chamber["cells"]


def make_chamber(seed, hidden):
    # Deterministic small layouts; occluders block the fast west-entry scan.
    x = 3 + (seed % 2)
    y = 2 + (seed % 3)
    target = (x, y)
    cells = {(i, j) for i in range(N) for j in range(N)}
    if hidden:
        # A vertical occluder between the entry and target; leave a legal
        # slow inspection path around it.
        ox = max(1, x - 1)
        occ = {(ox, j) for j in range(N) if j != (y + 1) % N}
    else:
        occ = set()
    cells -= occ
    return {"seed": seed, "target": target, "occluders": occ,
            "cells": cells, "hidden": hidden}


def normal_view(a, b):
    av = fast_visible(a)
    bv = fast_visible(b)
    return {
        "map": {"a_cells": sorted(a["cells"]), "b_cells": sorted(b["cells"]),
                "a_occluders": sorted(a["occluders"]),
                "b_occluders": sorted(b["occluders"])},
        "a_fast_coverage": sorted(av),
        "a_uncovered": sorted(a["cells"] - av),
        "b_fast_coverage": sorted(bv),
        "b_uncovered": sorted(b["cells"] - bv),
        "a_fast_miss_notice": bool(a["hidden"]),
        "c_status": "free",
    }


def experience_view(a, a_fast, a_slow):
    return {
        "condition": {"chamber": a["seed"], "hidden": a["hidden"],
                      "occluders": sorted(a["occluders"])},
        "attempt": {"plan": "fast_boundary", "completed": True,
                    "coverage": sorted(fast_visible(a))},
        "outcome": {"confirmed_miss": bool(a_fast["miss"]),
                    "slow_would_find": bool(a_slow["found"])},
    }


def simulate_a(chamber, plan):
    if plan == "fast":
        found = fast_found(chamber)
        return {"plan": plan, "coverage": sorted(fast_visible(chamber)),
                "found": found, "miss": not found, "completed": True,
                "time": 4, "uses_c": False}
    return {"plan": plan, "coverage": sorted(chamber["cells"]),
            "found": slow_found(chamber), "miss": not slow_found(chamber),
            "completed": True, "time": 9, "uses_c": True}


def simulate_b(chamber, choice):
    # B's own independent node; the same coverage facts decide its outcome.
    if choice == "fast":
        found = fast_found(chamber)
        return {"choice": choice, "found": found, "time": 4, "uses_c": False}
    found = slow_found(chamber)
    return {"choice": choice, "found": found, "time": 9, "uses_c": True}


def strong_coverage_choice(normal):
    # Strong legal coverage/visibility rule: use the uncovered-cell map, no
    # A experience needed.
    return "slow" if normal["b_uncovered"] else "fast"


def experience_choice(exp, normal):
    if exp["outcome"]["confirmed_miss"]:
        return "slow"
    return "fast"


def permuted_experience(seed):
    # Fixed sibling layout; same fields, different chamber.
    other = make_chamber(seed + 100, True)
    fast = simulate_a(other, "fast")
    slow = simulate_a(other, "slow")
    return experience_view(other, fast, slow)


def run_dev(pairs=8):
    rows = []
    for kind, offset in (("relevant", 7001), ("nonapp", 7101)):
        for seed in range(offset, offset + pairs):
            a = make_chamber(seed, hidden=(kind == "relevant"))
            b = make_chamber(seed + 50, hidden=(kind == "relevant"))
            a_fast = simulate_a(a, "fast")
            a_slow = simulate_a(a, "slow")
            nv = normal_view(a, b)
            ev = experience_view(a, a_fast, a_slow)
            for arm in ("normal", "experience", "permuted"):
                if arm == "normal":
                    choice = strong_coverage_choice(nv)
                elif arm == "experience":
                    choice = experience_choice(ev, nv)
                else:
                    choice = experience_choice(permuted_experience(seed), nv)
                res = simulate_b(b, choice)
                rows.append({
                    "seed": seed, "kind": kind, "arm": arm,
                    "a_fast_miss": a_fast["miss"],
                    "a_slow_found": a_slow["found"],
                    "a_fast_completed": a_fast["completed"],
                    "a_fast_time": a_fast["time"],
                    "a_slow_time": a_slow["time"],
                    "b_choice": choice, "b_found": res["found"],
                    "b_time": res["time"], "b_uses_c": res["uses_c"],
                    "normal_choice": strong_coverage_choice(nv),
                    "experience_choice": experience_choice(ev, nv),
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
