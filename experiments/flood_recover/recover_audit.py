#!/usr/bin/env python3
"""Read-only audit of the three failed-to-fail recheck seeds.

Does not choose a B decision time and does not simulate B.  It regenerates
the three frozen scenarios, runs A only, hashes the scenario inputs, and
records whether the frozen code has an A-independent B clock and whether the
four arms have disjoint input fields.
"""

import hashlib
import inspect
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "flood_pilot"))
sys.path.insert(0, os.path.join(HERE, "..", "flood_recheck"))
import flood_recheck as fr  # noqa: E402

SEEDS = [("relevant_pump", 7017),
         ("relevant_pump", 7022),
         ("nonapp_handled", 7304)]


def _hash(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main():
    one_src = inspect.getsource(fr.one)
    choose_src = inspect.getsource(fr._choose)
    has_a_loss_clock = 't0 = ra["loss_time"]' in one_src
    raises_on_no_failure = 'if not ra["lost"]' in one_src
    independent_clock = not (has_a_loss_clock or raises_on_no_failure)
    shared_ctx = "a_ctx" in choose_src and "exec_q" in choose_src
    order_used = "order" in choose_src
    outcome_used = "outcome" in choose_src
    rows = []
    for category, seed in SEEDS:
        scn = fr.make_scenario(seed, category)
        ra = fr.base._sim_area(scn.a, scn.shared, scn.local_a,
                               "pump_first", scn.horizon)
        input_obj = {
            "category": category, "seed": seed,
            "a": scn.a.__dict__, "b": scn.b.__dict__,
            "shared": scn.shared, "local_a": scn.local_a,
            "local_b": scn.local_b, "iso_travel": scn.iso_travel,
            "iso_work": scn.iso_work, "iso_effect": scn.iso_effect,
            "b_lag": scn.b_lag, "horizon": scn.horizon,
        }
        rows.append({
            "category": category, "seed": seed,
            "input_sha256": _hash(input_obj),
            "a_preserved": bool(ra["preserved"]),
            "a_lost": bool(ra["lost"]),
            "a_preserved_time": ra["preserved_time"],
            "a_loss_time": ra["loss_time"],
            "b_decision_time": None,
            "b_outcome": None,
            "absolute_preservation": None,
            "no_message_fallback": {
                "failure_notice": False,
                "sensor_packet": False,
                "exec_context": False,
                "arms_would_share_normal_state": True,
                "arms_actually_run": False,
                "paired_difference": None,
            },
        })
    out = {
        "frozen_code_commit": "e1c6fac",
        "source_audit": {
            "a_loss_is_b_clock": has_a_loss_clock,
            "raises_if_a_not_lost": raises_on_no_failure,
            "independent_b_clock": independent_clock,
            "shared_a_ctx_passed_to_all_arms": shared_ctx,
            "choose_uses_order": order_used,
            "choose_uses_outcome": outcome_used,
            "sensor_arm_can_recompute_exec_q_from_pump_work":
                ("pump_work" in inspect.getsource(fr.one)
                 and "exec_q" in choose_src),
        },
        "rows": rows,
        "decision": {
            "strict_recovery_possible": False,
            "reason": "no frozen B decision clock independent of A failure",
            "merged_96_run": False,
            "absolute_preservation": "missing",
            "formal_status": "undetermined",
        },
    }
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
