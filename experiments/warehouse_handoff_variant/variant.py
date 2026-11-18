"""Handoff-interface stall variant for Warehouse-A.

The only environment change is a pre-sampled, episode-seeded short interface
stall.  It is independent of the Actor, role, actions and failure labels.  A
Fetch pass that would otherwise transfer an object fails only when its
completion step falls inside a stall interval.  No action is forced and no
timer/order/reward is otherwise changed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def stable_seed(*parts):
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def sample_stall_intervals(episode_seed, max_steps=200, n_intervals=2,
                           min_length=6, max_length=14, salt=20261118):
    """Deterministic short stall intervals from the episode seed only."""
    rng = np.random.RandomState(stable_seed(episode_seed, salt) & 0x7FFFFFFF)
    intervals = []
    attempts = 0
    while len(intervals) < n_intervals and attempts < 100:
        attempts += 1
        length = int(rng.randint(min_length, max_length + 1))
        start = int(rng.randint(1, max(2, max_steps - length - 1)))
        end = start + length - 1
        if any(not (end < a or start > b) for a, b in intervals):
            continue
        intervals.append((start, end))
    return sorted(intervals)


def interval_at(intervals, step):
    for start, end in intervals:
        if start <= step <= end:
            return [int(start), int(end)]
    return None


class StallController:
    def __init__(self, intervals, enabled=True):
        self.intervals = [tuple(map(int, x)) for x in intervals]
        self.enabled = bool(enabled)
        self.triggered = []

    def active(self, step):
        return interval_at(self.intervals, step) if self.enabled else None

    def install(self, fetch, recorder, shared_state):
        """Wrap the already-installed Fetch step wrapper.

        The stall is applied before the base wrapper snapshots state, so the
        base recorder sees ``serving_failed`` and records a normal handoff
        failure.  Attribution to the stall is stored separately.
        """
        original = fetch.step

        def wrapper(action, agents):
            step = int(recorder.step)
            interval = self.active(step)
            before_action = int(fetch.cur_action.idx) if fetch.cur_action is not None else -1
            before_time = float(fetch.cur_action_time_left)
            before_done = bool(fetch.cur_action_done)
            completed = (before_action in (1, 2) and not before_done
                         and 0.0 < before_time <= 1.0 + 1e-9)
            target = before_action - 1
            causal = False
            if interval is not None and completed and target in (0, 1):
                turtle = agents[target]
                causal = (
                    not bool(fetch.serving_failed)
                    and len(fetch.found_objs) > 0
                    and float(turtle.cur_action_time_left) < 0.0
                    and turtle.cur_action is not None
                    and turtle.cur_action.name == "Get_Tool"
                )
                if causal:
                    # The same branch a not-ready interface takes; it adds the
                    # original drop penalty and leaves the object with Fetch.
                    fetch.serving_failed = True
            reward = original(action, agents)
            if causal:
                record = {
                    "type": "interface_stall_failure",
                    "stall": list(interval),
                    "step": step,
                    "target": int(target),
                    "fetch_action": int(before_action),
                    "time_left_before": before_time,
                    "basket_before": [np.asarray(t.objs_in_basket).tolist()
                                      for t in agents[:2]],
                    "basket_after": [np.asarray(t.objs_in_basket).tolist()
                                     for t in agents[:2]],
                }
                self.triggered.append(record)
                shared_state.setdefault("stall_triggered", []).append(record)
            return reward
        fetch.step = wrapper


def t1_decision_record(env, step, chosen_action, stall_controller, obs_1):
    """Record T1's real local options at a macro-action decision."""
    t1 = env.agents[1]
    basket = np.asarray(t1.objs_in_basket, dtype=np.float32).copy()
    deliver_options = []
    human_state = []
    for human in env.humans:
        h = int(human.idx)
        human_state.append({
            "agent": h,
            "cur_step": int(human.cur_step),
            "next_obj": int(human.next_request_obj_idx),
            "obtained": bool(human.next_requested_obj_obtained),
            "finished": bool(human.whole_task_finished),
            "served": h in env.finish_serve_humans,
        })
        if h in env.finish_serve_humans or human.whole_task_finished:
            continue
        if human.next_requested_obj_obtained:
            continue
        if basket[int(human.next_request_obj_idx)] > 0:
            deliver_options.append(h)
    chosen = int(chosen_action)
    chosen_delivery = chosen if chosen in (0, 1) and chosen in deliver_options else None
    return {
        "step": int(step),
        "agent": 1,
        "chosen_action": chosen,
        "chosen_tool": chosen in (2, 3),
        "chosen_delivery": chosen_delivery,
        "deliver_options": deliver_options,
        "tool_option": True,
        "two_options": bool(deliver_options),
        "stall_interval": stall_controller.active(step),
        "stall_active": bool(stall_controller.active(step) is not None),
        "basket": basket.tolist(),
        "human_state": human_state,
        "obs_1": np.asarray(obs_1, dtype=np.float32).reshape(-1).tolist(),
        "t1_bwp": int(t1.cur_BWP.idx) if t1.cur_BWP is not None else -1,
    }


def link_windows(failures, decisions):
    """For each stall failure, find T1's first real forward after it.

    A window requires both valid options, the interface still stalled, and a
    chosen action that actually uses the shared tool interface (actions 2/3).
    Non-window failures remain in the report so coverage is not lost.
    """
    linked = []
    for failure in failures:
        failure_step = int(failure["step"])
        after = [d for d in decisions if int(d["step"]) > failure_step]
        after.sort(key=lambda d: int(d["step"]))
        decision = after[0] if after else None
        row = {
            "failure_step": failure_step,
            "failure_target": int(failure["target"]),
            "stall": failure["stall"],
            "t1_decision_step": int(decision["step"]) if decision else None,
            "failure_to_decision_delay": (int(decision["step"]) - failure_step) if decision else None,
            "two_options": bool(decision["two_options"]) if decision else False,
            "deliver_options": decision["deliver_options"] if decision else [],
            "chosen_action": decision["chosen_action"] if decision else None,
            "chosen_tool": decision["chosen_tool"] if decision else None,
            "stall_active_at_decision": decision["stall_active"] if decision else False,
            "interface_relevant_choice": bool(decision["chosen_tool"]) if decision else False,
            "window": bool(decision and decision["two_options"]
                           and decision["stall_active"] and decision["chosen_tool"]),
            "decision": decision,
        }
        linked.append(row)
    return linked


def window_events(linked, episode_index, episode_seed, target=0):
    """Probe events for every stall failure; only windowed rows get a deadline.

    The probe's window is inclusive of T1's decision/forward step.  We encode
    that as an exclusive sentinel deadline (decision + 1); the actual decision
    step is kept in ``t1_decision_step`` for reporting.
    """
    events = []
    for row in linked:
        decision_step = row["t1_decision_step"]
        events.append({
            "type": "interface_handoff_failure",
            "agent": 1,
            "target": int(target),
            "step": int(row["failure_step"]),
            "next_relevant_decision_step": (int(decision_step) + 1
                                            if row["window"] and decision_step is not None else None),
            "next_any_decision_step": (int(decision_step) + 1
                                       if decision_step is not None else None),
            "first_agent_decision_step": (int(decision_step)
                                          if decision_step is not None else None),
            "t1_decision_step": int(decision_step) if decision_step is not None else None,
            "stall": row["stall"],
            "two_options": row["two_options"],
            "deliver_options": row["deliver_options"],
            "chosen_action": row["chosen_action"],
            "stall_active_at_decision": row["stall_active_at_decision"],
            "window": bool(row["window"]),
            "event_episode": int(episode_index),
            "event_seed": int(episode_seed),
        })
    return events
