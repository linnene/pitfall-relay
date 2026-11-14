"""Read-only decision and activation recorder for ORION's TestWorker.

The recorder never changes policy outputs: forward hooks clone activations and
logits, the ``select_next_waypoint`` wrapper returns the original result
unchanged, and the ``Env.test_step`` wrapper only observes. Activations are
saved exactly as observed; a missing hook produces ``activation_present=false``
rather than a fabricated tensor.
"""

import json
import time
from pathlib import Path

import numpy as np


class DecisionRecorder:
    def __init__(self, policy_net, worker, activation_dir=None, record_activations=True,
                 record_decisions=True):
        self.policy_net = policy_net
        self.worker = worker
        self.record_activations = bool(record_activations)
        self.record_decisions = bool(record_decisions)
        self.activation_dir = Path(activation_dir) if activation_dir else None
        self.step = 0
        self.decisions = []       # one entry per network call
        self.executed = []        # one entry per executed move (Env.test_step)
        self.warnings = []
        self._pending = {}        # (step, agent_id) -> decision dict
        self._ctx = None
        self._next_decision_id = 0
        self._handles = []
        self._activation_buffer = []
        self._patched = []

        if self.record_activations:
            self._handles.append(
                policy_net.current_embedding.register_forward_hook(self._embedding_hook)
            )
            self._handles.append(policy_net.register_forward_hook(self._policy_hook))

        for agent in worker.robot_list:
            if self.record_decisions:
                self._patch_select(agent)
        self._patch_env()

        self.initial = {
            "locations": np.asarray(worker.env.robot_locations, dtype=float).tolist(),
            "targets": np.asarray(worker.target_list, dtype=float).tolist(),
            "map_path": worker.map_path,
            "episode_index": int(worker.env.episode_index),
            "n_agents": len(worker.robot_list),
        }

    # ------------------------------------------------------------------ hooks
    def _embedding_hook(self, module, inputs, output):
        if self._ctx is None:
            return
        self._ctx["activation_calls"] += 1
        self._ctx["activation"] = output.detach().cpu().clone()

    def _policy_hook(self, module, inputs, output):
        if self._ctx is None:
            return
        logp, termination_logits, option = output
        self._ctx["logp"] = logp.detach().cpu().clone()
        self._ctx["termination_logits"] = termination_logits.detach().cpu().clone()
        self._ctx["current_option"] = option.detach().cpu().clone()

    # ----------------------------------------------------------------- patch
    def _patch_select(self, agent):
        original = agent.select_next_waypoint
        recorder = self

        def wrapper(observation, greedy=False):
            ctx = {
                "activation_calls": 0,
                "activation": None,
                "logp": None,
                "termination_logits": None,
                "current_option": None,
            }
            recorder._ctx = ctx
            started = time.time()
            try:
                next_position, action_index, option = original(observation, greedy=greedy)
            finally:
                recorder._ctx = None
            elapsed = time.time() - started
            recorder._record_decision(
                agent, observation, next_position, action_index, option, greedy, ctx, elapsed
            )
            return next_position, action_index, option

        agent.select_next_waypoint = wrapper
        self._patched.append(("agent", agent))

    def _patch_env(self):
        env = self.worker.env
        original_test_step = env.test_step
        original_evaluate = env.evaluate_exploration_rate
        recorder = self

        def test_step_wrapper(robot_id, next_waypoint):
            key = (recorder.step, int(robot_id))
            decision = recorder._pending.pop(key, None)
            if decision is None:
                stale = [k for k in recorder._pending if k[1] == int(robot_id)]
                if stale:
                    recorder.warnings.append(
                        f"stale network decision for agent {robot_id} at step {recorder.step}: {stale}"
                    )
            robot = recorder.worker.robot_list[robot_id]
            if decision is not None:
                kind = "network"
            elif not recorder.record_decisions:
                kind = "uninstrumented_move"
            elif not recorder.worker.reach_target[robot_id] and robot.rapid_convergence_flag:
                kind = "rapid_convergence"
            elif recorder.worker.reach_target[robot_id] and robot.back_trick:
                kind = "back_path"
            else:
                kind = "other_bypass"

            before = np.asarray(robot.location, dtype=float).copy()
            original_test_step(robot_id, next_waypoint)
            after = np.asarray(robot.location, dtype=float).copy()
            entry = {
                "step": recorder.step,
                "agent_id": int(robot_id),
                "kind": kind,
                "decision_id": decision["decision_id"] if decision else None,
                "from": before.tolist(),
                "to": np.asarray(next_waypoint, dtype=float).reshape(-1).tolist(),
                "after": after.tolist(),
                "reach_target": int(recorder.worker.reach_target[robot_id]),
                "explore_over": int(recorder.worker.explore_over[robot_id]),
                "rapid_convergence_flag": bool(robot.rapid_convergence_flag),
                "back_trick": bool(robot.back_trick),
                "finding_target": [int(x) for x in recorder.worker.env.finding_target],
            }
            recorder.executed.append(entry)

        def evaluate_wrapper():
            original_evaluate()
            recorder._close_step()
            recorder.step += 1

        env.test_step = test_step_wrapper
        env.evaluate_exploration_rate = evaluate_wrapper
        self._patched.append(("env", env))

    # ------------------------------------------------------------- recording
    def _record_decision(self, agent, observation, next_position, action_index,
                         option, greedy, ctx, elapsed):
        decision_id = self._next_decision_id
        self._next_decision_id += 1

        prev_option = observation[-1]
        current_index = observation[3]
        current_edge = observation[4]
        action = int(action_index.reshape(-1)[0]) if hasattr(action_index, "reshape") else int(action_index)
        next_node_index = int(current_edge[0, action, 0])
        expected_position = np.asarray(agent.node_coords[next_node_index], dtype=float).reshape(-1)
        returned_position = np.asarray(next_position, dtype=float).reshape(-1)
        option_value = option
        if hasattr(option_value, "reshape"):
            option_value = int(option_value.reshape(-1)[0])
        else:
            option_value = int(option_value)

        record = {
            "decision_id": decision_id,
            "step": self.step,
            "agent_id": int(agent.id),
            "greedy": bool(greedy),
            "phase": "reached_target" if self.worker.reach_target[agent.id] else "en_route",
            "observation_shapes": [
                list(x.shape) if hasattr(x, "shape") else None for x in observation
            ],
            "current_index": int(current_index.reshape(-1)[0]),
            "num_edges": int(current_edge.shape[1]),
            "prev_option": int(prev_option.reshape(-1)[0]) if hasattr(prev_option, "reshape") else int(prev_option),
            "action_index": action,
            "next_node_index": next_node_index,
            "next_position": returned_position.tolist(),
            "expected_next_position": expected_position.tolist(),
            "next_node_matches_returned_position": bool(np.allclose(expected_position, returned_position)),
            "option": option_value,
            "option_switched": bool(option_value != (int(prev_option.reshape(-1)[0]) if hasattr(prev_option, "reshape") else int(prev_option))),
            "agent_location": np.asarray(agent.location, dtype=float).reshape(-1).tolist(),
            "agent_target": np.asarray(agent.target, dtype=float).reshape(-1).tolist(),
            "select_elapsed_s": elapsed,
            "activation_present": False,
            "activation_shape": None,
            "activation_finite": None,
            "activation_l2": None,
            "logp_present": False,
            "logp_argmax": None,
            "argmax_matches_action": None,
            "termination_prob": None,
        }

        if ctx["activation"] is not None:
            activation = ctx["activation"].numpy()
            finite = bool(np.isfinite(activation).all())
            record["activation_present"] = True
            record["activation_shape"] = list(activation.shape)
            record["activation_finite"] = finite
            record["activation_l2"] = float(np.linalg.norm(activation))
            self._activation_buffer.append(
                (decision_id, activation.reshape(-1).astype(np.float32))
            )
            if ctx["activation_calls"] != 1:
                self.warnings.append(
                    f"decision {decision_id}: current_embedding hook fired {ctx['activation_calls']} times"
                )
        else:
            self.warnings.append(f"decision {decision_id}: no current_embedding activation captured")

        if ctx["logp"] is not None:
            logp = ctx["logp"].numpy().reshape(-1)
            record["logp_present"] = True
            record["logp_shape"] = list(ctx["logp"].shape)
            record["logp_finite"] = bool(np.isfinite(logp).all())
            record["logp_argmax"] = int(np.argmax(logp))
            record["argmax_matches_action"] = bool(int(np.argmax(logp)) == action)
            record["logp"] = [float(x) for x in logp]
            if ctx["termination_logits"] is not None:
                t = float(ctx["termination_logits"].reshape(-1)[0])
                record["termination_prob"] = float(1.0 / (1.0 + np.exp(-t)))

        self.decisions.append(record)
        self._pending[(self.step, int(agent.id))] = record

    def _close_step(self):
        leftovers = [k for k in self._pending if k[0] < self.step]
        if leftovers:
            self.warnings.append(f"unexecuted network decisions before step {self.step}: {leftovers}")
            for key in leftovers:
                self._pending.pop(key, None)

    # ---------------------------------------------------------------- finish
    def finalize(self):
        self._close_step()
        worker = self.worker
        per_agent = {}
        for agent in worker.robot_list:
            agent_id = int(agent.id)
            executed = [e for e in self.executed if e["agent_id"] == agent_id]
            counts = {"network": 0, "rapid_convergence": 0, "back_path": 0,
                      "other_bypass": 0, "uninstrumented_move": 0}
            for entry in executed:
                counts[entry["kind"]] = counts.get(entry["kind"], 0) + 1
            network_calls = sum(1 for d in self.decisions if d["agent_id"] == agent_id)
            per_agent[str(agent_id)] = {
                "instrumented": self.record_decisions,
                "actual_decisions_executed": len(executed),
                "network_decisions": network_calls,
                "rapid_convergence": counts["rapid_convergence"],
                "back_path": counts["back_path"],
                "other_bypass": counts["other_bypass"],
                "network_coverage_of_actual": (
                    network_calls / len(executed)
                    if executed and self.record_decisions else None
                ),
                "bypass_decisions": (
                    len(executed) - network_calls if self.record_decisions else None
                ),
                "bypass_share_of_actual": (
                    (len(executed) - network_calls) / len(executed)
                    if executed and self.record_decisions else None
                ),
                "reach_target": int(worker.reach_target[agent_id]),
                "explore_over": int(worker.explore_over[agent_id]),
                "travel_dist": float(agent.travel_dist),
                "final_location": np.asarray(agent.location, dtype=float).reshape(-1).tolist(),
                "target": np.asarray(agent.target, dtype=float).reshape(-1).tolist(),
            }

        activation_stats = {
            "decisions": len(self.decisions),
            "with_activation": sum(1 for d in self.decisions if d["activation_present"]),
            "all_finite": all(
                d["activation_finite"] for d in self.decisions if d["activation_present"]
            ),
            "all_argmax_match_action": all(
                d["argmax_matches_action"] for d in self.decisions if d["logp_present"]
            ),
            "all_next_node_match_position": all(
                d["next_node_matches_returned_position"] for d in self.decisions
            ),
        }

        summary = {
            "initial": self.initial,
            "metrics": dict(worker.perf_metrics),
            "success": bool(worker.perf_metrics.get("success_rate", False)),
            "total_step": int(worker.perf_metrics.get("total_step", 0)),
            "per_agent": per_agent,
            "activation_stats": activation_stats,
            "executed": self.executed,
            "executed_count": len(self.executed),
            "decision_count": len(self.decisions),
            "warnings": self.warnings,
            "final_finding_target": [int(x) for x in worker.env.finding_target],
            "explored_rate": float(worker.env.explored_rate),
        }
        return summary

    def activation_array(self):
        if not self._activation_buffer:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        ids = np.array([x[0] for x in self._activation_buffer], dtype=np.int64)
        acts = np.stack([x[1] for x in self._activation_buffer])
        return ids, acts

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        for kind, obj in reversed(self._patched):
            if kind == "agent":
                obj.__dict__.pop("select_next_waypoint", None)
            elif kind == "env":
                obj.__dict__.pop("test_step", None)
                obj.__dict__.pop("evaluate_exploration_rate", None)
        self._patched = []


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
