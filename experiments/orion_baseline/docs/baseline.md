# ORION public-checkpoint execution probe

Question: can the published ORION cooperative policy, without FLEX, complete
its original three-agent exploration task, and which execution decisions have
usable `PolicyNet.current_embedding` activations?

## Provenance and contract

| Item | Value |
| --- | --- |
| Upstream | `https://github.com/marmotlab/ORION` |
| Commit | `a935c4bab6aed90e36a24d596bfb9a74a8b234e9` |
| Weight | `model/checkpoint.pth`, 73,454,665 bytes |
| Weight SHA256 | `f8147246d288bb9ed1bf5710a0f6a17a9a255f6bd7747d4d361695dc2896b8e5` |
| Checkpoint episode | 49,984 |
| Policy | `PolicyNet(9, 128)`, 2,852,803 parameters, 184 state-dict entries |
| Load | `checkpoint["policy_model"]` + `load_state_dict(strict=True)`; no missing or unexpected keys |
| Upstream default | `test_driver.py` requests `model/09_30_0.051/checkpoint_50000.pth`; that path is absent, so this adapter passes the public path explicitly |
| License | MIT, Copyright (c) 2025 MARMot Lab @ NUS-ME |

No upstream source file is modified. The adapter imports `TestWorker` from the
pinned checkout and runs it in one CPU process; upstream's Ray driver is not
used.

## Protocol

- Three agents (`TEST_N_AGENTS = 3`), `greedy=True`, `MAX_EPISODE_STEP = 128`.
- Map selected by `episode_index % 50` over the 50 sorted
  `maps_priori_test_new_3/img_*.png`; ground truth is the matching
  `maps_GT_test_new_3/img_*.png`.
- Seeds fixed for `random`, `numpy`, `torch`; one torch thread; no training.
- Success is upstream's own definition: `all(explore_over)` at the end of an
  episode, i.e. every agent reached its target after the team found all three
  targets.
- Every executed move passes through `Env.test_step`. A move is a **network
  decision** if `Agent.select_next_waypoint` was called for that agent in the
  same loop iteration; otherwise upstream flags classify it as
  **rapid_convergence** (`robot.path` A* shortcut), **back_path**
  (`robot.back_path`), or **other_bypass**. Idle `explore_over` agents are not
  decisions.

## Recording

- Forward hook on `PolicyNet.current_embedding` clones its output; forward
  hook on `PolicyNet` clones `logp`, termination logits and the selected
  option. Hooks do not modify any tensor.
- A wrapper around `Agent.select_next_waypoint` returns the original tuple
  unchanged and records observation shapes/indices, returned position, action
  index, inferred next node, option and termination probability.
- Alignment checks: hook fired exactly once per network call; activation is
  finite; `argmax(logp)` equals the executed action index; the node reached by
  that index equals the returned waypoint.
- Hook-off runs (`--mode nohook`) install no PolicyNet hooks and do not wrap
  `select_next_waypoint`; only a passive `Env.test_step` logger records
  positions, so the same map/seed can be compared.

## Results

Five fixed maps (indices 0-4, `img_0.png` ... `img_4.png`), one seed, greedy
decoding. All five episodes completed the original task: every agent ended
with `explore_over = 1` after reaching its assigned target.

| Map index | Ground-truth map | Success | Steps | Wall time (s) | Network / actual | Rapid | Back |
| ---: | --- | :---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `maps_GT_test_new_3/img_0.png` | yes | 60 | 283.0 | 168 / 171 (98.25%) | 3 | 0 |
| 1 | `maps_GT_test_new_3/img_1.png` | yes | 103 | 403.4 | 211 / 219 (96.35%) | 8 | 0 |
| 2 | `maps_GT_test_new_3/img_2.png` | yes | 71 | 272.1 | 150 / 155 (96.77%) | 5 | 0 |
| 3 | `maps_GT_test_new_3/img_3.png` | yes | 74 | 319.7 | 195 / 202 (96.53%) | 4 | 3 |
| 4 | `maps_GT_test_new_3/img_4.png` | yes | 48 | 217.3 | 131 / 135 (97.04%) | 4 | 0 |
| **all** |  | **5/5** |  | **1495.5** | **855 / 882 (96.94%)** | **24** | **3** |

Per-agent coverage over the five episodes:

| Agent | Actual moves | Network | Rapid | Back | Network share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 323 | 313 | 7 | 3 | 96.90% |
| 1 | 285 | 277 | 8 | 0 | 97.19% |
| 2 | 274 | 265 | 9 | 0 | 96.72% |

The 27 bypass moves are exactly the 24 rapid-convergence moves plus the 3
back-path moves; no `other_bypass` occurred. Every network decision produced
one activation of shape `[1, 1, 128]`: 855/855 present, all finite, zero
`argmax(logp)`/action mismatches, zero returned-waypoint/next-node mismatches.
Of the 855 decisions, 823 were in the en-route phase and 32 in the
reached-target phase (maps 1-4); 852 changed the option at that step.

### Complete successful trajectory

Map 0 is the reference trajectory: 60 loop steps, 171 executed moves, three
agents reaching `(148,120)`, `(0,124)` and `(120,128)` respectively, with
travel distances 446.7, 437.3 and 487.6 in map units. The full per-move record
is `runs/map000_record/trajectory.csv` and
`runs/map000_record/episode_000_record.json` on the experiment host; each row
carries step, agent, decision kind, decision id and from/to positions.

### Which stages lack activations

- **Rapid convergence** (24 moves): the agent pops a previously computed A*
  path (`test_worker.py:68`); `PolicyNet` is not called, so no activation
  exists for the executed waypoint.
- **Back path** (3 moves, agent 0 on map 3): after reaching the target with
  `back_trick` set, the agent pops `back_path` (`test_worker.py:87`); no
  network call.
- **Idle agents**: agents with `explore_over = 1` are skipped and are not
  decisions.
- **Post-policy coordination**: `solve_path_conflict` can change the executed
  waypoint after the network call; the activation describes the network input,
  not the conflict-resolved output. `update_graph`/`update_planning_state`
  happen between decisions and are not represented in a per-decision
  activation either.

Thus the recorded activation covers every network decision but cannot cover
the deterministic A*/back-path stages, and it is a single current-state
embedding, not a history of teammate observations.

### Hook on/off comparison

Same map, same seed (0), same greedy evaluation; the off run installs no
PolicyNet hooks and does not wrap `select_next_waypoint`.

| Map | Success on/off | Steps on/off | Metrics | Per-agent trajectory | Wall on/off (s) |
| ---: | :---: | :---: | :---: | :---: | ---: |
| 0 | true / true | 60 / 60 | identical | identical | 283.0 / 277.0 |
| 1 | true / true | 103 / 103 | identical | identical | 403.4 / 392.9 |

`metrics_equal` is true for both pairs (all distances, explored rate and
success), and every agent's ordered sequence of executed positions is equal.
Recording adds about 2-3% wall time and does not change the policy or task
result.

## Where teammate experience could enter a later decision (static, no intervention)

These are read-only observations from the pinned source. Nothing here is
implemented.

| Position | Role | How teammate experience could affect the next decision |
| --- | --- | --- |
| `agent.py:222` `update_planning_state(robot_locations, finding_target, reach_target, explore_over)` | Called at episode start and after every step from `test_worker.py` | Teammate positions/target-found/reach/explore flags already enter here; changing their representation changes the next observation and option state. |
| `agent.py:210` `update_graph(...)` | Rebuilds the agent's belief graph from the shared map | Teammate sensor updates enter through `Env.update_robot_belief`; they alter node features consumed by the policy. |
| `agent.py:492` `get_observation` | Builds the 9-dim node features, edges and masks passed to `PolicyNet.forward` | A teammate-experience feature would have to be added to `node_inputs`/`navi_node_inputs` or to a new embedding before the policy call. |
| `agent.py:623` `select_next_waypoint` | Sole `PolicyNet` call site (`agent.py:626`) | A sidecar decision-time intervention would sit between this call and the action execution in `test_worker.py:80/92`. |
| `model.py:223,273` `current_embedding` | Concatenates enhanced and raw current-node features (256 -> 128) | This is the recorded activation; its input contract is fixed at 256, so extra experience features need a new projection rather than appending here. |
| `model.py:293-298` termination | Option switch uses only the current node feature (plus option embedding) | A teammate signal could bias termination/switch before `current_option` is chosen; the action head would then see the switched option. |
| `test_worker.py:107` `solve_path_conflict` | Post-policy coordination of selected waypoints | Teammate experience could change the executed waypoint without another network call; no activation corresponds to this stage. |
| `test_worker.py:103-105` `back_trick` / `back_path` | Team target-found coordination and dead-space return | These are bypass stages with no `current_embedding` activation. |

The 168 network decisions in map 0 each produced one activation; the 3
rapid-convergence moves (one per agent) produced none because the network was
not called. `back_path` did not occur in map 0. This is the structural reason
coverage is below 100%: the policy is bypassed by deterministic A* shortcuts,
not by missing records.
