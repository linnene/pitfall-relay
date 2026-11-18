# Warehouse-A reproduction and SAFE pre-study

Question: can the official Warehouse-A MacIAICC segment be reproduced on
`amgg`, and does the local Actor's GRU latent provide a failure signal that
arrives before the next relevant teammate decision?

## Locked protocol

- Upstream `yuchen-x/MacroMARL` at
  `f5899cd7b96bb3dbf7b265494a3ba8a15b2e3fe2`; source tarball SHA-256
  `9d1e71a220d293202f23d8528a8555d64f754ef6e8c3885b61e9c98b64873d11`.
- Official first Warehouse-A block: `MacIAICC`, `OSD-D-v7`, 3 agents,
  4 environments, 200 steps, 40,000 episodes, original hyperparameters.
- Training seeds 11/22/33 are explicit; one seed runs at a time. The official
  script does not pass `--seed` and therefore uses the default/unseeded path;
  this difference is recorded in every run manifest.
- Evaluation is a separate process with its own random stream
  (`eval_seed_base=7100`), no epsilon exploration (`test_mode=True`), no online
  update, and no centralized critic in the execution loop.
- Success is `done` from the environment: both humans finish their tasks;
  delivery/completion time is the number of environment steps. Undiscounted
  episode return is the original shared reward sum; the paper plots mean
  discounted return, so its curve is not numerically comparable without the
  paper's 20-trial/100-episode protocol.

## Checkpoint and manifest contract

Every `checkpoints/epi_XXXXXXXX/manifest.json` binds:

- upstream commit, FLEX execution SHA, config SHA-256, seed/run ID;
- episode count, parent checkpoint, wall time, recent train return, eval return;
- each checkpoint file name, byte size and SHA-256.

Files include the upstream generic state (episode count, Python/NumPy/Torch
RNG, returns, eval returns), per-env RNG states, and per-agent actor, critic,
target nets and optimizers. The adapter also saves `actor_tgt_net`, which the
upstream `mac_iaicc` saver omits. Replay buffer contents and per-env GRU
hidden states are not in the upstream checkpoint and are not claimed as
bit-exact resume state.

## Evaluation and events

`eval_policy.py` records per episode:

- local observation, selected action, actual macro-action, `mac_done`,
  `valid`, rewards (stored but never used as detector input);
- real Actor forward time, input hidden state, output hidden state and logits;
- Turtlebot action/timer/basket, Fetch serving state, ready objects, human
  states;
- confirmed `handoff_failure`, `handoff_success`, `tool_wait_timeout`, plus
  `handoff_attempt_start`, `tool_wait_start`, `serving_failed_start`. Each
  event records the next decision of the event agent and the next relevant
  teammate decision.

Event timing follows the upstream macros, not the step on which the next
macro starts:

- `handoff_attempt_start` is the step where Fetch selects `Pass_Obj_T0/T1`.
- `handoff_failure`/`handoff_success` is the step where that pass completes
  and the basket transfer is or is not observed. Completion is detected from
  the pre-step timer (`0 < time_left <= 1`), not from the following
  re-selection step.
- `tool_wait_start` is the first step whose `Get_Tool` timer becomes negative
  after the turtle reaches the table.
- `tool_wait_timeout` is the `Get_Tool` done transition; v7 resets the timer
  to `-1` on timeout, so the post-step timer cannot be used as the marker.

On the early 200-episode calibration actor, a two-episode smoke produced 54
handoff attempts, 44 failures, 9 successes, 25 tool waits and 14 timeouts.
This is a plumbing check on a non-final policy, not an algorithm result.
The initial event logic double-counted pass completions and missed timeouts;
the corrected counts above are the first valid event semantics.

Truth labels are stored in the episode JSON; detector features in
`probe_latent.py` read only local observations and selected/actual action
history, optionally the real latent. Rewards and env internals are not
features.

Initial and final Actors are evaluated on identical episode seeds. Hook-on and
hook-off runs use the same checkpoint and seeds; action sequences, steps,
returns and success are compared exactly.

## Probe alignment

A probe sample at decision step `t` uses completed observations/actions through
`t-1` plus the Actor forward hidden state at `t`. This keeps the local-history
baseline and the latent arm on the same information; using `obs[t]` would give
the baseline one step more than the latent. The shuffled-latent control uses a
fixed cross-episode donor mapping so thresholds and paired bootstrap scores are
frozen. Splits are complete episodes with separate eval seeds. The calibration
threshold is the 95th percentile of normal calibration episode maxima; the
test false-alarm rate and event recall are reported without refitting.

## Run

```bash
D=/data/amgg_flex_demo/flex-warehouse-safe-20261118-95e0c41a
cd "$D/FLEX/experiments/warehouse_safe"
# formal training (one seed at a time)
bash run_seeds.sh
# fixed-actor evaluation, for example
PYTHONPATH="$D/MacroMARL/src" "$D/envs/macro/bin/python" eval_policy.py \
  --upstream-root "$D/MacroMARL" \
  --config configs/warehouse_a_maciaicc.json \
  --checkpoint-dir "$D/runs/seed_11/checkpoints/epi_040000" \
  --label seed11_final --seed 11 --eval-seed 7111 \
  --episodes 200 --out-dir "$D/runs/eval/seed11_final" --hooks on \
  --flex-sha "$FLEX_SHA"
```

## Data

All raw logs, metrics, checkpoints, evaluation arrays/events and probe outputs
remain under `/data/amgg_flex_demo/flex-warehouse-safe-20261118-95e0c41a/`.
Failed calibration and resume tests are preserved there as evidence.
