# Warehouse-A reproduction and SAFE pre-study

Task: `flex-warehouse-safe-20261118-95e0c41a`.

This directory runs the official Warehouse-A MacIAICC first segment and adds
immutable launch/checkpoint records. Upstream code is imported unchanged from
the pinned commit; no reward, macro-action rate, task, or algorithm change is
made.

## Upstream

- Repository: <https://github.com/yuchen-x/MacroMARL>
- Commit: `f5899cd7b96bb3dbf7b265494a3ba8a15b2e3fe2`
- Source tarball SHA-256 (fetched on amgg):
  `9d1e71a220d293202f23d8528a8555d64f754ef6e8c3885b61e9c98b64873d11`
- Official script: `experiments/Warehouse/mac_iaicc.sh`, first block
  (Warehouse-A, `MacIAICC`, `OSD-D-v7`).
- License: no repository-level `LICENSE`; upstream `setup.py` declares MIT.
  This is recorded as an upstream licensing uncertainty.

## Locked run

`configs/warehouse_a_maciaicc.json` is the single source of truth. It keeps the
official first-block hyperparameters: 3 agents, 4 training environments,
200 steps, 40,000 episodes, `l_mode=0`, `train_freq=4`,
`c_target_update_freq=32`, `n_step_TD=5`, `gamma=1.0`,
`a_rnn_layer_size=32`, `c_rnn_layer_size=64`, `sample_epi`,
`eval_policy`, and the same reward/penalty settings.

Deliberate differences from the official launcher, recorded for every run:

1. Training seeds are explicitly `11/22/33` (official `mac_iaicc.sh` does not
   pass `--seed`, so it uses the unseeded/default path). `run_id` equals the
   seed.
2. Evaluation uses a separate fixed random stream (`eval_seed_base=7100` plus
   seed), not the training stream.
3. The runner implements the same loop as `MacIAICC.learn` but adds milestone
   checkpoints and immutable manifests; upstream only saves at the end.
4. `torch.set_num_threads(1)` and CPU device are fixed for reproducibility
   and to avoid oversubscription; the official command leaves device/threads
   at defaults.
5. The adapter imports `MacIAICC` directly and does not import
   `gym_macro_overcooked` (unused for OSD). This avoids an unavailable
   optional package and does not change the environment or algorithm.

## Checkpoints

Each seed writes:

```
seed_<seed>/
  run_manifest.json
  run_events.jsonl
  train_metrics.jsonl
  checkpoint_index.jsonl
  checkpoints/epi_000000/ ... epi_040000/
  work/performance/warehouse_A_maciaicc_s<seed>/
  work/policy_nns/warehouse_A_maciaicc_s<seed>/
```

Every milestone contains the upstream generic checkpoint (episode count,
Python/NumPy/Torch RNG states, returns, eval returns), per-environment RNG
states, and per-agent actor/critic/target/optimizer state. The upstream
`mac_iaicc` saver omits `actor_tgt_net`; the adapter adds
`<run_id>_agent_<i>_extra.tar` with that state and restores it on resume.

Known resume gaps, kept explicit: upstream does not save the replay buffer or
the per-environment GRU hidden state; on resume the environment runners reset
and the buffer is rebuilt, matching upstream's own resume path. No checkpoint
is presented as bit-exact mid-episode resume.

## Run

On `amgg` only:

```bash
D=/data/amgg_flex_demo/flex-warehouse-safe-20261118-95e0c41a
PYTHONPATH=$D/MacroMARL/src \
$D/envs/macro/bin/python run_train.py \
  --upstream-root $D/MacroMARL \
  --config configs/warehouse_a_maciaicc.json \
  --seed 11 --out-root $D/runs \
  --flex-sha "$FLEX_SHA" --flex-dirty "$FLEX_DIRTY"
```

Short calibration (never counted as formal reproduction):

```bash
... --seed 11 --out-root $D/runs/calibration --max-epi 200
```

## Data retention

Raw stdout, manifests, checkpoints, evaluation trajectories, event labels,
latent arrays, probe outputs, failed attempts, and logs all stay under the
task directory on `amgg`. Nothing is downloaded to the Mac; only the final
brief and remote paths are kept locally.
