# ORION public-checkpoint baseline and activation probe

Read-only evaluation of the public ORION cooperative exploration policy
(`marmotlab/ORION`) with three agents, on CPU, without FLEX. The adapter
imports upstream code unchanged, loads the published `model/checkpoint.pth`
through an explicit path with a strict state-dict check, records the
`PolicyNet.current_embedding` activation at every network decision, and
classifies decisions that bypass the network (rapid convergence / back path).

Upstream: <https://github.com/marmotlab/ORION>, pinned commit
`a935c4bab6aed90e36a24d596bfb9a74a8b234e9`.
License: MIT, Copyright (c) 2025 MARMot Lab @ NUS-ME. Upstream source and
weights are not vendored in this repository; they are downloaded on the
experiment host.

## Why the checkpoint path is overridden

Upstream `test_driver.py` asks for `model/<FOLDER_NAME>/checkpoint_50000.pth`
(`FOLDER_NAME = "09_30_0.051"`), which does not exist in the public release.
The public weight is `model/checkpoint.pth`. `orion_adapter.load_policy`
therefore takes the checkpoint path explicitly, checks for the `policy_model`
entry, and calls `load_state_dict(..., strict=True)`. A mismatch is reported
and never patched, renamed, substituted or trained around.

## Environment

Python 3.12.3 venv; CPU-only. NumPy must stay below 2.0 because upstream
`sensor.py` calls `ndarray.itemset`, removed in NumPy 2:

```bash
python3 -m venv env
. env/bin/activate
pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.26.4 scipy==1.11.4 scikit-learn==1.3.2 \
    scikit-image==0.22.0 matplotlib==3.8.4 contourpy==1.2.1 \
    tifffile==2024.8.30 imageio==2.34.0
```

Ray is not installed: the adapter constructs `TestWorker` directly instead of
using upstream's 30-process Ray driver, preserving the same evaluation code
path with one process and one CPU thread.

## Run

```bash
ORION_ROOT=/path/to/ORION
CKPT=$ORION_ROOT/model/checkpoint.pth
OUT=/path/to/runs/map000_record

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python run_episode.py --orion-root "$ORION_ROOT" --checkpoint "$CKPT" \
    --episode-index 0 --seed 0 --mode record --out-dir "$OUT" --torch-threads 1

# same map, no PolicyNet hooks and no decision wrappers
python run_episode.py ... --mode nohook --out-dir .../map000_nohook

python analyze.py --out-dir .../runs --write
```

`episode_index` selects the map as upstream does:
`sorted(os.listdir("maps_priori_test_new_3"))[episode_index % 50]`, so indices
0-4 are five distinct fixed maps. `greedy=True` and seeds for `random`,
`numpy`, `torch` are fixed.

## Outputs

- `episode_<idx>_<mode>.json`: contract, metrics, per-agent coverage, executed
  moves, timing, warnings.
- `episode_<idx>_record_decisions.jsonl`: one row per network decision with
  observation shapes/indices, action index and node, option/termination,
  logits and alignment checks.
- `episode_<idx>_record_activations.npz`: `decision_id` and the flattened
  `current_embedding` activation (`float32`) exactly as observed. Missing hook
  output is recorded as `activation_present=false`; nothing is imputed.
- `analysis.json`: aggregate coverage, activation audit and hook on/off
  comparison.

## Decision taxonomy

`Env.test_step` is reached once per executed move. A move is `network` when a
`select_next_waypoint` call happened for that agent in the same loop step;
otherwise it is classified from upstream flags:

- `rapid_convergence`: `not reach_target and robot.rapid_convergence_flag`;
  the move pops `robot.path` (A* result) and never calls `PolicyNet`.
- `back_path`: `reach_target and robot.back_trick`; pops `robot.back_path`.
- `other_bypass`: fallback, expected to be empty.

Agents already `explore_over` are idle and are not counted as decisions.
