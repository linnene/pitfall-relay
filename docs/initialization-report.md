# Initialization report

Date: 2026-10-30. Scope authorized by the user: initial engineering setup for
Conda + PyTorch + BenchMARL/TorchRL + recurrent MAPPO, plus package tests,
minimal CI, and documentation. No task, environment package, method component,
training, rollout, dataset or model download (ordinary pip package installation is
separate), `wyb` connection, or Git commit was involved.

Separately owned files were left untouched: `README.md` and `docs/assets/` (Finn),
`docs/research-brief.md` (user). Literature files were not modified.

## Result

A local conda environment `pitfall-relay` exists and is reproducible from
`requirements/stack.txt`; the pinned stack imports; BenchMARL's MAPPO and GRU
configurations construct; the test suite passes in the environment and skips
cleanly without it.

## Verified environment

| Item | Value |
| --- | --- |
| Platform | macOS 26.6.1 (Darwin 25.6.0), arm64 |
| Python | 3.11.18 (conda, `pitfall-relay`) |
| conda / pip | 26.9.4 / 26.2.2 |
| torch | 2.10.0 (MPS available, no CUDA) |
| torchvision | 0.25.0 |
| tensordict | 0.11.0 |
| torchrl | 0.11.1 |
| benchmarl | 1.5.2 |
| av | 13.1.0 |
| hydra-core | 1.3.2 |
| numpy | 2.4.6 |
| pytest | 8.3.5 |

Sources: default PyPI index only, HTTPS, no mirror or TLS workaround. Every
package except `benchmarl` installed from a prebuilt wheel; `benchmarl` publishes
only an sdist on PyPI, so pip built a `py3-none-any` wheel locally (no
compilation of native code). `av` resolved to the cp311 macOS arm64 wheel, so the
ffmpeg/header risk noted in `docs/base-stack-candidates.md` did not materialize
on this platform.

Installed wheel tags: `torch cp311-none-macosx_11_0_arm64`,
`torchvision cp311-cp311-macosx_11_0_arm64`, `tensordict cp311-cp311-macosx_14_0_arm64`,
`torchrl cp311-cp311-macosx_11_0_arm64`, `av cp311-cp311-macosx_11_0_arm64`,
`benchmarl py3-none-any` (built from sdist). Environment size on disk: 778 MB.

## How the pins were chosen

- BenchMARL 1.5.2 declares `torchrl>=0.10,<0.12`, `torchvision`, `av<14`,
  `numpy`, `hydra-core` (verified against the published metadata).
- torchrl 0.11.1 requires `tensordict>=0.11.0,<0.12.0`; 0.11.0 is the only
  release in that range. This is the actual binding constraint.
- torchrl 0.11.1 was published 2026-02-05; torch 2.10.0 (2026-01-21) is the
  release contemporary with it and the one verified here. Newer torch releases
  (up to 2.15.0) exist but were not tested. BenchMARL constrains torchrl, not
  torch, so this pairing is a verified choice rather than a metadata-forced
  exclusion.
- torchvision 0.25.0 declares `torch==2.10.0`; given that torchvision, the
  torch/torchvision pair is forced.
- Python 3.11 satisfies every pin (`torchrl` needs >=3.10, torch wheels exist for
  cp311); no minor-version adjustment was needed.

## Verification run (actual output)

```text
$ conda run -n pitfall-relay python -m pytest tests -q
..........                                                               [100%]
10 passed, 4 warnings in 27.81s        # warnings: torchrl WeightUpdater deprecations

$ PYTHONPATH=src python -m unittest discover -s tests   # in the env
Ran 10 tests in 0.185s
OK

$ PYTHONPATH=src /Users/a0000/miniforge3/bin/python -m unittest discover -s tests   # no torch
Ran 10 tests in 0.000s
OK (skipped=9)
```

The forward-pass check in `tests/test_stack.py` builds a `Gru` policy
(`hidden_size=16`) on random tensors and asserts output shape
`[2, 4, 3, 2]` and determinism across two calls; it is a shape/state check, not
a learning result.

## Files added or changed

| Path | Change |
| --- | --- |
| `requirements/stack.txt` | New. Human-readable pins. |
| `requirements/stack-lock.txt` | New. `pip freeze` of the verified env (platform-specific). |
| `scripts/create_env.sh` | New. Creates the isolated env and installs the pins. |
| `tests/test_stack.py` | New. Import, version, MAPPO/GRU/MLP config, and Hydra config checks; skip when the stack is absent. |
| `pyproject.toml` | Added `dev` extra and pytest configuration. Package runtime dependencies remain empty. |
| `.github/workflows/ci.yml` | New. Scaffold tests on push/PR; stack smoke job on manual dispatch. |
| `docs/development.md` | New. Commands, pins, verified/unverified status. |
| `docs/initialization-report.md` | This file. |

Nothing was committed or pushed. `README.md` already links to
`docs/development.md` and this report.

## Deviations and caveats

- `requirements/stack-lock.txt` omits conda's `packaging` distribution: pip
  freeze records it with a `file:///home/conda/...` URL that cannot be
  reinstalled elsewhere. Regenerate the lock in the target environment instead.
- The lock is macOS arm64 specific. The Linux environment on `wyb` is untested
  and must be created there independently after authorization.
- The `stack-smoke` CI job installs roughly 1 GB of Linux torch wheels and has
  not been executed.
- `tests/test_stack.py` pins exact versions. A deliberate upgrade requires
  updating the pins in one place and re-running the suite.
- The stack tests treat any `ImportError` as "stack absent", so a broken
  installation could skip instead of fail. On the next change to that file,
  narrow the guard to the specific optional top-level modules. This run executed
  all 10 tests, so the gap did not affect the verification above.

## Remaining before any experiment

1. Task selection, failure definition, supervision source, within-episode vs
   cross-episode protocol, and execution-time parameter policy — all still the
   user's decisions.
2. Task environment package installation (vmas / pettingzoo / meltingpot) only
   after the task is chosen.
3. A minimal BenchMARL experiment smoke run (sample → update → eval →
   checkpoint) to validate the training loop, not just configuration objects.
4. Compute budget, step units, seed policy, and evaluation protocol recorded
   before the first run, per `AGENTS.md`.

None of the above is started.
