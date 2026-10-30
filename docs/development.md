# Development environment

Status of every command on this page is stated explicitly. "Verified" means it
was run on the machine and platform named below; it does not mean the code
learns anything.

- Platform verified: macOS 26.6.1 (Darwin 25.6.0), arm64, CPython 3.11.18,
  conda 26.9.4, pip 26.2.2.
- Not verified: Linux, `wyb`, CUDA, any training or rollout.
- What exists today: an isolated conda environment, pinned dependencies, and
  import/configuration tests. No task, environment package, or algorithm is
  implemented.

## Layout

```text
requirements/stack.txt       human-readable pins (edit this)
requirements/stack-lock.txt  exact installed set for this platform (generated)
scripts/create_env.sh        create env + install pins
tests/test_stack.py          import and configuration checks
.github/workflows/ci.yml     scaffold tests; stack smoke job on manual dispatch
```

## Create the environment

Verified (creates `pitfall-relay`, Python 3.11, installs the pins, installs the
package in editable mode):

```bash
scripts/create_env.sh
```

To delete and rebuild an existing environment:

```bash
REINSTALL=1 scripts/create_env.sh
```

Manual equivalent, if you prefer explicit commands:

```bash
conda create -y -n pitfall-relay python=3.11 pip
conda run -n pitfall-relay python -m pip install -r requirements/stack.txt
conda run -n pitfall-relay python -m pip install -e .
```

The environment is isolated. Do not install into `base` or reuse another
project's environment.

## Pins and why they are not "latest"

| Package | Pin | Reason |
| --- | --- | --- |
| torch | 2.10.0 | Released 2026-01-21, contemporary with torchrl 0.11.1 (2026-02-05); torchvision 0.25.0 requires exactly this version. Newer torch releases exist but were not tested. BenchMARL constrains torchrl, not torch, so this pairing is a verified choice rather than something the dependency metadata forces. |
| torchvision | 0.25.0 | Declares `torch==2.10.0`; any other pair fails resolution or breaks at import. |
| tensordict | 0.11.0 | torchrl 0.11.1 requires `>=0.11.0,<0.12.0`; 0.11.0 is the only release in that range. |
| torchrl | 0.11.1 | BenchMARL 1.5.2 requires `>=0.10,<0.12`. |
| benchmarl | 1.5.2 | Current release; PyPI ships an sdist that builds a `py3-none-any` wheel. |
| av | 13.1.0 | BenchMARL lists `av<14` unconditionally. 13.1.0 has a cp311 macOS arm64 wheel, so no source build and no ffmpeg headers are needed. |
| hydra-core | 1.3.2 | BenchMARL configuration layer. |
| numpy | 2.4.6 (locked) | Resolved transitively; `stack.txt` only bounds `<3`. |

`requirements/stack-lock.txt` is a `pip freeze` of the verified environment. It
is platform-specific and excludes conda's own `packaging` build (it carries a
`file://` URL that pip cannot install); regenerate it instead of editing it.

## Run the tests

Verified in the `pitfall-relay` environment (10 passed):

```bash
conda run -n pitfall-relay python -m pytest tests -q
conda run -n pitfall-relay python -m unittest discover -s tests -v
```

Verified without the stack (conda base interpreter, Python 3.14, no torch): the
same unittest command reports 1 passed, 9 skipped. The stack tests skip
themselves when `benchmarl` is not importable, so the package check stays usable
in a bare checkout.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## What the stack tests check

`tests/test_stack.py` verifies, on random tensors and default configs only:

- the installed versions equal the pins, and `av` imports;
- `torchrl` and `tensordict` are in the compatible version range;
- `MappoConfig.get_from_yaml()` builds, exposes the expected defaults and flags,
  and survives `dataclasses.replace`;
- a `GruConfig` policy builds under BenchMARL's spec conventions and returns
  logits of shape `[batch, seq, agents, actions]`; the same input twice gives
  the same output, i.e. no hidden state leaks between calls when `h_0` is unset;
- an `MlpConfig` critic builds and returns `[batch, 1]` without NaN;
- BenchMARL's `conf/experiment/base_experiment.yaml` loads through OmegaConf.

These are engineering checks. A model that runs a forward pass is not evidence
of learning, and no result here says anything about experience sharing.

## Framework conventions worth knowing before writing code

- BenchMARL builds specs per agent group, e.g.
  `Composite({"agents": Composite({"observation": ...}, shape=(n_agents,))})`.
  The agent dimension lives inside the group spec, not in the batch.
- Recurrent models receive a root `is_init` key of shape `[batch, seq, 1]`.
  With recurrent models the batch layout is `[batch, seq, agents, features]`;
  `h_0` is read from the model's hidden-state key and is `None` during training.
- `MappoConfig` defaults (`share_param_critic=True`, `lmbda=0.9`,
  `clip_epsilon=0.2`) are BenchMARL defaults, not project decisions.

## Not installed, not verified

- Task environment packages: `vmas`, `pettingzoo`, `dm-meltingpot`, and the
  `gnn`/`logging` extras are deliberately absent. The task is not selected.
- No BenchMARL experiment has been run: that requires a task, which requires a
  decision the user has not made.
- No performance, memory, throughput, or latency measurement exists. Any budget
  statement needs a run on the experiment host.
- macOS MPS is available to torch, but nothing has been run on it.
- CI installs the pins on Linux only in the manually dispatched `stack-smoke`
  job; that job has not been executed.
