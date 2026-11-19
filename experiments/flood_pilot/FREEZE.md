# Freeze

Task `flex-flood-pilot-b71e8a42`, frozen 2026-11-19 before the formal holdout.

- Gate 1 development seeds: `high_a` 1–8, `urgent_b` 1–8 (16 total).
- Gate 2 holdout seeds: relevant 2001–2024, nonapplicable 3001–3024 (48
  independent groups; labels used only for evaluation).
- Scenario parameters and policy constants are exactly those in
  `flood_pilot.py` at the code commit containing this file. No holdout result
  was inspected before this freeze.
- Runtime: Python 3.12.3 on `amgg`, single CPU process, no external packages.
- No training checkpoint, no Isaac/SAFE/FLEX, no system changes.
- The execution SHA, script sha256 and command line are recorded in
  `results/anchor.json` on `amgg` at run time.

The full gate-2 holdout is run once after this commit. Gate 1 results may be
re-run for reproducibility; the holdout is not rerun or extended.
