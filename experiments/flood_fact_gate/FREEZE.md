# Freeze

Task `flex-flood-fact-gate-4d926eb3`, frozen 2026-11-19 before the formal
holdout.

- Development: 16 groups, 8 relevant + 8 non-applicable; no holdout seeds
  inspected before this freeze.
- Holdout seeds: relevant 9001–9024, non-applicable 9101–9124, no-timely
  9201–9224 (24 each, 72 total).
- The same `decide` function and action rule are used for all arms. Baseline
  receives only normal telemetry; the fact arm receives the local execution
  fact object; the permuted arm receives a fixed cross-scenario fact object.
  No-timely arms receive no fact and must be step-equivalent.
- The execution SHA, script hash, Python version and command are recorded in
  `results/anchor.json` on `amgg` at run time.
- No training checkpoint; no Isaac/SAFE/FLEX; single CPU on `amgg`.
- The holdout is run once. No outcome-based exclusion, re-draw or threshold
  change is allowed after this commit.
