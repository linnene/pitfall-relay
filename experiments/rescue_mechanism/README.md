# Rescue mechanism check

Task: `flex-rescue-mechanism-1f83ba62`
Date: 2026-11-20
Host: `amgg` only, single-CPU Python. No Actor training, SAFE/FLEX, Isaac, or
new environment search.

## Model

Two casualties A/B, one shared ambulance. Clearing takes time; transport to
the refuge takes time. Two high-level plans:

- `serial`: the ambulance serves B first while A is cleared, then serves A.
- `parallel`: the ambulance stages at A during A's clearing (real preparation
  occupying the shared ambulance), then serves A faster, then B.

HP decays with time; a casualty survives only if it reaches the refuge with
HP > 0. No plan/win is assigned by category, message, seed or arm. The stage
is real work during clearing and delays B; it is not waiting.

## Development precheck (12 scenarios)

Six `parallel_better` and six `serial_better` cases, varying local casualty
disturbances under the same facility/exogenous process.

- `parallel_better`: serial delivers 1/2, parallel 2/2 (6/6 reversal).
- `serial_better`: serial delivers 2/2, parallel 1/2 (6/6 reversal).

The order reversal is real: serial misses A's rescue window when A is urgent;
parallel occupies the ambulance and misses B when B is urgent.

## Information check

B has its own case and can use normal telemetry: current HP, trends, clear
times, transport times and the shared-resource ETA. A strong local rule
simulates both plans from these values. A's observed attempt is serial, and
its confirmed outcome identifies the better plan. In all 12 development
cases the strong local rule and the experience-derived choice agree
(`normal_choice == experience_choice`), and B's team delivery is 2/2 in both.
There is no decision the A experience adds beyond normal telemetry/trend.

Therefore the mechanism-order reversal is supported, but the **information
increment precheck fails**: ordinary state/trend already recovers the same
choice. Stop before the formal 24+24 comparison; no holdout, no training.

## Limits

The task is a minimal abstract rescue-ordering prototype, not RCRS or a real
rescue system. The reversal supports a mechanism existence claim only; the
failure of the information precheck means this experience design is not a
FLEX candidate.

## Run

```
python3 rescue_mechanism.py --out results/dev.json
python3 -m unittest -v test_rescue_mechanism.py
```
