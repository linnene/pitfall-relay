# Research brief

## Confirmed intent

An agent executes an operation and receives a bad outcome. A learned mechanism encodes the experience and shares it with teammates. Recipients use it to reduce repeated mistakes in related situations during execution. Representation and sharing triggers may be learned with neural networks; a hand-written warning or language summary is not the intended method.

## Open choices

- Application task and failure definition.
- Experience window, encoder, message budget, recipient conditioning, and trigger mechanism.
- Host policy learner and whether any world model is useful.
- Novelty relative to prior experience-sharing, communication, and memory methods.

## Candidate hypothesis—not a result

At matched information and communication budgets, messages learned from a sender’s failed interaction can reduce recipients’ repeated failures on held-out, related situations beyond current-observation messages or uninformative messages.

## Required comparisons to discuss before running

- No experience sharing.
- Current-observation or state-only communication.
- Same budget with shuffled, stale, or mismatched failure messages.
- Shared experience, evaluated on novel conditions, with team success and negative transfer measured—not just sender reward.

Use multiple training seeds for claims about learning. Do not treat episodes from one trained policy as independent training replications.

## Decision gate

If a simple location/status warning solves the selected task, or improvement disappears with matched information access and cost, reconsider the task or mechanism rather than claiming learned failure transfer.

Do not require every component to be novel. Select one technical contribution only after task and nearest-neighbor review.
