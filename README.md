# PitfallRelay

**Learned Failure Experience Sharing for Multi-Agent Collaboration**

Can one agent’s failure during execution help teammates avoid repeating similar mistakes?

PitfallRelay explores learning a latent representation of an unsuccessful interaction and communicating it to teammates, so that their policies can respond to similar—but not identical—situations. The intended experience representation is neural and learned, not a hand-written rule or a natural-language summary.

## Status

Early-stage research scaffold. No learning algorithm, environment, benchmark result, or robot deployment has been implemented here yet. The architecture and application task remain open.

## Research scope

- **Encode experience:** retain useful information from observations, actions, and negative outcomes.
- **Learn when to share:** identify experiences that can help another agent, rather than broadcasting everything.
- **Use experience at execution time:** condition a recipient’s decisions on a learned message without requiring an optimizer update at deployment.
- **Generalize:** reduce repeated failures beyond the sender’s exact trajectory or object instance.

A world model is **not required**. Model-free and model-based learners are possible hosts, not architectural commitments. These are functional questions, not claims of separate contributions or novelty.

## First milestone

Choose a task where a hidden, recurring action-dependent risk can affect several agents, and where observation-only warnings are insufficient. Write a falsifiable experiment brief before implementing the method. See [research brief](docs/research-brief.md).

## Development

Python 3.11+, with no runtime dependencies in this scaffold. Work on `dev`; `main` holds reviewed milestones. Local editing, tests, and Git; training compute on `ssh wyb`, only after an explicit experiment budget is agreed.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

This command checks package import only, not learning performance. No dataset or model download is needed.

## Layout

```text
src/pitfall_relay/   Python package scaffold
tests/              Lightweight scaffolding checks
docs/               Research framing and experiment decisions
```

No paper or novelty claim is made. Related-work review and task selection are pending.
