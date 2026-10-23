# Working rules

- Research status: early exploration. Do not report scaffold tests as algorithm results.
- Development branch: `dev`. Preserve unrelated changes; do not rewrite published history.
- A world model, an LLM, and natural-language messages are not requirements.
- Do not choose the application task or lock the algorithm architecture without discussion.
- Start no training or downloads without an agreed budget. Compute belongs on `ssh wyb`, not the local Mac.
- Before a run, record one question, discriminating predictions, stopping conditions, code SHA, seed, full configuration, step units, and evaluation protocol.
- Compare on equal information access and budgets. Track team task success, repeated failures, communication cost, and negative transfer.
- Keep credentials, data, checkpoints, and run outputs out of Git. Do not install dependencies into another project’s environment.
- Use `apply_patch` for source edits. Run `PYTHONPATH=src python3 -m unittest discover -s tests -v` for scaffold changes.
