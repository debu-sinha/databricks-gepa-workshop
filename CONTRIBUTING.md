# Contributing

Thanks for considering a contribution. This repo is a workshop template, so contributions that improve clarity, fix bugs, or extend the workshop's reach are welcome.

## How to contribute

1. Fork the repo
2. Make your change on a feature branch
3. Run the validation smoke test before submitting:
   ```bash
   python scripts/validate.py
   ```
4. Open a PR with a clear description of what changed and why

## What's in scope

- Bug fixes (the deterministic mini-optimizer should be reproducible)
- Documentation improvements (clearer prerequisites, better workshop facilitation notes)
- New synthetic test cases that broaden coverage
- Knob additions or tweaks that match real DSPy / MLflow GEPA behavior more closely
- Notebook polish for Databricks workspace flow

## What's out of scope

- Real-customer data or proprietary prompts (this is an open template — keep it generic)
- Heavy framework changes that make the lab harder to teach (the priority is clarity over features)
- Optimizer changes that make `mini_gepa` non-deterministic (it's a teaching scaffold; reproducibility matters)

## Style

- Keep code Python-idiomatic and well-typed where it doesn't hurt readability
- Notebook cells should be short and self-explanatory — workshop attendees read them under time pressure
- Keep comments tight; avoid restating what the code does

## Commit messages

Plain English, imperative mood, under 72 chars on the first line. No conventional-commit prefixes.

Good: "Add multi-prompt mode example to MLflow notebook"
Bad: "feat: added multi-prompt mode"
