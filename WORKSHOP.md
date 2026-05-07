# Workshop facilitation guide

A 90-minute workshop that walks attendees through the end-to-end GEPA-style optimization flow on Databricks. Designed for technical practitioners (data scientists, ML engineers, platform engineers) who already understand prompt engineering and want to learn systematic prompt + program optimization.

## Outcomes

By the end of the workshop, attendees should be able to:

- Explain GEPA-style optimization and how it differs from manual prompt tuning, RAG-only approaches, and full RL post-training
- Read a quality / cost / latency scorecard and identify the Pareto frontier
- Know when to reach for DSPy GEPA vs MLflow Prompt Optimization (beta)
- Adapt the synthetic test setup to their own enterprise search / retrieval use case
- Understand which knobs matter most: candidate-selection strategy, optimization scope, max metric calls, reflection minibatch size

## Format

| Time | Section | Format |
|---|---|---|
| 0:00–0:10 | Intro: the cost-quality problem in enterprise search | Slides + discussion |
| 0:10–0:20 | The GEPA mental model (from a paper-level summary) | Slides |
| 0:20–0:30 | Setup — `00_quickstart_databricks.py` | Hands-on |
| 0:30–1:00 | Main workshop — `01_end_to_end_smallscale.py` | Hands-on, paced |
| 1:00–1:15 | Reading the scorecard, discussing the Pareto frontier | Group discussion |
| 1:15–1:30 | Knob exploration — try different `--candidate-selection-strategy` and scope settings | Hands-on |

For longer sessions (3 hours), add the optional notebooks (`02_optional_real_dspy_gepa.py`, `03_optional_real_mlflow_gepa.py`) and a customer-data adaptation exercise.

## Pre-workshop setup checklist (facilitator)

- [ ] Confirm everyone has Databricks workspace access (or Path A locally — see `PREREQUISITES.md`)
- [ ] Pre-fork the repo so attendees can clone fast
- [ ] Send out `PREREQUISITES.md` 24-48 hours ahead so anyone hitting setup snags resolves them before the session
- [ ] Pre-warm a serverless compute resource so the first install is fast
- [ ] Have a backup model endpoint configured for the optional notebooks in case attendees can't run them on their workspace

## During the workshop

### Before kicking off the hands-on portion

Set the framing: this is a teaching test stack with synthetic data and a deterministic optimizer. The point is the SHAPE of the workflow, not the absolute numbers. Real GEPA on real data will look different, but the decision flow — baseline → eval → feedback → optimization candidates → Pareto selection → scorecard — is the same.

### When attendees ask "why don't the numbers match real production"

Two answers:

1. The synthetic scorer is intentionally simple — it's checking keyword overlap and a few structure heuristics, not real customer-grade quality. Real evaluation needs real eval pipelines (LLM judges, human grading, downstream task accuracy).
2. The cost numbers are derived from token counts and a fixed price-per-token table, not real billing. They're useful for relative comparisons within a single run, not for absolute customer pricing.

### When attendees want to swap in their own data

The two extension points are:

1. `src/gepa_lab/data.py` — replace `enterprise_documents()` and `eval_examples()` with their own corpus + question set
2. `src/gepa_lab/metrics.py` — wire in their real eval test stack if they have one

Everything else (retrieval tools, agent, optimizer, scorecard) is reusable.

## Common pitfalls

- **`pip install -e .` fails on first cell** — usually because the cluster is still spinning up. Wait 30 seconds and retry, or use serverless.
- **Optional notebook 03 (MLflow) hangs** — Prompt Optimization beta needs MLflow 3.5+. Check `import mlflow; print(mlflow.__version__)` first.
- **Optional notebook 02 (DSPy) errors on import** — make sure `pip install dspy>=3.0.0 gepa>=0.1.0` happened in the same cluster session.
- **Attendees confused why mini_gepa is deterministic** — it's a teaching optimizer. Real GEPA uses LLM-driven reflection, which needs an LLM call per candidate. We made the teaching version deterministic so the workshop produces consistent scorecards across runs.

## Adapting for customer engagements

For customer-specific delivery:

- Sanitize any internal references in the slides
- Replace the "current high-cost baseline" simulator with a connector to the customer's actual existing system (Opus, GPT-5.5, whatever they're running today)
- Replace synthetic eval with the customer's real eval test stack early in the session
- Keep the Pareto-frontier discussion — it's the most important payoff for customer audiences

## Feedback collection

After every workshop session:

- 1 thing that landed well
- 1 thing that confused attendees
- 1 thing to add or remove next time

Track in a simple log so the workshop iteratively improves.
