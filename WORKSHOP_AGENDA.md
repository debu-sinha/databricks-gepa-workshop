# GEPA workshop agenda

A focused workshop on **GEPA-style prompt and program optimization for enterprise search and retrieval**. Built around the Meta engagement use case: replacing high-cost Opus/GPT-class baselines with smaller models that match quality at a fraction of the cost via systematic prompt + program optimization.

> **Scope note:** This workshop covers GEPA optimization specifically. It's a sibling to the **MLflow Eval at Scale workshop** (separate repo, built around the Scribd use case — traces, judges, review app, observability, DataDog integration). The two workshops are independent and serve different ASQs.

## Audience

Technical practitioners who already understand prompt engineering and want to learn systematic prompt + program optimization. Data scientists, ML engineers, AI platform engineers.

## Outcomes

By the end, attendees should be able to:

- Explain GEPA-style optimization and how it differs from manual prompt tuning, RAG-only approaches, and full RL post-training
- Read a quality / cost / latency scorecard and identify the Pareto frontier
- Pick when to reach for DSPy GEPA versus MLflow Prompt Optimization (beta)
- Adapt the synthetic test setup to their own enterprise search / retrieval use case
- Understand which knobs matter most: candidate-selection strategy, optimization scope, max metric calls, reflection minibatch size

## Three formats

### 90-minute version (default)

| Time | Section | Format |
|---|---|---|
| 0:00–0:10 | The cost-quality problem in enterprise search | Slides |
| 0:10–0:20 | The GEPA mental model (paper-level summary) | Slides |
| 0:20–0:30 | Setup — `notebooks/00_quickstart_databricks.py` | Hands-on |
| 0:30–1:00 | Main workshop — `notebooks/01_end_to_end_smallscale.py` | Hands-on |
| 1:00–1:15 | Reading the scorecard, Pareto frontier discussion | Group discussion |
| 1:15–1:30 | Knob exploration — different `--candidate-selection-strategy` and scope settings | Hands-on |

This is the primary delivery format.

### 3-hour deep dive

Add the optional notebooks after the 90-minute core:

| Time | Section |
|---|---|
| 1:30–2:15 | Real DSPy GEPA — `notebooks/02_optional_real_dspy_gepa.py` |
| 2:15–3:00 | MLflow Prompt Optimization (beta) — `notebooks/03_optional_real_mlflow_gepa.py` |

The deep dive requires a live Databricks workspace with Foundation Model API access.

### Customer-engagement adaptation

For customer-specific delivery (e.g., the Meta engagement):

- Replace synthetic data in `src/gepa_lab/data.py` with the customer's corpus + question set
- Replace synthetic eval in `src/gepa_lab/metrics.py` with the customer's real eval pipeline
- Keep the optimization scaffolding (retrieval, agent, mini_gepa, scorecard) — that's reusable
- Add a customer baseline by wiring the customer's existing high-cost system (Opus, GPT-5.5, etc.) into the baseline simulator
- Lock the workshop to half a day or less for senior customer audiences

## Pre-workshop checklist (facilitator)

- [ ] Confirm attendees have Databricks workspace access OR Python 3.10+ locally (see `PREREQUISITES.md`)
- [ ] Pre-fork the repo so attendees clone fast
- [ ] Send `PREREQUISITES.md` 24-48 hours ahead so setup snags resolve before the session
- [ ] Pre-warm a serverless compute resource so the first install is fast
- [ ] Have a backup model endpoint for the optional notebooks if any attendee can't run them on their workspace

## During the workshop

### Set the framing first

This is a teaching test stack with synthetic data and a deterministic optimizer. The point is the SHAPE of the workflow, not the absolute numbers. Real GEPA on real data will look different, but the decision flow — baseline → eval → feedback → optimization candidates → Pareto selection → scorecard — is the same.

### When attendees ask "why don't the numbers match real production"

Two answers:

1. The synthetic scorer is intentionally simple — it's checking keyword overlap and a few structure heuristics, not real customer-grade quality. Real evaluation needs real eval pipelines (LLM judges, human grading, downstream task accuracy).
2. The cost numbers come from token counts and a fixed price-per-token table, not real billing. They're useful for relative comparisons within a single run, not for absolute customer pricing.

### When attendees want to swap in their own data

Two extension points:

1. `src/gepa_lab/data.py` — replace `enterprise_documents()` and `eval_examples()` with their corpus + question set
2. `src/gepa_lab/metrics.py` — wire in their real eval pipeline if they have one

Everything else (retrieval tools, agent, optimizer, scorecard) is reusable.

## Common pitfalls

- **`pip install -e .` fails on first cell** — usually because the cluster's still spinning up. Wait 30 seconds and retry, or use serverless.
- **Optional notebook 03 (MLflow) hangs** — Prompt Optimization beta needs MLflow 3.5+. Check `import mlflow; print(mlflow.__version__)` first.
- **Optional notebook 02 (DSPy) errors on import** — make sure `pip install dspy>=3.0.0 gepa>=0.1.0` happened in the same cluster session.
- **Attendees confused why mini_gepa is deterministic** — it's a teaching optimizer. Real GEPA uses LLM-driven reflection, which needs an LLM call per candidate. We made the teaching version deterministic so the workshop produces consistent scorecards across runs.

## Feedback loop

After every workshop session, capture:

- 1 thing that landed well
- 1 thing that confused attendees
- 1 thing to add or remove next time

Track in a simple log so the workshop iteratively improves.

## Sibling workshop

The **MLflow Eval at Scale workshop** (separate repo) covers the other side of the production-AI story: traces at scale, custom scorers, LLM-judge calibration, human review apps, and observability integration (DataDog / open telemetry). When delivering for a customer who needs both, run them as a two-day pair — GEPA on day 1, eval at scale on day 2.

Both workshops can be run independently for customers who only need one.
