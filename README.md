# Databricks GEPA workshop

A small, working, end-to-end lab that teaches GEPA-style prompt and program optimization for enterprise search and retrieval. Designed as a Databricks workshop template — runs locally or on a Databricks workspace, no customer data needed.

It's intentionally tiny and safe:

- No proprietary data
- No model endpoint required for the default path
- No GPU required
- No external API calls in the default path
- Runs on Databricks (Free Edition / serverless notebooks both work) or locally with Python 3.10+

The lab compares a **current high-cost application baseline** (the kind of setup that uses Opus / GPT-class models for retrieval and generation today) against a **smaller-model baseline** and a **GEPA-optimized smaller-model prototype**.

> **Status:** Workshop template. See [`WORKSHOP.md`](WORKSHOP.md) for facilitation guide and [`PREREQUISITES.md`](PREREQUISITES.md) for setup paths.

## What you will learn

You will see the full flow:

```text
Synthetic enterprise docs
        ↓
Synthetic eval examples
        ↓
Semantic search / lexical search / tiny KG tools
        ↓
Current high-cost app baseline
        ↓
Smaller-model raw baseline
        ↓
DSPy-style full-program GEPA optimization
        ↓
MLflow-style prompt-only GEPA optimization
        ↓
Quality + cost + latency scorecard
```

The deterministic optimizer in this repo is called `mini_gepa`. It isn't a replacement for real GEPA — it's a transparent teaching scaffold that follows the same shape:

```text
evaluate → collect textual feedback → reflect/propose changes → evaluate candidates → keep Pareto candidates
```

Real DSPy GEPA and MLflow Prompt Optimization use LLM-driven reflection. Optional templates are included under `notebooks/02_optional_real_dspy_gepa.py` and `notebooks/03_optional_real_mlflow_gepa.py`.

## Run in Databricks

Upload this folder to a Databricks workspace, then run:

```text
notebooks/00_quickstart_databricks.py
notebooks/01_end_to_end_smallscale.py
```

The quickstart notebook runs:

```python
%pip install -e .
```

The core lab uses only standard Python. MLflow tracking is optional.

## Run locally

From this repository root:

```bash
python -m pip install -e .
python scripts/run_all.py --no-mlflow --output-dir outputs/local_run
```

Or run the validation smoke test:

```bash
python scripts/validate.py
```

## Expected outputs

The run writes:

```text
outputs/<run>/summary.json
outputs/<run>/scorecard_test.csv
outputs/<run>/dspy_style_history.csv
outputs/<run>/mlflow_style_history.csv
outputs/<run>/dspy_style_optimizer.json
outputs/<run>/mlflow_style_optimizer.json
outputs/<run>/predictions_current_baseline_test.jsonl
outputs/<run>/predictions_small_raw_test.jsonl
outputs/<run>/predictions_dspy_optimized_test.jsonl
outputs/<run>/predictions_mlflow_optimized_test.jsonl
```

The scorecard compares:

1. current high-cost app baseline
2. smaller model raw baseline
3. DSPy-style GEPA optimized smaller model
4. MLflow-style prompt GEPA optimized smaller model

## Why there are two GEPA styles

### DSPy-style GEPA

This path can optimize the **whole program**:

- query rewriting
- semantic search top-k
- lexical search use
- KG/EPKG tool use
- citation behavior
- abstention behavior
- grounding check
- answer synthesis style
- context budget
- max tool calls

This is the path you want when the problem is an agentic retrieval system, not a single answer prompt.

### MLflow-style GEPA

The default MLflow-style path optimizes mostly the **answer prompt behavior**:

- citations
- abstention
- grounding check
- answer style

You can switch it to a more capable multi-prompt mode:

```bash
python scripts/run_all.py --no-mlflow --mlflow-style-scope multi_prompt
```

That approximates optimizing planner + answer prompts together.

## Useful knobs

```bash
python scripts/run_all.py \
  --no-mlflow \
  --max-rounds 5 \
  --max-metric-calls 240 \
  --reflection-minibatch-size 4 \
  --candidate-selection-strategy pareto \
  --dspy-style-scope full_program \
  --mlflow-style-scope answer_only
```

See `docs/KNOBS.md` for a simple explanation of each knob.

## How to connect to real Databricks endpoints later

The default lab doesn't call a model. Once you've got Databricks endpoint access, install optional dependencies:

```python
%pip install -r requirements-optional.txt
```

Then inspect:

```text
notebooks/02_optional_real_dspy_gepa.py
notebooks/03_optional_real_mlflow_gepa.py
```

You'll need:

- a Databricks workspace
- a model serving endpoint
- an eval scorer or pipeline
- train/dev/test examples
- permission to log prompts, traces, retrieved docs, model outputs, and eval results

## How to use this as a workshop scaffold

When you adapt this for a real customer engagement, use the lab as the shape for your first prototype scorecard:

| System | Purpose |
|---|---|
| Current Opus/GPT-style app baseline | What the customer is using today |
| Smaller model raw | What happens before optimization |
| Smaller model + DSPy/GEPA | Best full-program optimization path |
| Smaller model + MLflow GEPA | More governed prompt-registry optimization path |

The first real decision isn't "GEPA vs RL forever." It's:

> Can prompt/program optimization get a smaller model close enough to the current quality bar while reducing cost and latency?

If not, the next path is RL or PEFT or distillation. This lab gives you the scorecard shape that informs that call.
