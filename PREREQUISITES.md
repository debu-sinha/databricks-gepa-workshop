# Prerequisites

This workshop has two paths. Pick whichever fits your situation.

## Path A — Local only (no Databricks)

You'll get the full GEPA-style optimization flow with a deterministic mini optimizer, synthetic data, and a printed scorecard. No model endpoints, no cloud, no money.

Requirements:
- Python 3.10 or later
- ~500 MB disk for the venv

Setup:

```bash
git clone https://github.com/debu-sinha/databricks-gepa-workshop.git
cd databricks-gepa-workshop
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/validate.py
```

If `validate.py` prints "Validation passed" and a scorecard table, you're done.

## Path B — Databricks workspace (full workshop, recommended)

You'll get everything from Path A plus the optional notebooks that exercise real DSPy GEPA, MLflow Prompt Optimization (beta), Mosaic AI Agent Framework, MLflow tracing, and Foundation Model API serving.

### Requirements on the workspace

- Databricks workspace with:
  - **Unity Catalog** enabled (for the `traces in UC` quickstart and governance demos)
  - **Foundation Model API** access (used to call hosted models for non-deterministic baselines)
  - **MLflow 3.5+** (for prompt registry + Prompt Optimization beta)
  - **Compute** that can install packages from PyPI (cluster + serverless both work; serverless is simpler for the workshop)
- A user account with permission to:
  - Create / write to a personal schema in Unity Catalog
  - Read/write to MLflow experiments
  - Use serverless compute or attach to an interactive cluster

### Setup on FEVM (Databricks Field Engineering)

If you're a Databricks Field Engineer, FEVM gives you a free dev workspace fast.

1. **Provision a workspace via FEVM** (vibe plugin or your team's FEVM URL). Pick a region close to you.
2. **Open the workspace** and create a new Repo:
   - Repos → Add Repo
   - URL: `https://github.com/debu-sinha/databricks-gepa-workshop.git` (or fork it first)
   - Click "Create"
3. **Open `notebooks/00_quickstart_databricks.py`** and attach to a cluster (serverless works fine).
4. **Run the cells top-to-bottom.** First cell installs the package via `%pip install -e .`.
5. **Open `notebooks/01_end_to_end_smallscale.py`** and run it. This is the main workshop content.
6. **Optional:** `notebooks/02_optional_real_dspy_gepa.py` and `notebooks/03_optional_real_mlflow_gepa.py` if you want to exercise the real DSPy / MLflow integrations against a model endpoint.

### Setup on a customer Databricks workspace

If you're delivering this as a workshop to a customer:

1. Have the customer fork this repo into their org / private GitHub
2. Have them create a Repo from their fork in their workspace
3. Run `notebooks/00_quickstart_databricks.py` first to confirm the setup
4. Run `notebooks/01_end_to_end_smallscale.py` together as the workshop body
5. For the optional notebooks, they'll need:
   - A Unity Catalog schema they own
   - Foundation Model API access (or their own endpoint)
   - MLflow 3.5+ on the workspace

### Workspace-side configuration

Set these once at the top of any notebook that hits real model endpoints:

```python
import os
os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get(scope="<your-scope>", key="<your-key>")
os.environ["DATABRICKS_HOST"] = "https://<your-workspace>.cloud.databricks.com"
```

For Foundation Model API calls, no token is needed inside a Databricks notebook — the SDK picks up workspace auth automatically.

## What if you don't have a Databricks workspace?

You can still run Path A locally. The deterministic `mini_gepa` optimizer teaches the full GEPA-style learning flow without any model endpoint. The scorecard you produce is meaningful for understanding the technique, even though the absolute numbers are synthetic.
