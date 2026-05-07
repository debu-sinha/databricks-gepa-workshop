# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - End-to-end small-scale GEPA lab
# MAGIC
# MAGIC This notebook runs the full deterministic flow:
# MAGIC
# MAGIC 1. synthetic enterprise corpus
# MAGIC 2. train/dev/test eval examples
# MAGIC 3. current high-cost app baseline
# MAGIC 4. smaller model raw baseline
# MAGIC 5. DSPy-style full-program GEPA optimization
# MAGIC 6. MLflow-style prompt GEPA optimization
# MAGIC 7. scorecard and artifacts

# COMMAND ----------

from gepa_lab.config import ExperimentConfig, OptimizerConfig
from gepa_lab.experiment import run_all

cfg = ExperimentConfig(
    output_dir="outputs/smallscale_run",
    use_mlflow_tracking=True,  # set False if not in Databricks or MLflow is unavailable
    optimizer=OptimizerConfig(
        max_rounds=5,
        max_metric_calls=240,
        reflection_minibatch_size=4,
        candidate_selection_strategy="pareto",
        dspy_style_mutation_scope="full_program",
        mlflow_style_mutation_scope="answer_only",  # try "multi_prompt" too
    ),
)

summary = run_all(cfg)
summary["scorecard_test"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect optimized policies

# COMMAND ----------

summary["best_policies"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Where artifacts were written

# COMMAND ----------

summary["artifact_dir"]
