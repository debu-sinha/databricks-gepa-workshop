# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Optional real MLflow Prompt Optimization template
# MAGIC
# MAGIC The primary lab logs to MLflow if available. Use this after you have MLflow >= 3.5, Prompt Registry access, and a model endpoint.

# COMMAND ----------

# MAGIC %pip install -r requirements-optional.txt

# COMMAND ----------

from gepa_lab.optional_true_mlflow import print_real_mlflow_gepa_template
print_real_mlflow_gepa_template()
