# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Optional real DSPy GEPA template
# MAGIC
# MAGIC The primary lab is deterministic. Use this after you have a Databricks model endpoint and want to wire the toy search program into real DSPy GEPA.

# COMMAND ----------

# MAGIC %pip install -r requirements-optional.txt

# COMMAND ----------

from gepa_lab.optional_true_dspy import run_real_dspy_gepa_if_available
run_real_dspy_gepa_if_available()
