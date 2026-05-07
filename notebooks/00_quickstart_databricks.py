# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Quickstart setup
# MAGIC
# MAGIC Upload or clone this folder into Databricks, then run this notebook from the repository root.
# MAGIC The core lab uses only standard Python. MLflow tracking is optional and usually available in Databricks.

# COMMAND ----------

# MAGIC %pip install -e .

# COMMAND ----------

# MAGIC %md
# MAGIC Optional packages for real model endpoint / real GEPA runs:
# MAGIC
# MAGIC ```python
# MAGIC %pip install -r requirements-optional.txt
# MAGIC ```
# MAGIC
# MAGIC The no-API lab does not require these packages.

# COMMAND ----------

import sys
from gepa_lab import __version__
print("gepa_lab version:", __version__)
print("Python:", sys.version)
