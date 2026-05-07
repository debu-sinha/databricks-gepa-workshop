#!/usr/bin/env python3
"""Offline validation for the no-API lab path."""
from pathlib import Path

from gepa_lab.config import ExperimentConfig, OptimizerConfig
from gepa_lab.experiment import run_all


if __name__ == "__main__":
    out = Path("outputs_validation")
    cfg = ExperimentConfig(
        output_dir=str(out),
        use_mlflow_tracking=False,
        optimizer=OptimizerConfig(max_rounds=5, max_metric_calls=240, reflection_minibatch_size=4),
    )
    summary = run_all(cfg)
    assert (out / "summary.json").exists(), "summary.json not written"
    assert (out / "scorecard_test.csv").exists(), "scorecard not written"
    rows = summary["scorecard_test"]
    systems = {row["system"] for row in rows}
    assert "current high-cost app baseline" in systems
    assert "DSPy-style GEPA optimized" in systems
    assert "MLflow-style prompt GEPA optimized" in systems
    dspy = next(row for row in rows if row["system"] == "DSPy-style GEPA optimized")
    raw = next(row for row in rows if row["system"] == "smaller model raw")
    assert dspy["score"] >= raw["score"], "DSPy-style optimization should not regress on this toy lab"
    print("Validation passed.")
