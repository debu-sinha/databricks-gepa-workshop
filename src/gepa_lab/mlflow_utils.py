from __future__ import annotations

from pathlib import Path


def try_log_to_mlflow(
    *,
    experiment_name: str,
    run_name: str,
    params: dict,
    metrics: dict,
    artifact_dir: str | Path,
) -> str | None:
    """Best-effort MLflow logging.

    The core lab runs without MLflow. On Databricks, this will usually work and
    gives you the governance/tracking shape you need for the real engagement.
    """

    try:
        import mlflow  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        print(f"[mlflow] MLflow not available; skipped tracking. Reason: {e}")
        return None

    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            for key, value in params.items():
                if isinstance(value, (dict, list, tuple)):
                    mlflow.log_param(key, str(value)[:500])
                else:
                    mlflow.log_param(key, value)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
            if Path(artifact_dir).exists():
                mlflow.log_artifacts(str(artifact_dir))
            print(f"[mlflow] logged run_id={run.info.run_id}")
            return run.info.run_id
    except Exception as e:  # pragma: no cover - environment-specific
        print(f"[mlflow] Tracking failed but local artifacts were written. Reason: {e}")
        return None
