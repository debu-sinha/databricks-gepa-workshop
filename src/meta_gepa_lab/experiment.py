from __future__ import annotations

import argparse
from pathlib import Path

from meta_gepa_lab.agent import SearchAgent
from meta_gepa_lab.config import (
    CURRENT_APP_BASELINE_POLICY,
    CURRENT_LARGE_MODEL,
    MLFLOW_STYLE_START_POLICY,
    SMALL_MODEL,
    SMALL_RAW_POLICY,
    ExperimentConfig,
    OptimizerConfig,
)
from meta_gepa_lab.data import build_documents, build_eval_examples, by_split
from meta_gepa_lab.metrics import aggregate, evaluate_agent, evaluate_run
from meta_gepa_lab.mini_gepa import run_mini_gepa
from meta_gepa_lab.mlflow_utils import try_log_to_mlflow
from meta_gepa_lab.reporting import (
    aggregate_to_row,
    ensure_dir,
    optimizer_to_artifacts,
    print_scorecard,
    rows_to_dicts,
    write_csv,
    write_json,
    write_jsonl,
)
from meta_gepa_lab.retrieval import SearchTools


def run_all(config: ExperimentConfig | None = None) -> dict:
    cfg = config or ExperimentConfig()
    out_dir = ensure_dir(cfg.output_dir)

    docs = build_documents()
    examples = build_eval_examples()
    train = by_split(examples, "train")
    dev = by_split(examples, "dev")
    test = by_split(examples, "test")
    tools = SearchTools(docs)

    print("Loaded synthetic enterprise corpus:")
    print(f"  docs={len(docs)} train={len(train)} dev={len(dev)} test={len(test)}")
    print("\nRunning baselines...")

    current_agent = SearchAgent(tools=tools, model=CURRENT_LARGE_MODEL, policy=CURRENT_APP_BASELINE_POLICY)
    small_raw_agent = SearchAgent(tools=tools, model=SMALL_MODEL, policy=SMALL_RAW_POLICY)

    _, current_test_rows, current_test_metrics = evaluate_agent("current_high_cost_app_baseline", test, current_agent)
    _, small_raw_test_rows, small_raw_test_metrics = evaluate_agent("smaller_model_raw_prompt", test, small_raw_agent)

    print("\nOptimizing DSPy-style full program policy with mini-GEPA...")
    dspy_output = run_mini_gepa(
        approach="dspy_style_full_program_gepa",
        start_policy=SMALL_RAW_POLICY,
        tools=tools,
        model=SMALL_MODEL,
        train_examples=train,
        cfg=cfg.optimizer,
        mutation_scope=cfg.optimizer.dspy_style_mutation_scope,
    )
    dspy_best_agent = SearchAgent(tools=tools, model=SMALL_MODEL, policy=dspy_output.best.policy)
    dspy_dev_rows = [evaluate_run(example, dspy_best_agent.run(example.question)) for example in dev]
    dspy_test_rows = [evaluate_run(example, dspy_best_agent.run(example.question)) for example in test]
    dspy_dev_metrics = aggregate(dspy_dev_rows)
    dspy_test_metrics = aggregate(dspy_test_rows)

    print("\nOptimizing MLflow-style prompt policy with mini-GEPA...")
    mlflow_output = run_mini_gepa(
        approach="mlflow_style_prompt_gepa",
        start_policy=MLFLOW_STYLE_START_POLICY,
        tools=tools,
        model=SMALL_MODEL,
        train_examples=train,
        cfg=cfg.optimizer,
        mutation_scope=cfg.optimizer.mlflow_style_mutation_scope,
    )
    mlflow_best_agent = SearchAgent(tools=tools, model=SMALL_MODEL, policy=mlflow_output.best.policy)
    mlflow_dev_rows = [evaluate_run(example, mlflow_best_agent.run(example.question)) for example in dev]
    mlflow_test_rows = [evaluate_run(example, mlflow_best_agent.run(example.question)) for example in test]
    mlflow_dev_metrics = aggregate(mlflow_dev_rows)
    mlflow_test_metrics = aggregate(mlflow_test_rows)

    scorecard_rows = [
        aggregate_to_row(
            "current high-cost app baseline", current_test_metrics,
            {"split": "test", "model": CURRENT_LARGE_MODEL.name, "policy": CURRENT_APP_BASELINE_POLICY.name},
        ),
        aggregate_to_row(
            "smaller model raw", small_raw_test_metrics,
            {"split": "test", "model": SMALL_MODEL.name, "policy": SMALL_RAW_POLICY.name},
        ),
        aggregate_to_row(
            "DSPy-style GEPA optimized", dspy_test_metrics,
            {"split": "test", "model": SMALL_MODEL.name, "policy": dspy_output.best.policy.name},
        ),
        aggregate_to_row(
            "MLflow-style prompt GEPA optimized", mlflow_test_metrics,
            {"split": "test", "model": SMALL_MODEL.name, "policy": mlflow_output.best.policy.name},
        ),
    ]

    print("\nTest scorecard:")
    print_scorecard(scorecard_rows)

    # Persist artifacts.
    write_csv(out_dir / "scorecard_test.csv", scorecard_rows)
    write_csv(out_dir / "dspy_style_history.csv", dspy_output.history_rows())
    write_csv(out_dir / "mlflow_style_history.csv", mlflow_output.history_rows())
    write_json(out_dir / "dspy_style_optimizer.json", optimizer_to_artifacts(dspy_output))
    write_json(out_dir / "mlflow_style_optimizer.json", optimizer_to_artifacts(mlflow_output))
    write_jsonl(out_dir / "predictions_current_baseline_test.jsonl", rows_to_dicts(current_test_rows))
    write_jsonl(out_dir / "predictions_small_raw_test.jsonl", rows_to_dicts(small_raw_test_rows))
    write_jsonl(out_dir / "predictions_dspy_optimized_test.jsonl", rows_to_dicts(dspy_test_rows))
    write_jsonl(out_dir / "predictions_mlflow_optimized_test.jsonl", rows_to_dicts(mlflow_test_rows))

    summary = {
        "config": cfg.to_dict(),
        "dataset": {"docs": len(docs), "train": len(train), "dev": len(dev), "test": len(test)},
        "scorecard_test": scorecard_rows,
        "validation": {
            "dspy_style_gepa": dspy_dev_metrics.to_dict(),
            "mlflow_style_prompt_gepa": mlflow_dev_metrics.to_dict(),
        },
        "best_policies": {
            "dspy_style_gepa": dspy_output.best.policy.to_dict(),
            "mlflow_style_prompt_gepa": mlflow_output.best.policy.to_dict(),
        },
        "artifact_dir": str(Path(out_dir).resolve()),
        "important_note": (
            "This is a deterministic mini-GEPA teaching scaffold. It mirrors the evaluate -> feedback -> reflect -> mutate -> "
            "select flow, but real DSPy GEPA / MLflow Prompt Optimization use LLM-driven reflection."
        ),
    }
    write_json(out_dir / "summary.json", summary)

    if cfg.use_mlflow_tracking:
        # Log one compact MLflow run. The artifacts contain detailed histories and predictions.
        best_metrics = {
            "current_baseline_score": current_test_metrics.score,
            "small_raw_score": small_raw_test_metrics.score,
            "dspy_gepa_score": dspy_test_metrics.score,
            "mlflow_gepa_score": mlflow_test_metrics.score,
            "current_baseline_avg_cost_usd": current_test_metrics.avg_cost_usd,
            "small_raw_avg_cost_usd": small_raw_test_metrics.avg_cost_usd,
            "dspy_gepa_avg_cost_usd": dspy_test_metrics.avg_cost_usd,
            "mlflow_gepa_avg_cost_usd": mlflow_test_metrics.avg_cost_usd,
            "current_baseline_avg_latency_ms": current_test_metrics.avg_latency_ms,
            "small_raw_avg_latency_ms": small_raw_test_metrics.avg_latency_ms,
            "dspy_gepa_avg_latency_ms": dspy_test_metrics.avg_latency_ms,
            "mlflow_gepa_avg_latency_ms": mlflow_test_metrics.avg_latency_ms,
        }
        try_log_to_mlflow(
            experiment_name=cfg.mlflow_experiment_name,
            run_name="meta_gepa_smallscale_lab",
            params={
                "docs": len(docs),
                "train_examples": len(train),
                "dev_examples": len(dev),
                "test_examples": len(test),
                "optimizer_max_rounds": cfg.optimizer.max_rounds,
                "optimizer_max_metric_calls": cfg.optimizer.max_metric_calls,
                "dspy_style_mutation_scope": cfg.optimizer.dspy_style_mutation_scope,
                "mlflow_style_mutation_scope": cfg.optimizer.mlflow_style_mutation_scope,
            },
            metrics=best_metrics,
            artifact_dir=out_dir,
        )

    print(f"\nWrote artifacts to: {Path(out_dir).resolve()}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the small-scale Meta/KARL GEPA lab.")
    parser.add_argument("--output-dir", default="outputs", help="Where to write artifacts")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-metric-calls", type=int, default=240)
    parser.add_argument("--reflection-minibatch-size", type=int, default=4)
    parser.add_argument("--candidate-selection-strategy", choices=["pareto", "best_score", "random"], default="pareto")
    parser.add_argument("--mlflow-style-scope", choices=["answer_only", "multi_prompt"], default="answer_only")
    parser.add_argument("--dspy-style-scope", choices=["full_program", "answer_only"], default="full_program")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opt_cfg = OptimizerConfig(
        max_rounds=args.max_rounds,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        candidate_selection_strategy=args.candidate_selection_strategy,
        dspy_style_mutation_scope=args.dspy_style_scope,
        mlflow_style_mutation_scope=args.mlflow_style_scope,
    )
    cfg = ExperimentConfig(
        output_dir=args.output_dir,
        use_mlflow_tracking=not args.no_mlflow,
        optimizer=opt_cfg,
    )
    run_all(cfg)


if __name__ == "__main__":
    main()
