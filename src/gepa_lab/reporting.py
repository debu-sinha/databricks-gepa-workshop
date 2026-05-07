from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from gepa_lab.metrics import AggregateMetrics, EvaluationRow
from gepa_lab.mini_gepa import OptimizerOutput


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_to_row(name: str, metrics: AggregateMetrics, extra: dict | None = None) -> dict:
    row = {"system": name, **metrics.to_dict()}
    if extra:
        row.update(extra)
    return row


def print_scorecard(rows: list[dict]) -> None:
    columns = [
        "system",
        "score",
        "recall_at_5",
        "correctness",
        "completeness",
        "groundedness",
        "avg_latency_ms",
        "p95_latency_ms",
        "avg_cost_usd",
        "cost_per_success_usd",
        "success_rate",
        "avg_tool_calls",
    ]
    widths = {c: max(len(c), *(len(str(row.get(c, ""))) for row in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))


def rows_to_dicts(rows: list[EvaluationRow]) -> list[dict]:
    return [row.to_dict() for row in rows]


def optimizer_to_artifacts(output: OptimizerOutput) -> dict:
    return {
        "approach": output.approach,
        "best_candidate_id": output.best.candidate_id,
        "best_mutation": output.best.mutation,
        "best_metrics": output.best.metrics.to_dict(),
        "best_policy": output.best.policy.to_dict(),
        "frontier": [
            {
                "candidate_id": cand.candidate_id,
                "mutation": cand.mutation,
                "metrics": cand.metrics.to_dict(),
                "utility": cand.utility,
                "policy": cand.policy.to_dict(),
            }
            for cand in output.frontier
        ],
    }
