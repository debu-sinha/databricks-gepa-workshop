from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean

from gepa_lab.agent import AgentRun
from gepa_lab.data import EvalExample


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("$", "").replace("/", " ").split())


def _contains(answer: str, item: str) -> bool:
    return _normalize(item) in _normalize(answer)


@dataclass(frozen=True)
class EvaluationRow:
    qid: str
    split: str
    category: str
    question: str
    answer: str
    expected_answer: str
    expected_doc_ids: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...]
    citations: tuple[str, ...]
    tool_types: tuple[str, ...]
    recall_at_5: float
    recall_at_10: float
    mrr: float
    precision: float
    correctness: float
    completeness: float
    groundedness: float
    latency_ms: float
    tool_calls: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    score: float
    feedback: str

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ["expected_doc_ids", "retrieved_doc_ids", "citations", "tool_types"]:
            d[key] = list(d[key])
        return d


@dataclass(frozen=True)
class AggregateMetrics:
    n: int
    score: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    precision: float
    correctness: float
    completeness: float
    groundedness: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_tool_calls: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_cost_usd: float
    cost_per_success_usd: float
    success_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_run(example: EvalExample, run: AgentRun) -> EvaluationRow:
    retrieved = list(run.retrieved_doc_ids)
    expected = list(example.expected_doc_ids)
    retrieved_top5 = set(retrieved[:5])
    retrieved_top10 = set(retrieved[:10])
    expected_set = set(expected)
    recall_at_5 = len(expected_set & retrieved_top5) / max(1, len(expected_set))
    recall_at_10 = len(expected_set & retrieved_top10) / max(1, len(expected_set))

    reciprocal_rank = 0.0
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in expected_set:
            reciprocal_rank = 1.0 / idx
            break

    precision = len(expected_set & set(retrieved[: max(1, len(expected))])) / max(1, min(len(retrieved), len(expected)))

    answer = run.answer
    must = list(example.must_contain)
    hits = [item for item in must if _contains(answer, item)]
    completeness = len(hits) / max(1, len(must))

    # Correctness is slightly stricter than completeness.
    if completeness >= 0.999:
        correctness = 1.0
    elif completeness >= 0.67:
        correctness = 0.7
    elif completeness >= 0.34:
        correctness = 0.4
    else:
        correctness = 0.0

    # Simple contradiction checks for absence/ambiguity cases.
    lower_answer = _normalize(answer)
    if "yoga mats" in _normalize(example.question) and "eligible" in lower_answer and "does not" not in lower_answer:
        correctness = 0.0
        completeness = min(completeness, 0.25)
    if "only asks what rl means" in _normalize(example.question) and "reliability label" not in lower_answer:
        correctness = min(correctness, 0.45)
        completeness = min(completeness, 0.5)

    citation_hit = len(set(run.citations) & expected_set) / max(1, len(expected_set))
    if run.policy.require_citations:
        groundedness = min(1.0, 0.65 * citation_hit + 0.35 * recall_at_5)
    else:
        groundedness = 0.35 * recall_at_5 if answer.strip() else 0.0

    # Penalize unsupported confident answers.
    if answer.lower().startswith("i do not have enough") or "not enough" in answer.lower():
        groundedness = max(groundedness, 0.45 if recall_at_5 > 0 else 0.2)
        correctness = min(correctness, 0.55)

    # Operationally-aware quality score. This is intentionally quality-heavy;
    # cost/latency are reported separately and used by the Pareto selector.
    score = (
        0.22 * recall_at_5
        + 0.10 * reciprocal_rank
        + 0.32 * correctness
        + 0.18 * completeness
        + 0.18 * groundedness
    )

    feedback_parts: list[str] = []
    missing_docs = [doc_id for doc_id in expected if doc_id not in retrieved_top5]
    if missing_docs:
        feedback_parts.append(f"retrieval miss: expected docs not in top-5: {missing_docs}")
    missing_terms = [item for item in must if item not in hits]
    if missing_terms:
        feedback_parts.append(f"answer missing required concepts: {missing_terms}")
    missing_tools = [tool for tool in example.expected_tool_types if tool not in [tc.tool_type for tc in run.tool_calls]]
    if missing_tools:
        feedback_parts.append(f"tool policy miss: expected tool types not used: {missing_tools}")
    if not run.citations:
        feedback_parts.append("grounding miss: answer has no citations")
    elif citation_hit < 1:
        feedback_parts.append("grounding miss: citations do not cover all expected documents")
    if run.latency_ms > 1200 and run.model.size_label == "smaller candidate":
        feedback_parts.append("operational miss: smaller model run is still too slow")
    if not feedback_parts:
        feedback_parts.append("good: retrieved expected evidence and answer covered required concepts")

    return EvaluationRow(
        qid=example.qid,
        split=example.split,
        category=example.category,
        question=example.question,
        answer=answer,
        expected_answer=example.expected_answer,
        expected_doc_ids=example.expected_doc_ids,
        retrieved_doc_ids=run.retrieved_doc_ids,
        citations=run.citations,
        tool_types=tuple(tc.tool_type for tc in run.tool_calls),
        recall_at_5=round(recall_at_5, 4),
        recall_at_10=round(recall_at_10, 4),
        mrr=round(reciprocal_rank, 4),
        precision=round(precision, 4),
        correctness=round(correctness, 4),
        completeness=round(completeness, 4),
        groundedness=round(groundedness, 4),
        latency_ms=run.latency_ms,
        tool_calls=len(run.tool_calls),
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        estimated_cost_usd=run.estimated_cost_usd,
        score=round(score, 4),
        feedback=" | ".join(feedback_parts),
    )


def aggregate(rows: list[EvaluationRow]) -> AggregateMetrics:
    if not rows:
        return AggregateMetrics(0, *([0.0] * 15))  # type: ignore[arg-type]

    latencies = sorted(row.latency_ms for row in rows)
    p95_index = max(0, min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1)))))
    success_rows = [row for row in rows if row.score >= 0.80]
    total_cost = sum(row.estimated_cost_usd for row in rows)
    cost_per_success = total_cost / max(1, len(success_rows))
    return AggregateMetrics(
        n=len(rows),
        score=round(mean(row.score for row in rows), 4),
        recall_at_5=round(mean(row.recall_at_5 for row in rows), 4),
        recall_at_10=round(mean(row.recall_at_10 for row in rows), 4),
        mrr=round(mean(row.mrr for row in rows), 4),
        precision=round(mean(row.precision for row in rows), 4),
        correctness=round(mean(row.correctness for row in rows), 4),
        completeness=round(mean(row.completeness for row in rows), 4),
        groundedness=round(mean(row.groundedness for row in rows), 4),
        avg_latency_ms=round(mean(row.latency_ms for row in rows), 3),
        p95_latency_ms=round(latencies[p95_index], 3),
        avg_tool_calls=round(mean(row.tool_calls for row in rows), 3),
        avg_input_tokens=round(mean(row.input_tokens for row in rows), 3),
        avg_output_tokens=round(mean(row.output_tokens for row in rows), 3),
        avg_cost_usd=round(mean(row.estimated_cost_usd for row in rows), 8),
        cost_per_success_usd=round(cost_per_success, 8),
        success_rate=round(len(success_rows) / len(rows), 4),
    )


def evaluate_agent(name: str, examples: list[EvalExample], agent) -> tuple[str, list[EvaluationRow], AggregateMetrics]:
    rows = [evaluate_run(example, agent.run(example.question)) for example in examples]
    return name, rows, aggregate(rows)
