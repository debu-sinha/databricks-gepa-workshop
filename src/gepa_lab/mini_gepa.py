from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace
from typing import Iterable, Literal

from gepa_lab.agent import SearchAgent
from gepa_lab.config import ModelProfile, OptimizerConfig, PromptPolicy
from gepa_lab.data import EvalExample
from gepa_lab.metrics import AggregateMetrics, EvaluationRow, aggregate, evaluate_run
from gepa_lab.retrieval import SearchTools

MutationScope = Literal["full_program", "answer_only", "multi_prompt"]


@dataclass(frozen=True)
class CandidateResult:
    approach: str
    round_index: int
    candidate_id: str
    parent_id: str | None
    mutation: str
    policy: PromptPolicy
    rows: tuple[EvaluationRow, ...]
    metrics: AggregateMetrics
    utility: float

    def to_history_row(self) -> dict:
        return {
            "approach": self.approach,
            "round_index": self.round_index,
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "mutation": self.mutation,
            "policy_name": self.policy.name,
            "score": self.metrics.score,
            "correctness": self.metrics.correctness,
            "completeness": self.metrics.completeness,
            "groundedness": self.metrics.groundedness,
            "recall_at_5": self.metrics.recall_at_5,
            "mrr": self.metrics.mrr,
            "avg_latency_ms": self.metrics.avg_latency_ms,
            "p95_latency_ms": self.metrics.p95_latency_ms,
            "avg_cost_usd": self.metrics.avg_cost_usd,
            "avg_tool_calls": self.metrics.avg_tool_calls,
            "success_rate": self.metrics.success_rate,
            **{f"policy_{k}": v for k, v in self.policy.to_dict().items() if k != "instruction"},
        }


def _policy_signature(policy: PromptPolicy) -> str:
    # Ignore candidate name/instruction wording for duplicate detection; focus on behavior knobs.
    d = policy.to_dict().copy()
    d.pop("name", None)
    d.pop("instruction", None)
    raw = repr(sorted(d.items()))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _candidate_id(policy: PromptPolicy, round_index: int, salt: str) -> str:
    return f"r{round_index}_{_policy_signature(policy)}_{hashlib.sha1(salt.encode()).hexdigest()[:6]}"


def _utility(metrics: AggregateMetrics, cfg: OptimizerConfig) -> float:
    # Convert cost/latency to bounded penalties. These coefficients are tuned for the synthetic lab.
    latency_penalty = min(1.0, metrics.avg_latency_ms / 2500.0)
    cost_penalty = min(1.0, metrics.avg_cost_usd / 0.006)
    return round(
        cfg.frontier_quality_weight * metrics.score
        - cfg.frontier_cost_weight * cost_penalty
        - cfg.frontier_latency_weight * latency_penalty,
        6,
    )


def _evaluate_policy(
    approach: str,
    round_index: int,
    policy: PromptPolicy,
    tools: SearchTools,
    model: ModelProfile,
    examples: list[EvalExample],
    cfg: OptimizerConfig,
    mutation: str,
    parent_id: str | None,
) -> CandidateResult:
    agent = SearchAgent(tools=tools, model=model, policy=policy)
    rows = [evaluate_run(example, agent.run(example.question)) for example in examples]
    metrics = aggregate(rows)
    cid = _candidate_id(policy, round_index, mutation + (parent_id or "root"))
    return CandidateResult(
        approach=approach,
        round_index=round_index,
        candidate_id=cid,
        parent_id=parent_id,
        mutation=mutation,
        policy=policy,
        rows=tuple(rows),
        metrics=metrics,
        utility=_utility(metrics, cfg),
    )


def _feedback_signals(rows: Iterable[EvaluationRow]) -> set[str]:
    text = "\n".join(row.feedback.lower() for row in rows)
    signals: set[str] = set()
    if "retrieval miss" in text:
        signals.add("retrieval")
    if "tool policy miss" in text:
        if "lexical" in text:
            signals.add("lexical")
        if "kg" in text:
            signals.add("kg")
    if "answer missing" in text:
        signals.add("answer")
    if "grounding miss" in text:
        signals.add("grounding")
    if "too slow" in text:
        signals.add("latency")
    # Include category-derived signals when many rows are weak.
    for row in rows:
        if row.score < 0.75:
            if "relationship" in row.category or "multi_document" in row.category:
                signals.add("kg")
                signals.add("retrieval")
            if "absence" in row.category or "ambiguous" in row.category:
                signals.add("abstain")
            if "conditional" in row.category or "procedural" in row.category:
                signals.add("lexical")
    return signals


def _mutate_policy(policy: PromptPolicy, signals: set[str], scope: MutationScope, round_index: int) -> list[tuple[str, PromptPolicy]]:
    """Create candidate prompt/program policies from textual feedback.

    This deliberately resembles GEPA's reflect-and-propose loop, but is local and
    deterministic. Real GEPA would ask a reflection LM to propose the text changes.
    """

    mutations: list[tuple[str, PromptPolicy]] = []

    def add(name: str, **kwargs) -> None:
        new_instruction = kwargs.pop("instruction", None)
        base_instruction = policy.instruction
        if new_instruction:
            base_instruction = base_instruction + "\nOptimization note: " + new_instruction
        candidate = replace(
            policy,
            name=f"{policy.name}__{name}_r{round_index}",
            instruction=base_instruction,
            **kwargs,
        )
        mutations.append((name, candidate))

    # Answer-only is analogous to an MLflow Prompt Registry run where the final
    # answer prompt is optimized but the agent program/tool-routing stays fixed.
    if scope == "answer_only":
        if not policy.require_citations:
            add("require_citations", require_citations=True, instruction="Always cite supporting enterprise document IDs.")
        if not policy.abstain_when_low_confidence:
            add("abstain_on_missing_evidence", abstain_when_low_confidence=True, instruction="If the context is insufficient or ambiguous, say so instead of guessing.")
        if not policy.use_grounding_check:
            add("grounding_check", use_grounding_check=True, require_citations=True, instruction="Before final answer, verify every factual claim is supported by retrieved context.")
        if policy.answer_style != "grounded":
            add("grounded_answer_style", answer_style="grounded", instruction="Answer with the key fact and one sentence of evidence.")
        if policy.answer_style != "synthesis":
            add("synthesis_answer_style", answer_style="synthesis", instruction="When multiple evidence snippets are relevant, synthesize them into one complete answer.")
        return mutations

    # Multi-prompt is analogous to MLflow optimizing planner + answer prompts.
    # Full-program is analogous to DSPy optimizing multiple predictors/tool-use instructions.
    if "retrieval" in signals and not policy.use_query_rewrite:
        add("enable_query_rewrite", use_query_rewrite=True, instruction="Rewrite complex questions into search-friendly terms and subqueries.")
    if "retrieval" in signals and policy.top_k_semantic < 5:
        add("increase_semantic_k", top_k_semantic=policy.top_k_semantic + 1, instruction="Retrieve a slightly broader semantic context for harder questions.")
    if ("lexical" in signals or "retrieval" in signals) and not policy.use_lexical_for_exact:
        add(
            "enable_lexical_for_exact",
            use_lexical_for_exact=True,
            top_k_lexical=max(2, policy.top_k_lexical),
            max_tool_calls=max(policy.max_tool_calls, 2),
            instruction="Use lexical search for exact names, acronyms, IDs, numbers, thresholds, and policy codes.",
        )
    if "lexical" in signals and policy.top_k_lexical < 4:
        add("increase_lexical_k", top_k_lexical=policy.top_k_lexical + 1, use_lexical_for_exact=True, max_tool_calls=max(policy.max_tool_calls, 2))
    if "kg" in signals and not policy.use_kg_for_relationships:
        add(
            "enable_kg_for_relationships",
            use_kg_for_relationships=True,
            max_tool_calls=max(policy.max_tool_calls, 3),
            instruction="Use EPKG/KG tools for ownership, relationship, entity, claim, and summary questions.",
        )
    if ("answer" in signals or "grounding" in signals) and not policy.require_citations:
        add("require_citations", require_citations=True, instruction="Always cite doc IDs that support the answer.")
    if ("grounding" in signals or "abstain" in signals) and not policy.abstain_when_low_confidence:
        add("abstain_on_missing_evidence", abstain_when_low_confidence=True, instruction="Say what is missing when context does not explicitly support the answer.")
    if "grounding" in signals and not policy.use_grounding_check:
        add("grounding_check", use_grounding_check=True, require_citations=True, instruction="Run a final grounding check against retrieved context.")
    if "answer" in signals and policy.answer_style != "synthesis":
        add("synthesis_answer_style", answer_style="synthesis", instruction="For multi-document questions, produce a complete synthesized answer.")
    if policy.context_budget_docs < 6 and ("retrieval" in signals or "answer" in signals):
        add("increase_context_budget", context_budget_docs=policy.context_budget_docs + 1, instruction="Allow one more document into context when evidence is spread across docs.")

    # Latency/cost tradeoff mutations. Only propose once the policy has enough tools.
    if "latency" in signals or round_index >= 3:
        if policy.top_k_semantic > 2:
            add("reduce_semantic_k_for_latency", top_k_semantic=policy.top_k_semantic - 1, instruction="Reduce semantic top-k if quality holds but latency is high.")
        if policy.context_budget_docs > 3:
            add("reduce_context_budget_for_latency", context_budget_docs=policy.context_budget_docs - 1, instruction="Trim context to lower token cost if citations remain sufficient.")

    return mutations


def _pareto_frontier(candidates: list[CandidateResult]) -> list[CandidateResult]:
    frontier: list[CandidateResult] = []
    for cand in candidates:
        dominated = False
        for other in candidates:
            if other is cand:
                continue
            quality_better_or_equal = other.metrics.score >= cand.metrics.score
            cost_better_or_equal = other.metrics.avg_cost_usd <= cand.metrics.avg_cost_usd
            latency_better_or_equal = other.metrics.avg_latency_ms <= cand.metrics.avg_latency_ms
            at_least_one_strict = (
                other.metrics.score > cand.metrics.score
                or other.metrics.avg_cost_usd < cand.metrics.avg_cost_usd
                or other.metrics.avg_latency_ms < cand.metrics.avg_latency_ms
            )
            if quality_better_or_equal and cost_better_or_equal and latency_better_or_equal and at_least_one_strict:
                dominated = True
                break
        if not dominated:
            frontier.append(cand)
    frontier.sort(key=lambda c: (c.utility, c.metrics.score), reverse=True)
    return frontier


def _select_bases(candidates: list[CandidateResult], cfg: OptimizerConfig, rng: random.Random) -> list[CandidateResult]:
    if cfg.candidate_selection_strategy == "best_score":
        return sorted(candidates, key=lambda c: (c.metrics.score, c.utility), reverse=True)[:3]
    if cfg.candidate_selection_strategy == "random":
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        return shuffled[:3]
    return _pareto_frontier(candidates)[:4]


@dataclass(frozen=True)
class OptimizerOutput:
    approach: str
    best: CandidateResult
    history: tuple[CandidateResult, ...]
    frontier: tuple[CandidateResult, ...]

    def history_rows(self) -> list[dict]:
        return [cand.to_history_row() for cand in self.history]


def run_mini_gepa(
    *,
    approach: str,
    start_policy: PromptPolicy,
    tools: SearchTools,
    model: ModelProfile,
    train_examples: list[EvalExample],
    cfg: OptimizerConfig,
    mutation_scope: MutationScope,
) -> OptimizerOutput:
    """Run a deterministic small-scale GEPA-style optimization loop.

    The loop is intentionally transparent:
    1. Evaluate a candidate policy on examples.
    2. Read textual feedback from the metric.
    3. Propose new prompt/program policies by mutating knobs.
    4. Keep the best/Pareto candidates under quality, cost, and latency.

    This is not a replacement for real GEPA; it is a working teaching scaffold.
    """

    rng = random.Random(cfg.random_seed)
    seen_signatures: set[str] = set()
    metric_calls = 0
    history: list[CandidateResult] = []

    root = _evaluate_policy(
        approach=approach,
        round_index=0,
        policy=start_policy,
        tools=tools,
        model=model,
        examples=train_examples,
        cfg=cfg,
        mutation="initial_policy",
        parent_id=None,
    )
    history.append(root)
    seen_signatures.add(_policy_signature(root.policy))
    metric_calls += len(train_examples)

    for round_index in range(1, cfg.max_rounds + 1):
        if metric_calls >= cfg.max_metric_calls:
            break
        bases = _select_bases(history, cfg, rng)
        new_candidates: list[CandidateResult] = []
        for base in bases:
            weak_rows = sorted(base.rows, key=lambda r: r.score)[: cfg.reflection_minibatch_size]
            signals = _feedback_signals(weak_rows)
            mutations = _mutate_policy(base.policy, signals, mutation_scope, round_index)
            for mutation_name, policy in mutations:
                if metric_calls + len(train_examples) > cfg.max_metric_calls:
                    break
                sig = _policy_signature(policy)
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)
                cand = _evaluate_policy(
                    approach=approach,
                    round_index=round_index,
                    policy=policy,
                    tools=tools,
                    model=model,
                    examples=train_examples,
                    cfg=cfg,
                    mutation=mutation_name,
                    parent_id=base.candidate_id,
                )
                new_candidates.append(cand)
                metric_calls += len(train_examples)
        if not new_candidates:
            break
        # Keep all history for observability, but frontier guides the next round.
        history.extend(new_candidates)

    frontier = _pareto_frontier(history)
    # Best defaults to highest utility; if quality is very close, prefer cheaper/lower latency.
    best = sorted(frontier, key=lambda c: (c.utility, c.metrics.score, -c.metrics.avg_latency_ms), reverse=True)[0]
    return OptimizerOutput(approach=approach, best=best, history=tuple(history), frontier=tuple(frontier))
