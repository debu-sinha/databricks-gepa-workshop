from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ModelProfile:
    """Small cost/latency model used for the local simulation.

    These are not real vendor prices. They exist so the scorecard has the same
    shape as the Meta/KARL evaluation: quality + operational efficiency.
    """

    name: str
    size_label: str
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float
    base_latency_ms: float
    latency_per_tool_call_ms: float
    latency_per_1k_tokens_ms: float


CURRENT_LARGE_MODEL = ModelProfile(
    name="current_app_large_model_simulated_opus_or_gpt55",
    size_label="large/SOTA baseline",
    input_cost_per_1k_tokens=0.020,
    output_cost_per_1k_tokens=0.080,
    base_latency_ms=1400.0,
    latency_per_tool_call_ms=180.0,
    latency_per_1k_tokens_ms=220.0,
)

SMALL_MODEL = ModelProfile(
    name="candidate_smaller_model_simulated_qwen_or_meta_model",
    size_label="smaller candidate",
    input_cost_per_1k_tokens=0.002,
    output_cost_per_1k_tokens=0.008,
    base_latency_ms=450.0,
    latency_per_tool_call_ms=95.0,
    latency_per_1k_tokens_ms=80.0,
)


@dataclass(frozen=True)
class PromptPolicy:
    """The text/program knobs that this small lab lets GEPA-style search optimize.

    In the real engagement these would map to prompts and instructions, for example:
    - search planning prompt
    - query rewrite prompt
    - semantic vs lexical tool-selection prompt
    - KG/EPKG tool-selection prompt
    - final answer / citation / abstention prompt
    """

    name: str
    instruction: str
    use_query_rewrite: bool = False
    use_lexical_for_exact: bool = False
    use_kg_for_relationships: bool = False
    require_citations: bool = False
    abstain_when_low_confidence: bool = False
    use_grounding_check: bool = False
    top_k_semantic: int = 2
    top_k_lexical: int = 0
    context_budget_docs: int = 3
    max_tool_calls: int = 2
    answer_style: Literal["terse", "grounded", "synthesis"] = "terse"

    def to_dict(self) -> dict:
        return asdict(self)


CURRENT_APP_BASELINE_POLICY = PromptPolicy(
    name="current_high_cost_app_baseline",
    instruction=(
        "Strong current application baseline. Use semantic, lexical, and KG tools; "
        "synthesize across documents; cite supporting docs; abstain when evidence is missing."
    ),
    use_query_rewrite=True,
    use_lexical_for_exact=True,
    use_kg_for_relationships=True,
    require_citations=True,
    abstain_when_low_confidence=True,
    use_grounding_check=True,
    top_k_semantic=4,
    top_k_lexical=3,
    context_budget_docs=6,
    max_tool_calls=4,
    answer_style="synthesis",
)

SMALL_RAW_POLICY = PromptPolicy(
    name="smaller_model_raw_prompt",
    instruction=(
        "Naive smaller-model prompt. Use semantic search and answer briefly. "
        "Do not overthink tool choice."
    ),
    use_query_rewrite=False,
    use_lexical_for_exact=False,
    use_kg_for_relationships=False,
    require_citations=False,
    abstain_when_low_confidence=False,
    use_grounding_check=False,
    top_k_semantic=2,
    top_k_lexical=0,
    context_budget_docs=2,
    max_tool_calls=1,
    answer_style="terse",
)

MLFLOW_STYLE_START_POLICY = PromptPolicy(
    name="mlflow_style_answer_prompt_v1",
    instruction=(
        "Start with a simple RAG answer prompt. MLflow-style run will mostly optimize "
        "answer behavior: citations, abstention, groundedness, and answer style."
    ),
    use_query_rewrite=False,
    use_lexical_for_exact=False,
    use_kg_for_relationships=False,
    require_citations=False,
    abstain_when_low_confidence=False,
    use_grounding_check=False,
    top_k_semantic=2,
    top_k_lexical=0,
    context_budget_docs=3,
    max_tool_calls=1,
    answer_style="terse",
)


@dataclass(frozen=True)
class OptimizerConfig:
    """GEPA-style knobs exposed in the runnable local lab."""

    max_rounds: int = 5
    max_metric_calls: int = 240
    reflection_minibatch_size: int = 4
    candidate_selection_strategy: Literal["pareto", "best_score", "random"] = "pareto"
    frontier_quality_weight: float = 0.72
    frontier_cost_weight: float = 0.18
    frontier_latency_weight: float = 0.10
    min_delta_to_keep_candidate: float = 0.005
    random_seed: int = 7
    track_best_outputs: bool = True
    track_stats: bool = True

    # Which knobs each approach is allowed to mutate.
    dspy_style_mutation_scope: Literal["full_program", "answer_only"] = "full_program"
    mlflow_style_mutation_scope: Literal["answer_only", "multi_prompt"] = "answer_only"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: str = "outputs"
    use_mlflow_tracking: bool = True
    mlflow_experiment_name: str = "/Shared/meta_gepa_smallscale_lab"
    optimize_on_split: Literal["train", "dev"] = "train"
    test_split_name: str = "test"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["optimizer"] = self.optimizer.to_dict()
        return result
