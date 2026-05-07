# Knobs

## Experiment knobs

| Knob | Where | Simple meaning |
|---|---|---|
| `max_rounds` | `OptimizerConfig` | How many reflect/propose rounds to run |
| `max_metric_calls` | `OptimizerConfig` | Total eval budget; lower means faster, higher means more search |
| `reflection_minibatch_size` | `OptimizerConfig` | Number of weak examples used to generate feedback each round |
| `candidate_selection_strategy` | `OptimizerConfig` | How candidates are selected: `pareto`, `best_score`, or `random` |
| `frontier_quality_weight` | `OptimizerConfig` | How much the optimizer values quality |
| `frontier_cost_weight` | `OptimizerConfig` | How much the optimizer penalizes cost |
| `frontier_latency_weight` | `OptimizerConfig` | How much the optimizer penalizes latency |
| `dspy_style_mutation_scope` | `OptimizerConfig` | `full_program` or `answer_only` |
| `mlflow_style_mutation_scope` | `OptimizerConfig` | `answer_only` or `multi_prompt` |

## Program/prompt knobs

These live in `PromptPolicy`.

| Knob | What it controls | Why it matters |
|---|---|---|
| `use_query_rewrite` | Adds search-friendly terms/subqueries | Helps multi-hop and paraphrased queries |
| `use_lexical_for_exact` | Uses lexical search for exact terms | Helps acronyms, IDs, thresholds, names |
| `use_kg_for_relationships` | Uses KG/EPKG claims | Helps ownership/entity/relationship questions |
| `require_citations` | Adds document ID citations | Improves groundedness and auditability |
| `abstain_when_low_confidence` | Says missing/ambiguous instead of guessing | Helps absence and ambiguity questions |
| `use_grounding_check` | Verifies support before answering | Reduces unsupported answers |
| `top_k_semantic` | Number of semantic docs retrieved | Higher may improve recall but increases tokens/latency |
| `top_k_lexical` | Number of lexical docs retrieved | Higher may improve exact-match recall but increases tokens/latency |
| `context_budget_docs` | Max docs passed to answer step | Higher improves evidence coverage but increases cost |
| `max_tool_calls` | Max tools the agent can call | Higher improves complex tasks but increases latency |
| `answer_style` | `terse`, `grounded`, or `synthesis` | Controls completeness vs brevity |

## Recommended learning sequence

1. Run default settings.
2. Compare `dspy_style_history.csv` vs `mlflow_style_history.csv`.
3. Change MLflow-style scope from `answer_only` to `multi_prompt`.
4. Reduce `max_metric_calls` to see what happens under tight budget.
5. Increase `frontier_latency_weight` to force lower-latency candidates.
6. Turn off `use_mlflow_tracking` if you are outside Databricks.

## Real GEPA mapping

| This lab | Real DSPy GEPA / MLflow GEPA |
|---|---|
| deterministic feedback parser | reflection LM reads traces and textual feedback |
| policy mutations | prompt/program candidate proposals |
| Pareto frontier | GEPA candidate selection / Pareto-style optimization |
| synthetic scorer | Meta eval harness / LLM judges |
| simulated costs | endpoint/token/GPU serving costs |
