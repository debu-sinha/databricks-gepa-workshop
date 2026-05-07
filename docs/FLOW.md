# Flow

## 1. Build the tiny enterprise corpus

`src/gepa_lab/data.py` creates 15 fictional documents that mimic the types of content in the KARL evaluation plan:

- policy docs
- operational docs
- knowledge graph claims
- search tooling docs
- evaluation metric docs
- model serving notes

It also creates 18 eval questions split into train/dev/test.

## 2. Retrieve evidence

`src/gepa_lab/retrieval.py` implements three toy tools:

| Tool | Real-world analogue | Good for |
|---|---|---|
| `semantic_search` | vector / semantic search | conceptual search and paraphrases |
| `lexical_search` | keyword / BM25 search | exact names, IDs, acronyms, thresholds |
| `kg_claims` | EPKG / KG tools | entity relationships, ownership, claims |

## 3. Run agents

`src/gepa_lab/agent.py` runs a search agent with a `PromptPolicy`.

The policy controls the same kind of text/program behavior you would tune in the real engagement:

- query rewriting
- lexical tool use
- KG tool use
- citations
- abstention
- grounding checks
- answer style
- top-k
- context budget
- max tool calls

## 4. Evaluate

`src/gepa_lab/metrics.py` reports:

- Recall@5
- Recall@10
- MRR
- precision
- correctness
- completeness
- groundedness
- average latency
- p95 latency
- average tool calls
- input/output tokens
- estimated cost
- cost per successful answer

The metric also returns textual feedback, for example:

```text
retrieval miss: expected docs not in top-5
answer missing required concepts
tool policy miss: expected kg not used
grounding miss: answer has no citations
```

## 5. Optimize

`src/gepa_lab/mini_gepa.py` runs a deterministic GEPA-style loop:

```text
candidate policy
  -> eval examples
  -> textual feedback
  -> propose policy mutations
  -> evaluate candidates
  -> keep Pareto frontier
```

The DSPy-style path can mutate the full program.
The MLflow-style path defaults to final answer prompt behavior only.

## 6. Compare

`src/gepa_lab/experiment.py` writes the final scorecard:

```text
current high-cost app baseline
smaller model raw
DSPy-style GEPA optimized
MLflow-style prompt GEPA optimized
```

This mirrors the updated engagement direction: baselines are primarily the current applications using Opus/GPT-style models, not KARL out-of-the-box.
