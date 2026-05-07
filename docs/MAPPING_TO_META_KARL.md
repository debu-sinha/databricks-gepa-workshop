# Mapping to the Meta / KARL evaluation plan

The KARL executive summary is RL/post-training oriented. This lab translates it into a GEPA-first prototype.

## What stays the same

The scorecard keeps the same evaluation categories:

- retrieval quality
- generation quality
- operational efficiency

The metrics mirror the KARL plan:

- Recall@5 / Recall@10
- MRR
- precision
- correctness
- completeness
- groundedness
- latency
- average tool calls

This lab also adds cost-per-query and cost-per-success, because the updated engagement context is heavily cost/latency driven.

## What changes for GEPA

KARL setup language is about post-training. GEPA does not update model weights. For the prototype, translate the setups as follows:

| KARL/RL framing | GEPA prototype framing |
|---|---|
| KARL post-trained with Wiki-p60 | smaller model with GEPA-optimized prompts/programs over train examples |
| semantic retriever | semantic tool in the agent |
| semantic + lexical retriever | GEPA-optimized tool policy can choose semantic and lexical |
| semantic + lexical + KG | GEPA-optimized multi-tool policy can choose semantic, lexical, and KG/EPKG |
| post-training eval report | code-backed prototype + scorecard |

## Updated baseline interpretation

The latest thread says not to compare against KARL out-of-the-box for the GEPA kickoff. The primary baselines are the current applications using Opus/GPT-style large models, because those power retrieval/generation today.

So this lab compares:

1. current high-cost app baseline
2. smaller raw model
3. smaller model with DSPy-style full-program GEPA
4. smaller model with MLflow-style prompt GEPA

## Real kickoff questions this lab prepares you for

- What exactly are the current baseline quality/cost/latency numbers?
- Does the eval harness return scalar scores only or textual feedback/traces?
- Which examples are train/dev/test?
- What tools are available: semantic, lexical, EPKG/KG?
- Are retrieval and generation failures attributed separately?
- What model candidates should we try first?
- What can be logged in MLflow?
- What serving path is feasible if the prototype works?
