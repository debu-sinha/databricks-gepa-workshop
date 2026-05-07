"""Optional real MLflow Prompt Optimization sketch.

The deterministic lab logs to MLflow if available. This file shows the shape of a
real mlflow.genai.optimize_prompts call once you have MLflow >= 3.5 and a model
endpoint. It is a template, not part of the no-API smoke test.
"""

from __future__ import annotations


def print_real_mlflow_gepa_template() -> None:
    print(
        r'''
# Optional true MLflow GEPA direction. Run in Databricks with MLflow >= 3.5.
import mlflow
from mlflow.genai.optimize.optimizers import GepaPromptOptimizer

# 1. Register one or more prompts.
answer_prompt = mlflow.genai.register_prompt(
    name="<catalog>.<schema>.enterprise_search_answer_prompt",
    template="""
Answer the question using only the context below.
Question: {{question}}
Context: {{context}}
Return a concise answer with document ID citations.
""",
)

# 2. Your predict_fn loads the prompt and calls your Databricks endpoint / agent code.
def predict_fn(question: str, context: str) -> str:
    prompt = mlflow.genai.load_prompt(answer_prompt.uri)
    final_prompt = prompt.format(question=question, context=context)
    # call DatabricksOpenAI / custom serving endpoint here
    return call_model(final_prompt)

# 3. Train data should include examples from the optimization split.
train_data = [
    {"inputs": {"question": ex.question, "context": retrieve_context(ex.question)}, "expectations": {"expected_answer": ex.expected_answer}}
    for ex in train_examples
]

# 4. Define scorers. In the real engagement, use their eval harness / LLM judge.
def score_search_answer(inputs, outputs, expectations):
    # return a numeric score and, ideally, textual rationale/feedback
    return {"score": 0.0, "rationale": "..."}

optimizer = GepaPromptOptimizer(
    reflection_model="databricks:/<reflection-endpoint>",
    max_metric_calls=50,
    gepa_kwargs={
        "candidate_selection_strategy": "pareto",
        "reflection_minibatch_size": 4,
    },
)

result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=train_data,
    prompt_uris=[answer_prompt.uri],
    optimizer=optimizer,
    scorers=[score_search_answer],
)

print(result.optimized_prompts[0].template)
        '''
    )


if __name__ == "__main__":
    print_real_mlflow_gepa_template()
