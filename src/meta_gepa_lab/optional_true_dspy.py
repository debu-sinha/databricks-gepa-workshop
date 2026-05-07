"""Optional real DSPy GEPA sketch.

The default lab is deterministic and fully runnable. This file shows how you would
start converting the toy program into real DSPy once you have a Databricks model
endpoint. It deliberately fails gracefully if DSPy or a model endpoint is missing.
"""

from __future__ import annotations

import os


def print_real_dspy_gepa_template() -> None:
    print(
        r'''
# Optional true DSPy GEPA direction. Run only after installing dspy and configuring an endpoint.
import dspy

# Example; replace with your Databricks serving endpoint configuration.
# Depending on your workspace, you may use DatabricksOpenAI or DSPy's OpenAI-compatible LM.
lm = dspy.LM(
    model="openai/<your-databricks-endpoint-name>",
    api_base=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints",
    api_key=os.environ["DATABRICKS_TOKEN"],
)
dspy.configure(lm=lm)

class SearchPlan(dspy.Signature):
    """Plan enterprise search tool calls for a user question."""
    question: str = dspy.InputField()
    plan: str = dspy.OutputField()

class Answer(dspy.Signature):
    """Answer using retrieved context and cite document IDs."""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()

class EnterpriseSearchProgram(dspy.Module):
    def __init__(self, tools):
        self.plan = dspy.ChainOfThought(SearchPlan)
        self.answer = dspy.ChainOfThought(Answer)
        self.tools = tools

    def forward(self, question: str):
        plan = self.plan(question=question).plan
        # Call semantic / lexical / KG tools using the plan, then build context.
        context = "..."
        return self.answer(question=question, context=context)

# Your metric should return a scalar score and textual feedback. GEPA benefits from rich feedback.
def gepa_metric(example, prediction, trace=None):
    return score, feedback

optimizer = dspy.GEPA(
    metric=gepa_metric,
    auto="light",
    max_metric_calls=50,
    reflection_lm=lm,
    candidate_selection_strategy="pareto",
    track_stats=True,
)
compiled = optimizer.compile(EnterpriseSearchProgram(tools), trainset=trainset, valset=devset)
        '''
    )


def run_real_dspy_gepa_if_available() -> None:
    try:
        import dspy  # noqa: F401  # type: ignore
    except Exception as e:
        print(f"DSPy is not installed; skipping real DSPy GEPA. Reason: {e}")
        print_real_dspy_gepa_template()
        return
    if not os.getenv("DATABRICKS_MODEL_ENDPOINT"):
        print("DATABRICKS_MODEL_ENDPOINT is not set; skipping real DSPy GEPA.")
        print_real_dspy_gepa_template()
        return
    print("DSPy is installed and an endpoint is configured. Use the template below to wire the toy tools into DSPy.")
    print_real_dspy_gepa_template()


if __name__ == "__main__":
    run_real_dspy_gepa_if_available()
