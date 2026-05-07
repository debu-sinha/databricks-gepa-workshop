from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DatabricksLLMConfig:
    endpoint: str
    temperature: float = 0.0
    max_tokens: int = 512


def call_databricks_chat(prompt: str, cfg: DatabricksLLMConfig) -> str:
    """Optional Databricks Foundation Model / custom serving endpoint call.

    This is not used by the default deterministic lab. Use it after you have a
    Databricks workspace, endpoint name, and permissions. The endpoint is read
    from DATABRICKS_MODEL_ENDPOINT unless passed explicitly by your code.
    """

    try:
        from databricks_openai import DatabricksOpenAI  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "databricks-openai is not installed. Run: %pip install databricks-openai openai"
        ) from e

    if not os.getenv("DATABRICKS_HOST"):
        raise RuntimeError("DATABRICKS_HOST is not set. Run this in Databricks or set the env var.")

    client = DatabricksOpenAI()
    response = client.chat.completions.create(
        model=cfg.endpoint,
        messages=[{"role": "user", "content": prompt}],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    return response.choices[0].message.content or ""
