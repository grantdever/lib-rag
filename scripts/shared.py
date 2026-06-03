"""Shared utilities for the lib-rag pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

EMBED_DIMS = 768

OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/ali-books",
    "X-Title": "lib-rag",
}

LLM_PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-v4-flash",
        "api_key_env": "OPENROUTER_API_KEY",
        "extra_headers": OPENROUTER_HEADERS,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash-lite",
        "api_key_env": "GEMINI_API_KEY",
        "extra_headers": {},
    },
}

EMBED_MODEL = "google/gemini-embedding-001"
EMBED_MODEL_GEMINI = "gemini-embedding-001"


def is_retryable(exc) -> bool:
    """Check if an exception is retryable (rate limits, transient errors)."""
    if hasattr(exc, "status_code"):
        return exc.status_code in (429, 500, 502, 503)
    exc_str = str(exc)
    return "429" in exc_str or "500" in exc_str or "503" in exc_str or "RESOURCE_EXHAUSTED" in exc_str


def validate_api_key(provider: str) -> str:
    """Validate and return the API key for the given provider. Raises on failure."""
    env_var = "OPENROUTER_API_KEY" if provider == "openrouter" else "GEMINI_API_KEY"
    api_key = os.getenv(env_var)
    if not api_key or api_key in ("...", "sk-or-...") or api_key.startswith("sk-or-..."):
        url = "https://openrouter.ai/" if provider == "openrouter" else "https://aistudio.google.com/apikey"
        raise click.ClickException(f"Missing {env_var} in .env — get one from {url}")
    return api_key


def make_embed_fn(provider: str):
    """Return an (embed_batch, embed_query) tuple for the chosen provider.

    embed_batch(texts: list[str]) -> list[list[float]]  — for indexing
    embed_query(text: str) -> list[float]                — for querying
    """
    api_key = validate_api_key(provider)

    if provider == "gemini":
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=api_key)

        @retry(retry=retry_if_exception(is_retryable), wait=wait_exponential(multiplier=4, min=10, max=120), stop=stop_after_attempt(8))
        def embed_batch(texts: list[str]) -> list[list[float]]:
            response = client.models.embed_content(
                model=EMBED_MODEL_GEMINI,
                contents=texts,
                config=genai_types.EmbedContentConfig(outputDimensionality=EMBED_DIMS, taskType="RETRIEVAL_DOCUMENT"),
            )
            return [e.values for e in response.embeddings]

        @retry(retry=retry_if_exception(is_retryable), wait=wait_exponential(multiplier=2, min=2, max=30), stop=stop_after_attempt(3))
        def embed_query(text: str) -> list[float]:
            response = client.models.embed_content(
                model=EMBED_MODEL_GEMINI,
                contents=text,
                config=genai_types.EmbedContentConfig(outputDimensionality=EMBED_DIMS, taskType="RETRIEVAL_QUERY"),
            )
            return response.embeddings[0].values

        return embed_batch, embed_query

    elif provider == "openrouter":
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers=OPENROUTER_HEADERS,
        )

        @retry(retry=retry_if_exception(is_retryable), wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(5))
        def embed_batch(texts: list[str]) -> list[list[float]]:
            response = client.embeddings.create(model=EMBED_MODEL, input=texts, dimensions=EMBED_DIMS)
            return [e.embedding for e in response.data]

        @retry(retry=retry_if_exception(is_retryable), wait=wait_exponential(multiplier=2, min=2, max=30), stop=stop_after_attempt(3))
        def embed_query(text: str) -> list[float]:
            response = client.embeddings.create(model=EMBED_MODEL, input=[text], dimensions=EMBED_DIMS)
            return response.data[0].embedding

        return embed_batch, embed_query

    else:
        raise click.ClickException(f"Unknown embedding provider: {provider}")


def get_llm_client(provider: str):
    """Return (OpenAI_client, model_name) for LLM calls (map generation)."""
    from openai import OpenAI

    cfg = LLM_PROVIDERS[provider]
    api_key = validate_api_key(provider)
    client = OpenAI(
        base_url=cfg["base_url"],
        api_key=api_key,
        default_headers=cfg["extra_headers"],
    )
    return client, cfg["model"]
