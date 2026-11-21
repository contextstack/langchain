"""Model and embeddings factory supporting LiteLLM Gateway and deterministic CI mocks (spec §7)."""

from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import OpenAIEmbeddings

from core.config import Settings, get_settings
from gateway.mocks import DeterministicMockEmbeddings, ScriptedMockChatModel


def _tenant_litellm_key(tenant_id: str, settings: Settings) -> str:
    env_key = f"LITELLM_KEY_{tenant_id.upper().replace('-', '_')}"
    return os.getenv(env_key) or settings.litellm_master_key


def get_chat_model(
    *,
    alias: str = "ops-primary",
    tenant_id: str = "tenant-alpha",
    settings: Settings | None = None,
    mock_responses: list[AIMessage] | None = None,
) -> Any:
    """Return a ChatModel routed through LiteLLM, or ScriptedMockChatModel in mock mode."""
    cfg = settings or get_settings()
    if cfg.model_mode == "mock":
        return ScriptedMockChatModel(responses=mock_responses or [])

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=5.0,
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    )
    # Use native max_retries on the ChatModel rather than .with_retry() (RunnableRetry),
    # so .bind_tools() and .with_structured_output() remain available.
    model: BaseChatModel = init_chat_model(
        model=alias,
        model_provider="openai",
        base_url=cfg.litellm_url,
        api_key=_tenant_litellm_key(tenant_id, cfg),
        rate_limiter=rate_limiter,
        temperature=0.0,
        max_retries=3,
    )
    return model


def get_embeddings(
    *,
    tenant_id: str | None = None,
    settings: Settings | None = None,
) -> Embeddings:
    """Return 1536-dim embeddings (LiteLLM ops-embed in gateway mode, deterministic mock in CI)."""
    cfg = settings or get_settings()
    if cfg.model_mode == "mock":
        return DeterministicMockEmbeddings()
    api_key = _tenant_litellm_key(tenant_id, cfg) if tenant_id else cfg.litellm_master_key
    return OpenAIEmbeddings(
        model="ops-embed",
        base_url=cfg.litellm_url,
        api_key=api_key,
    )
