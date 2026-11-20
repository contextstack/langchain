"""Deterministic mock LLM and 1536-dim mock embeddings for $0 CI and drills."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

EMBEDDING_DIM = 1536


class DeterministicMockEmbeddings(Embeddings):
    """Hash-seeded deterministic 1536-dimension embeddings with zero network calls."""

    dimensions: int = EMBEDDING_DIM

    def _embed(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dimensions:
            digest = hashlib.sha256(f"{text}:{counter}".encode()).digest()
            for idx in range(0, len(digest), 2):
                if len(values) >= self.dimensions:
                    break
                raw = int.from_bytes(digest[idx : idx + 2], "big", signed=False)
                values.append((raw / 32767.5) - 1.0)
            counter += 1
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class ScriptedMockChatModel(BaseChatModel):
    """Deterministic BaseChatModel returning pre-scripted AIMessage turns."""

    responses: list[AIMessage] = Field(default_factory=list)
    cursor_state: dict[str, int] = Field(default_factory=lambda: {"index": 0})
    bound_tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-mock-chat-model"

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ScriptedMockChatModel:
        return self.model_copy(update={"bound_tools": list(tools)})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            msg = AIMessage(content="mock-ok")
        else:
            idx = min(self.cursor_state["index"], len(self.responses) - 1)
            msg = self.responses[idx]
            self.cursor_state["index"] += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])
