"""LangGraph state schema for the IT/Ops Change-Request & Incident Desk (spec §5.1)."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from core.models import ChangePlan


class IncidentDeskState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    request_text: str
    release_version: str
    retrieved_runbooks: list[dict[str, Any]]
    prior_memories: list[dict[str, Any]]
    investigation_summary: str
    change_plan: ChangePlan | dict[str, Any] | None
    policy_decision: Literal["require_approval", "deny"] | None
    operation_id: str | None
    approval_decision: dict[str, Any] | None
    approved_plan_hash: str | None
    execution_result: dict[str, Any] | None
    verification_passed: bool | None
    final_status: str | None
    error_category: (
        Literal[
            "transient",
            "model_recoverable",
            "user_fixable",
            "permission_denied",
            "outcome_unknown",
            "fatal",
        ]
        | None
    )
