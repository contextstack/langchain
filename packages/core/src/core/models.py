"""Core Pydantic domain contracts shared across all packages and services."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field


class CallerIdentity(BaseModel, frozen=True):
    """Authenticated caller identity extracted from an API JWT."""

    tenant_id: str
    actor_id: str
    roles: frozenset[str]


class RunContext(BaseModel, frozen=True):
    """Trusted runtime context passed via LangGraph context_schema (spec §4.1).

    The requester's identity is fixed on the thread and never changes on resume.
    """

    tenant_id: str
    requester_id: str
    requester_roles: frozenset[str]
    thread_id: str
    run_id: str
    request_id: str


class RunbookCitation(BaseModel, frozen=True):
    doc_id: str
    title: str


class ChangePlan(BaseModel, frozen=True):
    """Structured change proposal produced by the workflow and shown to approvers."""

    resource_type: Literal["service", "ticket", "none"] = "service"
    target_resource: str = ""
    action: Literal[
        "none",
        "rollback",
        "restart",
        "scale_up",
        "update_ticket",
        "delete_cluster",
        "drop_database",
    ] = "none"
    target_version: str | None = None
    rollback_plan: str = ""
    risk_level: Literal["low", "medium", "high", "critical"] = "medium"
    summary: str = ""
    citations: list[RunbookCitation] = Field(default_factory=list)
    ticket_id: str | None = None

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def plan_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class ToolContract(BaseModel, frozen=True):
    """Enterprise contract attached to every MCP/domain tool (spec §6)."""

    tool_id: str
    version: str = "1.0.0"
    read_write_class: Literal["read", "write", "destructive"]
    timeout_s: float = 10.0
    max_output_chars: int = 8000
    idempotency_required: bool = False
    audit_required: bool = False
