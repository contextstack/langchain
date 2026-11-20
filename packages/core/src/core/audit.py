"""Append-only authoritative audit log writer with idempotent UUIDv5 event IDs (spec §3.4)."""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from psycopg import AsyncConnection
from pydantic import BaseModel, Field

AUDIT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

AuditEventType = Literal[
    "authz_check",
    "approval_requested",
    "approval_decided",
    "approval_refused",
    "tool_write",
    "reconciliation",
    "run_reclaimed",
    "version_blocked",
]

AuditDecision = Literal[
    "allow",
    "deny",
    "approved",
    "rejected",
    "edited",
    "succeeded",
    "outcome_unknown",
    "failed",
    "refused",
]


def compute_audit_event_id(
    *,
    thread_id: str,
    event_type: str,
    step_or_op_key: str,
    attempt: int,
    decision: str,
) -> uuid.UUID:
    """Deterministic UUIDv5 that deduplicates replays without collapsing distinct outcomes."""
    seed = f"{thread_id}|{event_type}|{step_or_op_key}|{attempt}|{decision}"
    return uuid.uuid5(AUDIT_NAMESPACE, seed)


class AuditEvent(BaseModel, frozen=True):
    tenant_id: str
    actor_id: str
    thread_id: str
    run_id: str
    workflow_version: str
    event_type: AuditEventType
    decision: AuditDecision
    step_or_op_key: str
    attempt: int = 1
    requested_by: str | None = None
    approved_by: str | None = None
    tool_id: str | None = None
    operation_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def event_id(self) -> uuid.UUID:
        return compute_audit_event_id(
            thread_id=self.thread_id,
            event_type=self.event_type,
            step_or_op_key=self.step_or_op_key,
            attempt=self.attempt,
            decision=self.decision,
        )


async def write_audit_event(conn: AsyncConnection[Any], event: AuditEvent) -> uuid.UUID:
    """Insert an audit row idempotently (`ON CONFLICT DO NOTHING`)."""
    event_id = event.event_id
    await conn.execute(
        """
        INSERT INTO audit_log (
            event_id,
            tenant_id,
            actor_id,
            requested_by,
            approved_by,
            thread_id,
            run_id,
            workflow_version,
            event_type,
            tool_id,
            operation_id,
            decision,
            details
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (event_id) DO NOTHING
        """,
        (
            event_id,
            event.tenant_id,
            event.actor_id,
            event.requested_by,
            event.approved_by,
            event.thread_id,
            event.run_id,
            event.workflow_version,
            event.event_type,
            event.tool_id,
            event.operation_id,
            event.decision,
            json.dumps(event.details),
        ),
    )
    return event_id
