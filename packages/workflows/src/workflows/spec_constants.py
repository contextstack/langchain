"""Shared executable graph topology & planner prompt constants for release fingerprinting."""

from __future__ import annotations

GOVERNED_GRAPH_NODES: tuple[str, ...] = (
    "retrieve_context",
    "investigate",
    "propose_plan",
    "finalize_no_action",
    "evaluate_policy",
    "finalize_denied",
    "record_approval_request",
    "await_approval",
    "finalize_rejected",
    "revalidate_edit",
    "execute_change",
    "verify_end_state",
)

GOVERNED_GRAPH_EDGES: tuple[tuple[str, str], ...] = (
    ("START", "retrieve_context"),
    ("retrieve_context", "investigate"),
    ("investigate", "propose_plan"),
    ("propose_plan", "finalize_no_action"),
    ("propose_plan", "evaluate_policy"),
    ("finalize_no_action", "END"),
    ("evaluate_policy", "finalize_denied"),
    ("evaluate_policy", "record_approval_request"),
    ("finalize_denied", "END"),
    ("record_approval_request", "await_approval"),
    ("await_approval", "execute_change"),
    ("await_approval", "revalidate_edit"),
    ("await_approval", "finalize_rejected"),
    ("finalize_rejected", "END"),
    ("revalidate_edit", "finalize_denied"),
    ("revalidate_edit", "execute_change"),
    ("execute_change", "verify_end_state"),
    ("execute_change", "END"),
    ("verify_end_state", "END"),
)

PROPOSE_PLAN_PROMPT_TEMPLATE = (
    "Request: {request}\n"
    "Investigation: {investigation}\n"
    "Runbooks: {runbooks}"
)
