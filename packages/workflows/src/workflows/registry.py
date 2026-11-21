"""Workflow registry mapping `app_threads.workflow_name` -> LangGraph factory."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.config import Settings
from workflows.incident_desk.graph import build_incident_desk_graph
from workflows.incident_desk.version import (
    RELEASE_VERSION as INCIDENT_RELEASE_VERSION,
)
from workflows.incident_desk.version import (
    get_checkpoint_compat as get_incident_compat,
)
from workflows.refund_ops.graph import build_refund_ops_graph
from workflows.refund_ops.version import (
    RELEASE_VERSION as REFUND_RELEASE_VERSION,
)
from workflows.refund_ops.version import (
    get_checkpoint_compat as get_refund_compat,
)


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    build_graph: Callable[..., Any]
    get_checkpoint_compat: Callable[[], int]
    release_version: str
    reconcile_mcp_url: Callable[[Settings], str]


WORKFLOW_REGISTRY: dict[str, WorkflowSpec] = {
    "incident_desk": WorkflowSpec(
        name="incident_desk",
        build_graph=build_incident_desk_graph,
        get_checkpoint_compat=get_incident_compat,
        release_version=INCIDENT_RELEASE_VERSION,
        reconcile_mcp_url=lambda s: s.mcp_infra_url,
    ),
    "refund_ops": WorkflowSpec(
        name="refund_ops",
        build_graph=build_refund_ops_graph,
        get_checkpoint_compat=get_refund_compat,
        release_version=REFUND_RELEASE_VERSION,
        reconcile_mcp_url=lambda s: s.mcp_billing_url,
    ),
}


def get_workflow_spec(workflow_name: str) -> WorkflowSpec:
    spec = WORKFLOW_REGISTRY.get(workflow_name)
    if spec is None:
        known = ", ".join(sorted(WORKFLOW_REGISTRY))
        raise KeyError(f"Unknown workflow_name '{workflow_name}' (registered: {known})")
    return spec
