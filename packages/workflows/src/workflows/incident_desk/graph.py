"""IT/Ops Change-Request & Incident Desk LangGraph workflow (spec §5.2)."""

from __future__ import annotations

import json
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage

from core.models import ChangePlan, RunbookCitation, RunContext
from core.pii import redact_text
from core.tool_contract import TOOL_CONTRACTS, cap_tool_output
from workflows.incident_desk.version import INVESTIGATE_SYSTEM_PROMPT, RELEASE_VERSION
from workflows.lifecycle import (
    DomainHooks,
    StudioRunContext,
    WorkflowDeps,
    build_governed_workflow_graph,
)
from workflows.lifecycle import (
    coerce_plan as _coerce_plan,
)
from workflows.lifecycle import (
    maybe_inject_fault as _maybe_inject_fault,
)
from workflows.tool_executor import ToolExecutor, WriteSpec

__all__ = [
    "StudioRunContext",
    "WorkflowDeps",
    "_coerce_plan",
    "_maybe_inject_fault",
    "build_incident_desk_graph",
    "build_studio_graph",
]


def _deterministic_mock_plan(
    request_text: str,
    runbooks: list[dict[str, Any]],
) -> ChangePlan:
    """Deterministic ChangePlan synthesizer for MODEL_MODE=mock scenarios and drills."""
    lower = request_text.lower()
    citations = [
        RunbookCitation(doc_id=rb["doc_id"], title=rb["title"]) for rb in runbooks[:3]
    ]

    if "no action" in lower or "status only" in lower:
        return ChangePlan(
            resource_type="none",
            target_resource="",
            action="none",
            risk_level="low",
            summary="Service inspected; no infrastructure change required.",
            citations=citations,
        )

    if "drop_database" in lower or "delete_cluster" in lower:
        bad_action = "drop_database" if "drop_database" in lower else "delete_cluster"
        return ChangePlan(
            resource_type="service",
            target_resource="checkout-api",
            action=bad_action,  # type: ignore[arg-type]
            risk_level="critical",
            summary=f"Requested prohibited action {bad_action}.",
            citations=citations,
        )

    if "cross-tenant" in lower or "billing-db" in lower or "cache" in lower:
        return ChangePlan(
            resource_type="service",
            target_resource="tenant-beta/billing-db",
            action="restart",
            risk_level="high",
            summary="Restart tenant-beta/billing-db per diagnostic note.",
            citations=citations,
        )

    if "payments-worker" in lower or "latency" in lower:
        return ChangePlan(
            resource_type="service",
            target_resource="payments-worker",
            action="restart",
            rollback_plan="Verify payments-worker health after restart.",
            risk_level="high",
            summary="Restart payments-worker per latency troubleshooting note.",
            citations=citations,
        )

    return ChangePlan(
        resource_type="service",
        target_resource="checkout-api",
        action="rollback",
        target_version="v2.4.0",
        rollback_plan="Re-deploy v2.4.1 if v2.4.0 exhibits regressions.",
        risk_level="medium",
        summary="Roll back checkout-api from degraded v2.4.1 to v2.4.0.",
        citations=citations,
        ticket_id="INC-1001",
    )


def _build_read_tools(executor: ToolExecutor) -> list[Any]:
    @tool
    async def get_service_status(service_name: str, runtime: ToolRuntime[RunContext]) -> str:
        """Inspect current version and health of a service in the caller's tenant."""
        ctx = runtime.context
        try:
            res = await executor.invoke_read(
                ctx,
                tool_id="get_service_status",
                mcp_url=executor.settings.mcp_infra_url,
                mcp_args={"service_name": service_name},
                direct_fn=lambda: executor.domain.get_service_status(
                    tenant_id=ctx.tenant_id, service_name=service_name
                ),
            )
            return cap_tool_output(
                json.dumps(res), TOOL_CONTRACTS["get_service_status"].max_output_chars
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @tool
    async def list_recent_deploys(service_name: str, runtime: ToolRuntime[RunContext]) -> str:
        """List recent deploys for a service in the caller's tenant."""
        ctx = runtime.context
        try:
            res = await executor.invoke_read(
                ctx,
                tool_id="list_recent_deploys",
                mcp_url=executor.settings.mcp_infra_url,
                mcp_args={"service_name": service_name},
                direct_fn=lambda: executor.domain.list_recent_deploys(
                    tenant_id=ctx.tenant_id, service_name=service_name
                ),
            )
            return cap_tool_output(
                json.dumps(res), TOOL_CONTRACTS["list_recent_deploys"].max_output_chars
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @tool
    async def get_ticket(ticket_id: str, runtime: ToolRuntime[RunContext]) -> str:
        """Inspect an incident ticket belonging to the caller's tenant."""
        ctx = runtime.context
        try:
            res = await executor.invoke_read(
                ctx,
                tool_id="get_ticket",
                mcp_url=executor.settings.mcp_ticketing_url,
                mcp_args={"ticket_id": ticket_id},
                direct_fn=lambda: executor.domain.get_ticket(
                    tenant_id=ctx.tenant_id, ticket_id=ticket_id
                ),
            )
            return cap_tool_output(
                json.dumps(res), TOOL_CONTRACTS["get_ticket"].max_output_chars
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    return [get_service_status, list_recent_deploys, get_ticket]


def _mock_investigate_turns(
    req_text: str, ctx: RunContext, runbooks: list[dict[str, Any]]
) -> list[AIMessage]:
    lower = req_text.lower()
    target_svc = "payments-worker" if "payments-worker" in lower else "checkout-api"
    tool_calls: list[dict[str, Any]] = [
        {
            "id": "call_investigate_1",
            "name": "get_service_status",
            "args": {"service_name": target_svc},
        }
    ]
    if "deploy" in lower:
        tool_calls.append(
            {
                "id": "call_investigate_2",
                "name": "list_recent_deploys",
                "args": {"service_name": target_svc},
            }
        )
    if "inc-" in lower or "ticket" in lower:
        tool_calls.append(
            {
                "id": "call_investigate_3",
                "name": "get_ticket",
                "args": {"ticket_id": "INC-1001"},
            }
        )
    return [
        AIMessage(content="", tool_calls=tool_calls),
        AIMessage(
            content=(
                f"Investigated {target_svc} for tenant {ctx.tenant_id} "
                f"using {len(runbooks)} runbooks."
            )
        ),
    ]


def _resolve_write_spec(
    executor: ToolExecutor,
    ctx: RunContext,
    plan: ChangePlan,
    operation_id: str,
) -> WriteSpec:
    if plan.resource_type == "ticket" or plan.action == "update_ticket":
        ticket_id = plan.ticket_id or plan.target_resource
        return WriteSpec(
            tool_id="update_ticket",
            mcp_url=executor.settings.mcp_ticketing_url,
            mcp_args={
                "ticket_id": ticket_id,
                "status": "resolved",
                "note": plan.summary,
                "idempotency_key": operation_id,
            },
            direct_fn=lambda: executor.domain.update_ticket(
                tenant_id=ctx.tenant_id,
                ticket_id=ticket_id,
                status="resolved",
                note=plan.summary,
                operation_id=operation_id,
            ),
            reconcile_mcp_url=executor.settings.mcp_infra_url,
        )

    return WriteSpec(
        tool_id="execute_service_change",
        mcp_url=executor.settings.mcp_infra_url,
        mcp_args={
            "service_name": plan.target_resource,
            "action": plan.action,
            "target_version": plan.target_version,
            "idempotency_key": operation_id,
        },
        direct_fn=lambda: executor.domain.execute_service_change(
            tenant_id=ctx.tenant_id,
            service_name=plan.target_resource,
            action=plan.action,
            target_version=plan.target_version,
            operation_id=operation_id,
        ),
        reconcile_mcp_url=executor.settings.mcp_infra_url,
    )


async def _verify_end_state(
    executor: ToolExecutor,
    ctx: RunContext,
    plan: ChangePlan,
    operation_id: str,
    exec_res: dict[str, Any],
) -> bool:
    if plan.resource_type == "ticket" or plan.action == "update_ticket":
        ticket_id = plan.ticket_id or plan.target_resource
        ticket = await executor.invoke_read(
            ctx,
            tool_id="get_ticket",
            mcp_url=executor.settings.mcp_ticketing_url,
            mcp_args={"ticket_id": ticket_id},
            direct_fn=lambda: executor.domain.get_ticket(
                tenant_id=ctx.tenant_id, ticket_id=ticket_id
            ),
        )
        return ticket["status"] == "resolved"

    svc = await executor.invoke_read(
        ctx,
        tool_id="get_service_status",
        mcp_url=executor.settings.mcp_infra_url,
        mcp_args={"service_name": plan.target_resource},
        direct_fn=lambda: executor.domain.get_service_status(
            tenant_id=ctx.tenant_id, service_name=plan.target_resource
        ),
    )
    version_ok = svc["current_version"] == plan.target_version if plan.target_version else True
    return svc["status"] == "healthy" and version_ok


def _build_memory_payload(
    ctx: RunContext, plan: ChangePlan, operation_id: str
) -> dict[str, Any]:
    return {
        "thread_id": ctx.thread_id,
        "operation_id": operation_id,
        "target": plan.target_resource,
        "action": plan.action,
        "summary": redact_text(plan.summary),
    }


INCIDENT_DESK_HOOKS = DomainHooks(
    workflow_name="incident_desk",
    release_version=RELEASE_VERSION,
    investigate_system_prompt=INVESTIGATE_SYSTEM_PROMPT,
    build_read_tools=_build_read_tools,
    build_mock_investigate_turns=_mock_investigate_turns,
    deterministic_mock_plan=_deterministic_mock_plan,
    resolve_write_spec=_resolve_write_spec,
    verify_domain_end_state=_verify_end_state,
    build_memory_payload=_build_memory_payload,
)


def build_incident_desk_graph(
    *,
    deps: WorkflowDeps | None = None,
    checkpointer: Any = None,
    store: Any = None,
    context_schema: type[RunContext] = RunContext,
    default_context: RunContext | None = None,
):
    """Compile the full IT/Ops Desk StateGraph with context_schema=RunContext."""
    return build_governed_workflow_graph(
        INCIDENT_DESK_HOOKS,
        deps=deps,
        checkpointer=checkpointer,
        store=store,
        context_schema=context_schema,
        default_context=default_context,
    )


def build_studio_graph():
    """Compiled WITHOUT a checkpointer so `langgraph dev` supplies local persistence (spec §5.7)."""
    return build_incident_desk_graph(
        deps=WorkflowDeps(use_postgres=False),
        checkpointer=None,
        context_schema=StudioRunContext,
        default_context=StudioRunContext(),
    )
