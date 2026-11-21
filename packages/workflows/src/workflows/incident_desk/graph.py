"""IT/Ops Change-Request & Incident Desk LangGraph StateGraph (spec §5.2)."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import psycopg
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from core.audit import AuditEvent, write_audit_event
from core.auth import mint_mcp_token
from core.config import Settings, get_settings
from core.db import set_tenant_context
from core.models import ChangePlan, RunbookCitation, RunContext
from core.pii import API_KEY_PATTERN, EMAIL_PATTERN, PHONE_PATTERN, redact_obj, redact_text
from core.policy import PolicyDecision, PolicyEngine
from core.tool_contract import TOOL_CONTRACTS, cap_tool_output
from gateway.factory import get_chat_model, get_embeddings
from knowledge.fixtures import SEED_RUNBOOKS
from knowledge.retriever import retrieve_runbooks
from mcp_servers.domain_store import InMemoryDomainStore, PostgresDomainStore
from workflows.incident_desk.state import IncidentDeskState
from workflows.incident_desk.version import RELEASE_VERSION
from workflows.mcp_client import call_mcp_tool


def _maybe_inject_fault(point: str) -> None:
    """If FAULT_INJECT matches `point`, crash immediately (same as SIGKILL)."""
    if os.getenv("FAULT_INJECT") == point:
        os._exit(137)


def _coerce_plan(raw: ChangePlan | dict[str, Any] | None) -> ChangePlan | None:
    if raw is None:
        return None
    if isinstance(raw, ChangePlan):
        return raw
    return ChangePlan.model_validate(raw)


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
        # Simulates a model fooled by rb-alpha-poison-cross-tenant
        return ChangePlan(
            resource_type="service",
            target_resource="tenant-beta/billing-db",
            action="restart",
            risk_level="high",
            summary="Restart tenant-beta/billing-db per diagnostic note.",
            citations=citations,
        )

    if "payments-worker" in lower or "latency" in lower:
        # Simulates a model fooled by rb-alpha-poison-in-tenant
        return ChangePlan(
            resource_type="service",
            target_resource="payments-worker",
            action="restart",
            rollback_plan="Verify payments-worker health after restart.",
            risk_level="high",
            summary="Restart payments-worker per latency troubleshooting note.",
            citations=citations,
        )

    # Default happy path: checkout-api 502 rollback
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


@dataclass
class WorkflowDeps:
    """Injectable dependencies so the same StateGraph runs in Studio, pytest, and Worker."""

    settings: Settings = field(default_factory=get_settings)
    domain_store: PostgresDomainStore | InMemoryDomainStore | None = None
    policy: PolicyEngine = field(default_factory=PolicyEngine)
    use_postgres: bool = False
    use_mcp_http: bool = False
    approvals_memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    audit_memory: list[AuditEvent] = field(default_factory=list)
    attempt: int = 1
    simulate_write_timeout: bool = False

    def get_domain_store(self) -> PostgresDomainStore | InMemoryDomainStore:
        if self.domain_store is not None:
            return self.domain_store
        if self.use_postgres:
            return PostgresDomainStore(self.settings.database_url)
        self.domain_store = InMemoryDomainStore()
        return self.domain_store


class StudioRunContext(RunContext, frozen=True):
    """Dev fixture defaults so `langgraph dev` Studio works with an empty context payload."""

    tenant_id: str = "tenant-alpha"
    requester_id: str = "alice"
    requester_roles: frozenset[str] = frozenset({"engineer"})
    thread_id: str = "studio-thread"
    run_id: str = "studio-run"
    request_id: str = "studio-req"


def build_incident_desk_graph(
    *,
    deps: WorkflowDeps | None = None,
    checkpointer: Any = None,
    store: Any = None,
    context_schema: type[RunContext] = RunContext,
    default_context: RunContext | None = None,
):
    """Compile the full IT/Ops Desk StateGraph with context_schema=RunContext."""
    wf_deps = deps or WorkflowDeps()
    domain = wf_deps.get_domain_store()

    def _ctx(runtime: Runtime[RunContext]) -> RunContext:
        if runtime.context is not None:
            return runtime.context
        if default_context is not None:
            return default_context
        raise ValueError("Missing required RunContext")

    # ------------------------------------------------------------------
    # Read-only tools for the bounded `investigate` create_agent node
    # (Write tools NEVER exist on the LLM agent!)
    # ------------------------------------------------------------------
    @tool
    async def get_service_status(service_name: str, runtime: ToolRuntime[RunContext]) -> str:
        """Inspect current version and health of a service in the caller's tenant."""
        ctx = runtime.context
        try:
            if wf_deps.use_mcp_http:
                tok = mint_mcp_token(
                    tenant_id=ctx.tenant_id,
                    requester_id=ctx.requester_id,
                    run_id=ctx.run_id,
                    secret=wf_deps.settings.mcp_token_secret,
                )
                res = await call_mcp_tool(
                    url=wf_deps.settings.mcp_infra_url,
                    bearer_token=tok,
                    tool_name="get_service_status",
                    arguments={"service_name": service_name},
                )
            else:
                res = await domain.get_service_status(
                    tenant_id=ctx.tenant_id, service_name=service_name
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
            if wf_deps.use_mcp_http:
                tok = mint_mcp_token(
                    tenant_id=ctx.tenant_id,
                    requester_id=ctx.requester_id,
                    run_id=ctx.run_id,
                    secret=wf_deps.settings.mcp_token_secret,
                )
                res = await call_mcp_tool(
                    url=wf_deps.settings.mcp_infra_url,
                    bearer_token=tok,
                    tool_name="list_recent_deploys",
                    arguments={"service_name": service_name},
                )
            else:
                res = await domain.list_recent_deploys(
                    tenant_id=ctx.tenant_id, service_name=service_name
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
            if wf_deps.use_mcp_http:
                tok = mint_mcp_token(
                    tenant_id=ctx.tenant_id,
                    requester_id=ctx.requester_id,
                    run_id=ctx.run_id,
                    secret=wf_deps.settings.mcp_token_secret,
                )
                res = await call_mcp_tool(
                    url=wf_deps.settings.mcp_ticketing_url,
                    bearer_token=tok,
                    tool_name="get_ticket",
                    arguments={"ticket_id": ticket_id},
                )
            else:
                res = await domain.get_ticket(tenant_id=ctx.tenant_id, ticket_id=ticket_id)
            return cap_tool_output(
                json.dumps(res), TOOL_CONTRACTS["get_ticket"].max_output_chars
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    read_only_tools = [get_service_status, list_recent_deploys, get_ticket]

    async def _record_audit(event: AuditEvent) -> None:
        if wf_deps.use_postgres:
            async with (
                await psycopg.AsyncConnection.connect(wf_deps.settings.database_url) as conn,
                conn.transaction(),
            ):
                await set_tenant_context(conn, event.tenant_id)
                await write_audit_event(conn, event)
        else:
            wf_deps.audit_memory.append(event)

    # ------------------------------------------------------------------
    # Node 1: retrieve_context (ACL pgvector + optional Store lookup)
    # ------------------------------------------------------------------
    async def retrieve_context(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        ctx = _ctx(runtime)
        query = state.get("request_text", "")
        if wf_deps.use_postgres:
            embeddings = get_embeddings(tenant_id=ctx.tenant_id, settings=wf_deps.settings)
            async with await psycopg.AsyncConnection.connect(
                wf_deps.settings.database_url
            ) as conn:
                docs = await retrieve_runbooks(
                    conn,
                    embeddings=embeddings,
                    tenant_id=ctx.tenant_id,
                    roles=ctx.requester_roles,
                    query=query,
                    limit=4,
                )
            runbooks = [d.model_dump() for d in docs]
        else:
            # Strict in-memory ACL filter matching SQL WHERE tenant_id AND allowed_roles &&
            runbooks = [
                {
                    "doc_id": rb.doc_id,
                    "title": rb.title,
                    "content": rb.content,
                    "metadata": rb.metadata,
                }
                for rb in SEED_RUNBOOKS
                if rb.tenant_id == ctx.tenant_id
                and (set(rb.allowed_roles) & set(ctx.requester_roles))
            ]

        prior_memories: list[dict[str, Any]] = []
        if runtime.store is not None:
            mem_items = await runtime.store.asearch(
                (ctx.tenant_id, "memory", "incident_desk"),
                limit=5,
            )
            prior_memories = [dict(item.value) for item in mem_items]

        return {
            "release_version": RELEASE_VERSION,
            "retrieved_runbooks": runbooks,
            "prior_memories": prior_memories,
        }

    # ------------------------------------------------------------------
    # Node 2: investigate (LangChain v1 create_agent + Middleware)
    # ------------------------------------------------------------------
    async def investigate(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        _maybe_inject_fault("investigate")
        ctx = _ctx(runtime)
        req_text = state.get("request_text", "")

        if wf_deps.settings.model_mode == "mock":
            target_svc = (
                "payments-worker" if "payments-worker" in req_text.lower() else "checkout-api"
            )
            mock_turns = [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_investigate_1",
                            "name": "get_service_status",
                            "args": {"service_name": target_svc},
                        }
                    ],
                ),
                AIMessage(
                    content=(
                        f"Investigated {target_svc} for tenant {ctx.tenant_id} "
                        f"using {len(state.get('retrieved_runbooks', []))} runbooks."
                    )
                ),
            ]
            chat_model = get_chat_model(
                alias="ops-primary",
                tenant_id=ctx.tenant_id,
                settings=wf_deps.settings,
                mock_responses=mock_turns,
            )
            fast_model = get_chat_model(
                alias="ops-fast",
                tenant_id=ctx.tenant_id,
                settings=wf_deps.settings,
                mock_responses=[AIMessage(content="Summary")],
            )
        else:
            chat_model = get_chat_model(
                alias="ops-primary", tenant_id=ctx.tenant_id, settings=wf_deps.settings
            )
            fast_model = get_chat_model(
                alias="ops-fast", tenant_id=ctx.tenant_id, settings=wf_deps.settings
            )

        agent = create_agent(
            model=chat_model,
            tools=read_only_tools,
            system_prompt=(
                "You are an IT/Ops incident investigator. Inspect service health using "
                "read-only tools only. Summarize findings concisely."
            ),
            context_schema=RunContext,
            middleware=[
                PIIMiddleware("email", detector=EMAIL_PATTERN, strategy="redact"),
                PIIMiddleware("phone_number", detector=PHONE_PATTERN, strategy="redact"),
                PIIMiddleware("api_key", detector=API_KEY_PATTERN, strategy="redact"),
                SummarizationMiddleware(
                    model=fast_model,
                    trigger=("tokens", 4000),
                    keep=("messages", 10),
                ),
                ModelCallLimitMiddleware(run_limit=wf_deps.settings.max_model_calls),
                ToolCallLimitMiddleware(run_limit=wf_deps.settings.max_tool_calls),
            ],
        )

        agent_result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": req_text}]},
            context=ctx,
        )
        last_msg = agent_result["messages"][-1]
        summary = getattr(last_msg, "content", str(last_msg))
        return {
            "messages": agent_result["messages"],
            "investigation_summary": str(summary),
        }

    # ------------------------------------------------------------------
    # Node 3: propose_plan (Structured ChangePlan)
    # ------------------------------------------------------------------
    async def propose_plan(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        ctx = _ctx(runtime)
        runbooks = state.get("retrieved_runbooks", [])
        req_text = state.get("request_text", "")

        if wf_deps.settings.model_mode == "mock":
            plan = _deterministic_mock_plan(req_text, runbooks)
        else:
            model = get_chat_model(
                alias="ops-primary", tenant_id=ctx.tenant_id, settings=wf_deps.settings
            )
            structured = model.with_structured_output(ChangePlan)
            safe_req = redact_text(req_text)
            safe_summary = redact_text(state.get("investigation_summary", ""))
            safe_runbooks = redact_obj(runbooks)
            plan = await structured.ainvoke(
                f"Request: {safe_req}\n"
                f"Investigation: {safe_summary}\n"
                f"Runbooks: {json.dumps(safe_runbooks)}"
            )

        return {"change_plan": plan.model_dump(mode="json")}

    def route_after_plan(state: IncidentDeskState) -> str:
        plan = _coerce_plan(state.get("change_plan"))
        if plan is None or plan.action == "none":
            return "finalize_no_action"
        return "evaluate_policy"

    # ------------------------------------------------------------------
    # Node 4: evaluate_policy + durable operation_id generation
    # ------------------------------------------------------------------
    async def evaluate_policy(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        ctx = _ctx(runtime)
        plan = _coerce_plan(state.get("change_plan"))
        assert plan is not None

        res = wf_deps.policy.evaluate_change_plan(ctx=ctx, plan=plan)
        operation_id = f"op_{uuid.uuid4().hex[:16]}"
        audit_decision = "deny" if res.decision == PolicyDecision.DENY else "allow"
        await _record_audit(
            AuditEvent(
                tenant_id=ctx.tenant_id,
                actor_id=ctx.requester_id,
                thread_id=ctx.thread_id,
                run_id=ctx.run_id,
                workflow_version=state.get("release_version", RELEASE_VERSION),
                event_type="authz_check",
                decision=audit_decision,
                step_or_op_key=f"policy:{operation_id}",
                attempt=wf_deps.attempt,
                operation_id=operation_id,
                details={
                    "reason": res.reason,
                    "action": plan.action,
                    "target": plan.target_resource,
                },
            )
        )
        if res.decision == PolicyDecision.DENY:
            return {
                "policy_decision": "deny",
                "operation_id": operation_id,
                "error_category": "permission_denied",
            }
        return {
            "policy_decision": "require_approval",
            "operation_id": operation_id,
        }

    def route_after_policy(state: IncidentDeskState) -> str:
        if state.get("policy_decision") == "deny":
            return "finalize_denied"
        return "record_approval_request"

    # ------------------------------------------------------------------
    # Node 5: record_approval_request (Idempotent; NO interrupt() here)
    # ------------------------------------------------------------------
    async def record_approval_request(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        ctx = _ctx(runtime)
        plan = _coerce_plan(state.get("change_plan"))
        operation_id = state.get("operation_id")
        assert plan is not None and operation_id is not None

        approval_id = f"apr_{operation_id}"
        plan_json = plan.model_dump(mode="json")
        plan_hash = plan.plan_hash()

        if wf_deps.use_postgres:
            async with (
                await psycopg.AsyncConnection.connect(wf_deps.settings.database_url) as conn,
                conn.transaction(),
            ):
                await set_tenant_context(conn, ctx.tenant_id)
                await conn.execute(
                    """
                    INSERT INTO approvals
                        (approval_id, thread_id, tenant_id, requested_by,
                         operation_id, plan, plan_hash, status)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, 'pending')
                    ON CONFLICT (approval_id) DO NOTHING
                    """,
                    (
                        approval_id,
                        ctx.thread_id,
                        ctx.tenant_id,
                        ctx.requester_id,
                        operation_id,
                        json.dumps(plan_json),
                        plan_hash,
                    ),
                )
        else:
            wf_deps.approvals_memory.setdefault(
                approval_id,
                {
                    "approval_id": approval_id,
                    "thread_id": ctx.thread_id,
                    "tenant_id": ctx.tenant_id,
                    "requested_by": ctx.requester_id,
                    "operation_id": operation_id,
                    "plan": plan_json,
                    "plan_hash": plan_hash,
                    "status": "pending",
                },
            )

        await _record_audit(
            AuditEvent(
                tenant_id=ctx.tenant_id,
                actor_id=ctx.requester_id,
                thread_id=ctx.thread_id,
                run_id=ctx.run_id,
                workflow_version=state.get("release_version", RELEASE_VERSION),
                event_type="approval_requested",
                decision="allow",
                step_or_op_key=approval_id,
                attempt=wf_deps.attempt,
                operation_id=operation_id,
                details={"approval_id": approval_id, "plan_hash": plan_hash},
            )
        )
        return {}

    # ------------------------------------------------------------------
    # Node 6: await_approval (ONLY interrupt() — safe for replay)
    # ------------------------------------------------------------------
    async def await_approval(state: IncidentDeskState) -> dict[str, Any]:
        plan = _coerce_plan(state.get("change_plan"))
        operation_id = state.get("operation_id")
        decision_payload = interrupt(
            {
                "approval_id": f"apr_{operation_id}",
                "operation_id": operation_id,
                "plan": plan.model_dump(mode="json") if plan else None,
                "plan_hash": plan.plan_hash() if plan else None,
            }
        )
        updates: dict[str, Any] = {"approval_decision": decision_payload}
        if isinstance(decision_payload, dict) and decision_payload.get("decision") == "approve":
            updates["approved_plan_hash"] = decision_payload.get("approved_plan_hash")
        return updates

    def route_after_approval(state: IncidentDeskState) -> str:
        dec = (state.get("approval_decision") or {}).get("decision")
        if dec == "approve":
            return "execute_change"
        if dec == "edit":
            return "revalidate_edit"
        return "finalize_rejected"

    # ------------------------------------------------------------------
    # Node 7: revalidate_edit (Re-run PolicyEngine + new operation_id)
    # ------------------------------------------------------------------
    async def revalidate_edit(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        ctx = _ctx(runtime)
        plan = _coerce_plan(state.get("change_plan"))
        decision = state.get("approval_decision") or {}
        edits = decision.get("edits") or {}
        assert plan is not None

        merged = {**plan.model_dump(mode="json"), **edits}
        edited_plan = ChangePlan.model_validate(merged)
        res = wf_deps.policy.evaluate_change_plan(ctx=ctx, plan=edited_plan)
        new_op_id = f"op_{uuid.uuid4().hex[:16]}"

        await _record_audit(
            AuditEvent(
                tenant_id=ctx.tenant_id,
                actor_id=str(decision.get("approver_id", ctx.requester_id)),
                requested_by=ctx.requester_id,
                approved_by=str(decision.get("approver_id", "")),
                thread_id=ctx.thread_id,
                run_id=ctx.run_id,
                workflow_version=state.get("release_version", RELEASE_VERSION),
                event_type="authz_check",
                decision="deny" if res.decision == PolicyDecision.DENY else "edited",
                step_or_op_key=f"revalidate:{new_op_id}",
                attempt=wf_deps.attempt,
                operation_id=new_op_id,
                details={"reason": res.reason, "edited_action": edited_plan.action},
            )
        )
        if res.decision == PolicyDecision.DENY:
            return {
                "change_plan": edited_plan.model_dump(mode="json"),
                "policy_decision": "deny",
                "error_category": "permission_denied",
            }
        return {
            "change_plan": edited_plan.model_dump(mode="json"),
            "policy_decision": "require_approval",
            "operation_id": new_op_id,
            "approved_plan_hash": edited_plan.plan_hash(),
        }

    def route_after_edit(state: IncidentDeskState) -> str:
        if state.get("policy_decision") == "deny":
            return "finalize_denied"
        return "execute_change"

    # ------------------------------------------------------------------
    # Node 8: execute_change (Approved-hash check + Idempotent write)
    # ------------------------------------------------------------------
    async def execute_change(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        ctx = _ctx(runtime)
        plan = _coerce_plan(state.get("change_plan"))
        operation_id = state.get("operation_id")
        approved_hash = state.get("approved_plan_hash")
        decision = state.get("approval_decision") or {}
        approver_id = str(decision.get("approver_id") or "")

        assert plan is not None and operation_id is not None

        # AC-15: Refuse to write unless plan in state matches approved_plan_hash
        if not approved_hash or plan.plan_hash() != approved_hash:
            await _record_audit(
                AuditEvent(
                    tenant_id=ctx.tenant_id,
                    actor_id=approver_id or ctx.requester_id,
                    requested_by=ctx.requester_id,
                    approved_by=approver_id or None,
                    thread_id=ctx.thread_id,
                    run_id=ctx.run_id,
                    workflow_version=state.get("release_version", RELEASE_VERSION),
                    event_type="authz_check",
                    decision="deny",
                    step_or_op_key=f"hash_mismatch:{operation_id}",
                    attempt=wf_deps.attempt,
                    operation_id=operation_id,
                    details={
                        "reason": "approved_plan_hash mismatch",
                        "expected": approved_hash,
                        "actual": plan.plan_hash(),
                    },
                )
            )
            return {
                "error_category": "permission_denied",
                "final_status": "denied_plan_mismatch",
            }

        # Mint short-lived signed MCP token carrying approved_by & operation_id
        mcp_token = mint_mcp_token(
            tenant_id=ctx.tenant_id,
            requester_id=ctx.requester_id,
            run_id=ctx.run_id,
            approved_by=approver_id,
            operation_id=operation_id,
            secret=wf_deps.settings.mcp_token_secret,
        )

        try:
            if wf_deps.simulate_write_timeout:
                raise TimeoutError("Simulated downstream timeout")

            if plan.resource_type == "ticket" or plan.action == "update_ticket":
                tool_id = "update_ticket"
                if wf_deps.use_mcp_http:
                    result = await call_mcp_tool(
                        url=wf_deps.settings.mcp_ticketing_url,
                        bearer_token=mcp_token,
                        tool_name="update_ticket",
                        arguments={
                            "ticket_id": plan.ticket_id or plan.target_resource,
                            "status": "resolved",
                            "note": plan.summary,
                            "idempotency_key": operation_id,
                        },
                    )
                else:
                    result = await domain.update_ticket(
                        tenant_id=ctx.tenant_id,
                        ticket_id=plan.ticket_id or plan.target_resource,
                        status="resolved",
                        note=plan.summary,
                        operation_id=operation_id,
                    )
            else:
                tool_id = "execute_service_change"
                if wf_deps.use_mcp_http:
                    result = await call_mcp_tool(
                        url=wf_deps.settings.mcp_infra_url,
                        bearer_token=mcp_token,
                        tool_name="execute_service_change",
                        arguments={
                            "service_name": plan.target_resource,
                            "action": plan.action,
                            "target_version": plan.target_version,
                            "idempotency_key": operation_id,
                        },
                    )
                else:
                    result = await domain.execute_service_change(
                        tenant_id=ctx.tenant_id,
                        service_name=plan.target_resource,
                        action=plan.action,
                        target_version=plan.target_version,
                        operation_id=operation_id,
                    )

            # AC-2: Fault injection AFTER downstream commit, BEFORE checkpoint returns
            _maybe_inject_fault("after_write")

        except TimeoutError:
            # Explicit outcome_unknown -> reconcile via get_operation_status
            await _record_audit(
                AuditEvent(
                    tenant_id=ctx.tenant_id,
                    actor_id=approver_id or ctx.requester_id,
                    requested_by=ctx.requester_id,
                    approved_by=approver_id or None,
                    thread_id=ctx.thread_id,
                    run_id=ctx.run_id,
                    workflow_version=state.get("release_version", RELEASE_VERSION),
                    event_type="tool_write",
                    tool_id="execute_service_change",
                    operation_id=operation_id,
                    decision="outcome_unknown",
                    step_or_op_key=f"write:{operation_id}",
                    attempt=wf_deps.attempt,
                    details={"reason": "timeout; reconciling"},
                )
            )
            if wf_deps.use_mcp_http:
                reconciled = await call_mcp_tool(
                    url=wf_deps.settings.mcp_infra_url,
                    bearer_token=mcp_token,
                    tool_name="get_operation_status",
                    arguments={"operation_id": operation_id},
                )
            else:
                reconciled = await domain.get_operation_status(
                    tenant_id=ctx.tenant_id, operation_id=operation_id
                )
            rec_decision = "succeeded" if reconciled["status"] == "succeeded" else "outcome_unknown"
            await _record_audit(
                AuditEvent(
                    tenant_id=ctx.tenant_id,
                    actor_id=approver_id or ctx.requester_id,
                    requested_by=ctx.requester_id,
                    approved_by=approver_id or None,
                    thread_id=ctx.thread_id,
                    run_id=ctx.run_id,
                    workflow_version=state.get("release_version", RELEASE_VERSION),
                    event_type="reconciliation",
                    tool_id="get_operation_status",
                    operation_id=operation_id,
                    decision=rec_decision,
                    step_or_op_key=f"reconcile:{operation_id}",
                    attempt=wf_deps.attempt,
                    details=reconciled,
                )
            )
            if reconciled["status"] == "succeeded":
                return {"execution_result": reconciled["result_payload"]}
            return {
                "error_category": "outcome_unknown",
                "final_status": "outcome_unknown",
                "execution_result": reconciled,
            }

        await _record_audit(
            AuditEvent(
                tenant_id=ctx.tenant_id,
                actor_id=approver_id or ctx.requester_id,
                requested_by=ctx.requester_id,
                approved_by=approver_id or None,
                thread_id=ctx.thread_id,
                run_id=ctx.run_id,
                workflow_version=state.get("release_version", RELEASE_VERSION),
                event_type="tool_write",
                tool_id=tool_id,
                operation_id=operation_id,
                decision="succeeded",
                step_or_op_key=f"write:{operation_id}",
                attempt=wf_deps.attempt,
                details=result,
            )
        )
        return {"execution_result": result}

    def route_after_execute(state: IncidentDeskState) -> str:
        if state.get("error_category") is not None:
            return END
        return "verify_end_state"

    # ------------------------------------------------------------------
    # Node 9: verify_end_state (Deterministic read-back verification)
    # ------------------------------------------------------------------
    async def verify_end_state(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        ctx = _ctx(runtime)
        plan = _coerce_plan(state.get("change_plan"))
        assert plan is not None

        if plan.resource_type == "ticket" or plan.action == "update_ticket":
            if wf_deps.use_mcp_http:
                tok = mint_mcp_token(
                    tenant_id=ctx.tenant_id,
                    requester_id=ctx.requester_id,
                    run_id=ctx.run_id,
                    secret=wf_deps.settings.mcp_token_secret,
                )
                ticket = await call_mcp_tool(
                    url=wf_deps.settings.mcp_ticketing_url,
                    bearer_token=tok,
                    tool_name="get_ticket",
                    arguments={"ticket_id": plan.ticket_id or plan.target_resource},
                )
            else:
                ticket = await domain.get_ticket(
                    tenant_id=ctx.tenant_id, ticket_id=plan.ticket_id or plan.target_resource
                )
            passed = ticket["status"] == "resolved"
        else:
            if wf_deps.use_mcp_http:
                tok = mint_mcp_token(
                    tenant_id=ctx.tenant_id,
                    requester_id=ctx.requester_id,
                    run_id=ctx.run_id,
                    secret=wf_deps.settings.mcp_token_secret,
                )
                svc = await call_mcp_tool(
                    url=wf_deps.settings.mcp_infra_url,
                    bearer_token=tok,
                    tool_name="get_service_status",
                    arguments={"service_name": plan.target_resource},
                )
            else:
                svc = await domain.get_service_status(
                    tenant_id=ctx.tenant_id, service_name=plan.target_resource
                )
            version_ok = (
                svc["current_version"] == plan.target_version if plan.target_version else True
            )
            passed = svc["status"] == "healthy" and version_ok

        if passed and runtime.store is not None:
            op_key = state.get("operation_id") or ctx.run_id
            await runtime.store.aput(
                (ctx.tenant_id, "memory", "incident_desk"),
                op_key,
                {
                    "thread_id": ctx.thread_id,
                    "operation_id": op_key,
                    "target": plan.target_resource,
                    "action": plan.action,
                    "summary": redact_text(plan.summary),
                },
            )

        return {
            "verification_passed": passed,
            "final_status": "completed" if passed else "verification_failed",
        }

    # ------------------------------------------------------------------
    # Terminal finalizer nodes
    # ------------------------------------------------------------------
    def finalize_no_action(state: IncidentDeskState) -> dict[str, Any]:
        return {"final_status": "no_action", "verification_passed": True}

    def finalize_denied(state: IncidentDeskState) -> dict[str, Any]:
        return {"final_status": "denied", "verification_passed": False}

    def finalize_rejected(state: IncidentDeskState) -> dict[str, Any]:
        return {"final_status": "rejected", "verification_passed": False}

    # ------------------------------------------------------------------
    # Assemble StateGraph
    # ------------------------------------------------------------------
    builder = StateGraph(IncidentDeskState, context_schema=context_schema)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("investigate", investigate)
    builder.add_node("propose_plan", propose_plan)
    builder.add_node("finalize_no_action", finalize_no_action)
    builder.add_node("evaluate_policy", evaluate_policy)
    builder.add_node("finalize_denied", finalize_denied)
    builder.add_node("record_approval_request", record_approval_request)
    builder.add_node("await_approval", await_approval)
    builder.add_node("finalize_rejected", finalize_rejected)
    builder.add_node("revalidate_edit", revalidate_edit)
    builder.add_node("execute_change", execute_change)
    builder.add_node("verify_end_state", verify_end_state)

    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "investigate")
    builder.add_edge("investigate", "propose_plan")
    builder.add_conditional_edges(
        "propose_plan",
        route_after_plan,
        {"finalize_no_action": "finalize_no_action", "evaluate_policy": "evaluate_policy"},
    )
    builder.add_edge("finalize_no_action", END)
    builder.add_conditional_edges(
        "evaluate_policy",
        route_after_policy,
        {
            "finalize_denied": "finalize_denied",
            "record_approval_request": "record_approval_request",
        },
    )
    builder.add_edge("finalize_denied", END)
    builder.add_edge("record_approval_request", "await_approval")
    builder.add_conditional_edges(
        "await_approval",
        route_after_approval,
        {
            "execute_change": "execute_change",
            "revalidate_edit": "revalidate_edit",
            "finalize_rejected": "finalize_rejected",
        },
    )
    builder.add_edge("finalize_rejected", END)
    builder.add_conditional_edges(
        "revalidate_edit",
        route_after_edit,
        {"finalize_denied": "finalize_denied", "execute_change": "execute_change"},
    )
    builder.add_conditional_edges(
        "execute_change",
        route_after_execute,
        {"verify_end_state": "verify_end_state", END: END},
    )
    builder.add_edge("verify_end_state", END)

    return builder.compile(checkpointer=checkpointer, store=store)


def build_studio_graph():
    """Compiled WITHOUT a checkpointer so `langgraph dev` supplies local persistence (spec §5.7)."""
    return build_incident_desk_graph(
        deps=WorkflowDeps(use_postgres=False),
        checkpointer=None,
        context_schema=StudioRunContext,
        default_context=StudioRunContext(),
    )
