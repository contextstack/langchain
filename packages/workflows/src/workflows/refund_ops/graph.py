"""Customer Billing / Refund Operations Desk StateGraph (v2.4 second domain)."""

from __future__ import annotations

import json
import re
import uuid
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
from core.db import set_tenant_context
from core.models import ChangePlan, RunbookCitation, RunContext
from core.pii import API_KEY_PATTERN, EMAIL_PATTERN, PHONE_PATTERN, redact_obj, redact_text
from core.policy import PolicyDecision
from core.tool_contract import TOOL_CONTRACTS, cap_tool_output
from gateway.factory import get_chat_model, get_embeddings
from knowledge.fixtures import SEED_RUNBOOKS
from knowledge.retriever import retrieve_runbooks
from workflows.incident_desk.graph import (
    StudioRunContext,
    WorkflowDeps,
    _coerce_plan,
    _maybe_inject_fault,
)
from workflows.incident_desk.state import IncidentDeskState
from workflows.mcp_client import call_mcp_tool
from workflows.refund_ops.version import RELEASE_VERSION

_DOLLAR_RE = re.compile(r"\$([0-9][0-9,]*(?:\.[0-9]{1,2})?)")
_ACCOUNT_RE = re.compile(r"\b(ACC-\d+)\b", re.IGNORECASE)


def _parse_amount_cents(request_text: str) -> int:
    match = _DOLLAR_RE.search(request_text)
    if match:
        raw = match.group(1).replace(",", "")
        return int(round(float(raw) * 100))
    return 15_000  # Default $150.00 refund


def _parse_account_id(request_text: str) -> str:
    match = _ACCOUNT_RE.search(request_text)
    if match:
        return match.group(1).upper()
    return "ACC-1001"


def _deterministic_mock_refund_plan(
    request_text: str,
    runbooks: list[dict[str, Any]],
) -> ChangePlan:
    """Deterministic ChangePlan synthesizer for MODEL_MODE=mock billing scenarios."""
    lower = request_text.lower()
    citations = [
        RunbookCitation(doc_id=rb["doc_id"], title=rb["title"]) for rb in runbooks[:3]
    ]

    if "no action" in lower or "balance only" in lower:
        return ChangePlan(
            resource_type="none",
            target_resource="",
            action="none",
            risk_level="low",
            summary="Account inspected; no refund required.",
            citations=citations,
        )

    amount_cents = _parse_amount_cents(request_text)
    account_id = _parse_account_id(request_text)

    if "tenant-beta" in lower or account_id == "ACC-2001":
        return ChangePlan(
            resource_type="account",
            target_resource=f"tenant-beta/{account_id}",
            action="issue_refund",
            amount_cents=amount_cents,
            risk_level="high",
            summary=f"Issue ${amount_cents / 100:.2f} refund to tenant-beta/{account_id}.",
            citations=citations,
        )

    return ChangePlan(
        resource_type="account",
        target_resource=account_id,
        action="issue_refund",
        amount_cents=amount_cents,
        rollback_plan="Issue compensating invoice debit if refund was issued in error.",
        risk_level="high" if amount_cents > 50_000 else "medium",
        summary=f"Issue ${amount_cents / 100:.2f} refund to customer account {account_id}.",
        citations=citations,
    )


def build_refund_ops_graph(
    *,
    deps: WorkflowDeps | None = None,
    checkpointer: Any = None,
    store: Any = None,
    context_schema: type[RunContext] = RunContext,
    default_context: RunContext | None = None,
):
    """Compile the Customer Billing / Refund Ops StateGraph on the shared substrate."""
    wf_deps = deps or WorkflowDeps()
    domain = wf_deps.get_domain_store()

    def _ctx(runtime: Runtime[RunContext]) -> RunContext:
        if runtime.context is not None:
            return runtime.context
        if default_context is not None:
            return default_context
        raise ValueError("Missing required RunContext")

    # ------------------------------------------------------------------
    # Read-only tool for the bounded `investigate` create_agent node
    # (Write tools NEVER exist on the LLM agent!)
    # ------------------------------------------------------------------
    @tool
    async def get_account(account_id: str, runtime: ToolRuntime[RunContext]) -> str:
        """Inspect a customer billing account balance and status in the caller's tenant."""
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
                    url=wf_deps.settings.mcp_billing_url,
                    bearer_token=tok,
                    tool_name="get_account",
                    arguments={"account_id": account_id},
                )
            else:
                res = await domain.get_account(tenant_id=ctx.tenant_id, account_id=account_id)
            return cap_tool_output(
                json.dumps(res), TOOL_CONTRACTS["get_account"].max_output_chars
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    read_only_tools = [get_account]

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
    # Node 1: retrieve_context (ACL pgvector)
    # ------------------------------------------------------------------
    async def retrieve_context(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        ctx = _ctx(runtime)
        query = state.get("request_text", "")
        if wf_deps.use_postgres:
            embeddings = get_embeddings(settings=wf_deps.settings)
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
                (ctx.tenant_id, "memory", "refund_ops"),
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
            target_acct = _parse_account_id(req_text)
            mock_turns = [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_billing_investigate_1",
                            "name": "get_account",
                            "args": {"account_id": target_acct},
                        }
                    ],
                ),
                AIMessage(
                    content=(
                        f"Inspected billing account {target_acct} for tenant {ctx.tenant_id} "
                        f"using {len(state.get('retrieved_runbooks', []))} policy docs."
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
                "You are a Billing & Refund Operations investigator. Inspect customer "
                "account status using read-only tools only. Summarize findings concisely."
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
            plan = _deterministic_mock_refund_plan(req_text, runbooks)
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
                f"Policies: {json.dumps(safe_runbooks)}"
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
                    "amount_cents": plan.amount_cents,
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
                details={
                    "reason": res.reason,
                    "edited_action": edited_plan.action,
                    "amount_cents": edited_plan.amount_cents,
                },
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
    # Node 8: execute_change (Approved-hash check + Idempotent refund)
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

        mcp_token = mint_mcp_token(
            tenant_id=ctx.tenant_id,
            requester_id=ctx.requester_id,
            run_id=ctx.run_id,
            approved_by=approver_id,
            operation_id=operation_id,
            secret=wf_deps.settings.mcp_token_secret,
        )

        amount_cents = int(plan.amount_cents or 0)
        try:
            if wf_deps.simulate_write_timeout:
                raise TimeoutError("Simulated downstream billing timeout")

            if wf_deps.use_mcp_http:
                result = await call_mcp_tool(
                    url=wf_deps.settings.mcp_billing_url,
                    bearer_token=mcp_token,
                    tool_name="issue_refund",
                    arguments={
                        "account_id": plan.target_resource,
                        "amount_cents": amount_cents,
                        "reason": plan.summary,
                        "idempotency_key": operation_id,
                    },
                )
            else:
                result = await domain.issue_refund(
                    tenant_id=ctx.tenant_id,
                    account_id=plan.target_resource,
                    amount_cents=amount_cents,
                    reason=plan.summary,
                    operation_id=operation_id,
                )

            _maybe_inject_fault("after_write")

        except TimeoutError:
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
                    tool_id="issue_refund",
                    operation_id=operation_id,
                    decision="outcome_unknown",
                    step_or_op_key=f"write:{operation_id}",
                    attempt=wf_deps.attempt,
                    details={"reason": "timeout; reconciling"},
                )
            )
            if wf_deps.use_mcp_http:
                reconciled = await call_mcp_tool(
                    url=wf_deps.settings.mcp_billing_url,
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
                tool_id="issue_refund",
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
    # Node 9: verify_end_state (Deterministic account balance read-back)
    # ------------------------------------------------------------------
    async def verify_end_state(
        state: IncidentDeskState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        ctx = _ctx(runtime)
        plan = _coerce_plan(state.get("change_plan"))
        exec_res = state.get("execution_result") or {}
        assert plan is not None

        if wf_deps.use_mcp_http:
            tok = mint_mcp_token(
                tenant_id=ctx.tenant_id,
                requester_id=ctx.requester_id,
                run_id=ctx.run_id,
                secret=wf_deps.settings.mcp_token_secret,
            )
            acct = await call_mcp_tool(
                url=wf_deps.settings.mcp_billing_url,
                bearer_token=tok,
                tool_name="get_account",
                arguments={"account_id": plan.target_resource},
            )
        else:
            acct = await domain.get_account(
                tenant_id=ctx.tenant_id, account_id=plan.target_resource
            )

        expected_balance = exec_res.get("new_balance_cents")
        passed = (
            expected_balance is not None
            and int(acct["balance_cents"]) == int(expected_balance)
            and acct["status"] == "active"
        )
        if passed and runtime.store is not None:
            op_key = state.get("operation_id") or ctx.run_id
            await runtime.store.aput(
                (ctx.tenant_id, "memory", "refund_ops"),
                op_key,
                {
                    "thread_id": ctx.thread_id,
                    "operation_id": op_key,
                    "target": plan.target_resource,
                    "action": plan.action,
                    "amount_cents": plan.amount_cents,
                    "summary": redact_text(plan.summary),
                },
            )

        return {
            "verification_passed": passed,
            "final_status": "completed" if passed else "verification_failed",
        }

    def finalize_no_action(state: IncidentDeskState) -> dict[str, Any]:
        return {"final_status": "no_action", "verification_passed": True}

    def finalize_denied(state: IncidentDeskState) -> dict[str, Any]:
        return {"final_status": "denied", "verification_passed": False}

    def finalize_rejected(state: IncidentDeskState) -> dict[str, Any]:
        return {"final_status": "rejected", "verification_passed": False}

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
    """Compiled WITHOUT a checkpointer so `langgraph dev` supplies local persistence."""
    return build_refund_ops_graph(
        deps=WorkflowDeps(use_postgres=False),
        checkpointer=None,
        context_schema=StudioRunContext,
        default_context=StudioRunContext(),
    )
