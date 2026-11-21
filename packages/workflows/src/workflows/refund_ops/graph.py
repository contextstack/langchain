"""Customer Billing / Refund Operations Desk StateGraph (v2.4 second domain)."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage

from core.models import ChangePlan, RunbookCitation, RunContext
from core.pii import redact_text
from core.tool_contract import TOOL_CONTRACTS, cap_tool_output
from workflows.lifecycle import (
    DomainHooks,
    StudioRunContext,
    WorkflowDeps,
    build_governed_workflow_graph,
)
from workflows.refund_ops.version import RELEASE_VERSION
from workflows.tool_executor import ToolExecutor

_DOLLAR_RE = re.compile(r"\$([0-9][0-9,]*(?:\.[0-9]{1,2})?)")
_ACCOUNT_RE = re.compile(r"\b(ACC-\d+)\b", re.IGNORECASE)


def _parse_amount_cents(request_text: str) -> int:
    match = _DOLLAR_RE.search(request_text)
    if match:
        raw = match.group(1).replace(",", "")
        return int(round(float(raw) * 100))
    return 15_000


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


def _build_read_tools(executor: ToolExecutor) -> list[Any]:
    @tool
    async def get_account(account_id: str, runtime: ToolRuntime[RunContext]) -> str:
        """Inspect a customer billing account balance and status in the caller's tenant."""
        ctx = runtime.context
        try:
            res = await executor.invoke_read(
                ctx,
                tool_id="get_account",
                mcp_url=executor.settings.mcp_billing_url,
                mcp_args={"account_id": account_id},
                direct_fn=lambda: executor.domain.get_account(
                    tenant_id=ctx.tenant_id, account_id=account_id
                ),
            )
            return cap_tool_output(
                json.dumps(res), TOOL_CONTRACTS["get_account"].max_output_chars
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    return [get_account]


def _mock_investigate_turns(
    req_text: str, ctx: RunContext, runbooks: list[dict[str, Any]]
) -> list[AIMessage]:
    target_acct = _parse_account_id(req_text)
    return [
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
                f"using {len(runbooks)} policy docs."
            )
        ),
    ]


async def _execute_write(
    executor: ToolExecutor,
    ctx: RunContext,
    plan: ChangePlan,
    approver_id: str,
    operation_id: str,
    simulate_timeout: bool,
) -> tuple[str, dict[str, Any]]:
    amount_cents = int(plan.amount_cents or 0)
    result = await executor.invoke_write(
        ctx,
        tool_id="issue_refund",
        mcp_url=executor.settings.mcp_billing_url,
        mcp_args={
            "account_id": plan.target_resource,
            "amount_cents": amount_cents,
            "reason": plan.summary,
            "idempotency_key": operation_id,
        },
        direct_fn=lambda: executor.domain.issue_refund(
            tenant_id=ctx.tenant_id,
            account_id=plan.target_resource,
            amount_cents=amount_cents,
            reason=plan.summary,
            operation_id=operation_id,
        ),
        approved_by=approver_id,
        operation_id=operation_id,
        simulate_timeout=simulate_timeout,
    )
    return "issue_refund", result


async def _verify_end_state(
    executor: ToolExecutor,
    ctx: RunContext,
    plan: ChangePlan,
    operation_id: str,
    exec_res: dict[str, Any],
) -> bool:
    op_status = await executor.reconcile_operation(
        ctx,
        mcp_url=executor.settings.mcp_billing_url,
        operation_id=operation_id,
    )
    acct = await executor.invoke_read(
        ctx,
        tool_id="get_account",
        mcp_url=executor.settings.mcp_billing_url,
        mcp_args={"account_id": plan.target_resource},
        direct_fn=lambda: executor.domain.get_account(
            tenant_id=ctx.tenant_id, account_id=plan.target_resource
        ),
    )
    expected_balance = exec_res.get("new_balance_cents")
    op_committed = (
        op_status.get("status") == "succeeded"
        and (op_status.get("result_payload") or {}).get("amount_cents") == plan.amount_cents
    )
    balance_ok = (
        expected_balance is not None
        and 0 <= int(acct["balance_cents"]) <= int(expected_balance)
        and acct["status"] == "active"
    )
    return op_committed and balance_ok


def _build_memory_payload(
    ctx: RunContext, plan: ChangePlan, operation_id: str
) -> dict[str, Any]:
    return {
        "thread_id": ctx.thread_id,
        "operation_id": operation_id,
        "target": plan.target_resource,
        "action": plan.action,
        "amount_cents": plan.amount_cents,
        "summary": redact_text(plan.summary),
    }


REFUND_OPS_HOOKS = DomainHooks(
    workflow_name="refund_ops",
    release_version=RELEASE_VERSION,
    reconcile_mcp_url=lambda s: s.mcp_billing_url,
    default_write_tool_id="issue_refund",
    investigate_system_prompt=(
        "You are a Billing & Refund Operations investigator. Inspect customer "
        "account status using read-only tools only. Summarize findings concisely."
    ),
    build_read_tools=_build_read_tools,
    build_mock_investigate_turns=_mock_investigate_turns,
    deterministic_mock_plan=_deterministic_mock_refund_plan,
    execute_domain_write=_execute_write,
    verify_domain_end_state=_verify_end_state,
    build_memory_payload=_build_memory_payload,
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
    return build_governed_workflow_graph(
        REFUND_OPS_HOOKS,
        deps=deps,
        checkpointer=checkpointer,
        store=store,
        context_schema=context_schema,
        default_context=default_context,
    )


def build_studio_graph():
    """Compiled WITHOUT a checkpointer so `langgraph dev` supplies local persistence."""
    return build_refund_ops_graph(
        deps=WorkflowDeps(use_postgres=False),
        checkpointer=None,
        context_schema=StudioRunContext,
        default_context=StudioRunContext(),
    )
