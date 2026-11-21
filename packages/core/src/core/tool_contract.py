"""Enterprise tool contract enforcement: timeouts, output caps, and metadata (spec §6)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from core.models import ToolContract

TRUNCATION_MARKER = "\n...[TRUNCATED_BY_TOOL_CONTRACT]"


def cap_tool_output(output: str, max_chars: int) -> str:
    if len(output) <= max_chars:
        return output
    keep = max(0, max_chars - len(TRUNCATION_MARKER))
    return output[:keep] + TRUNCATION_MARKER


async def invoke_with_contract(
    contract: ToolContract,
    fn: Callable[..., Awaitable[Any]],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute an async tool callable under the contract's timeout and output cap."""
    result = await asyncio.wait_for(fn(*args, **kwargs), timeout=contract.timeout_s)
    if isinstance(result, str):
        return cap_tool_output(result, contract.max_output_chars)
    return result


# Canonical tool contracts for the IT/Ops Change-Request & Incident Desk
TOOL_CONTRACTS: dict[str, ToolContract] = {
    "get_service_status": ToolContract(
        tool_id="get_service_status",
        version="1.0.0",
        read_write_class="read",
        timeout_s=10.0,
        max_output_chars=8000,
    ),
    "list_recent_deploys": ToolContract(
        tool_id="list_recent_deploys",
        version="1.0.0",
        read_write_class="read",
        timeout_s=10.0,
        max_output_chars=8000,
    ),
    "get_operation_status": ToolContract(
        tool_id="get_operation_status",
        version="1.0.0",
        read_write_class="read",
        timeout_s=10.0,
        max_output_chars=8000,
    ),
    "get_ticket": ToolContract(
        tool_id="get_ticket",
        version="1.0.0",
        read_write_class="read",
        timeout_s=10.0,
        max_output_chars=8000,
    ),
    "execute_service_change": ToolContract(
        tool_id="execute_service_change",
        version="1.0.0",
        read_write_class="write",
        timeout_s=10.0,
        max_output_chars=8000,
        idempotency_required=True,
        audit_required=True,
    ),
    "update_ticket": ToolContract(
        tool_id="update_ticket",
        version="1.0.0",
        read_write_class="write",
        timeout_s=10.0,
        max_output_chars=8000,
        idempotency_required=True,
        audit_required=True,
    ),
    "get_account": ToolContract(
        tool_id="get_account",
        version="1.0.0",
        read_write_class="read",
        timeout_s=10.0,
        max_output_chars=8000,
    ),
    "issue_refund": ToolContract(
        tool_id="issue_refund",
        version="1.0.0",
        read_write_class="write",
        timeout_s=10.0,
        max_output_chars=8000,
        idempotency_required=True,
        audit_required=True,
    ),
}
