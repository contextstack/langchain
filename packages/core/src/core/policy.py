"""Deterministic PolicyEngine and separation-of-duties enforcement (spec §4.2, §5.2)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from core.models import CallerIdentity, ChangePlan, RunContext, ToolContract

DISALLOWED_ACTIONS = frozenset({"delete_cluster", "drop_database"})
ALLOWED_WRITE_ACTIONS = frozenset(
    {"rollback", "restart", "scale_up", "update_ticket", "issue_refund"}
)
MAX_REFUND_CENTS = 500_000  # $5,000 hard policy ceiling
REQUESTER_ROLES = frozenset({"engineer", "ops-approver", "admin"})
APPROVER_ROLES = frozenset({"ops-approver", "admin"})


class PolicyDecision(StrEnum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class PolicyResult(BaseModel, frozen=True):
    decision: PolicyDecision
    reason: str


class PolicyEngine:
    """Evaluates authorization outside the prompt ("Model proposes; platform authorizes")."""

    def evaluate_tool_call(
        self,
        *,
        ctx: RunContext,
        contract: ToolContract,
        target_tenant_id: str | None = None,
    ) -> PolicyResult:
        if target_tenant_id is not None and target_tenant_id != ctx.tenant_id:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=f"Cross-tenant access denied ({ctx.tenant_id} -> {target_tenant_id})",
            )

        if contract.read_write_class == "read":
            return PolicyResult(decision=PolicyDecision.ALLOW, reason="Read tool allowed in tenant")

        # In v1, no write or destructive tool is auto-allowed without approval.
        return PolicyResult(
            decision=PolicyDecision.REQUIRE_APPROVAL,
            reason=f"{contract.read_write_class} tool requires human approval in v1",
        )

    def evaluate_change_plan(
        self,
        *,
        ctx: RunContext,
        plan: ChangePlan,
    ) -> PolicyResult:
        if not (ctx.requester_roles & REQUESTER_ROLES):
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=f"Requester {ctx.requester_id} lacks an authorized requester role",
            )

        if plan.action == "none":
            return PolicyResult(decision=PolicyDecision.ALLOW, reason="No external write requested")

        # Reject any cross-tenant prefix smuggled into target_resource (e.g., "tenant-beta/db")
        if "/" in plan.target_resource:
            prefix = plan.target_resource.split("/", 1)[0]
            if prefix != ctx.tenant_id:
                return PolicyResult(
                    decision=PolicyDecision.DENY,
                    reason=f"Target resource '{plan.target_resource}' addresses another tenant",
                )

        if plan.action in DISALLOWED_ACTIONS:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=f"Action '{plan.action}' is prohibited by enterprise policy",
            )

        if plan.action not in ALLOWED_WRITE_ACTIONS:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=f"Unknown or unsupported action '{plan.action}'",
            )

        if plan.action == "issue_refund":
            if plan.amount_cents is None or plan.amount_cents <= 0:
                return PolicyResult(
                    decision=PolicyDecision.DENY,
                    reason="Refund amount_cents must be positive",
                )
            if plan.amount_cents > MAX_REFUND_CENTS:
                return PolicyResult(
                    decision=PolicyDecision.DENY,
                    reason=(
                        f"Refund {plan.amount_cents} cents exceeds policy ceiling "
                        f"({MAX_REFUND_CENTS} cents)"
                    ),
                )

        return PolicyResult(
            decision=PolicyDecision.REQUIRE_APPROVAL,
            reason=f"Write action '{plan.action}' on '{plan.target_resource}' requires approval",
        )

    def check_approval_authority(
        self,
        *,
        caller: CallerIdentity,
        approval_tenant_id: str,
        requested_by: str,
    ) -> PolicyResult:
        if caller.tenant_id != approval_tenant_id:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason="Tenant mismatch on approval decision",
            )
        if not (caller.roles & APPROVER_ROLES):
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=f"Caller '{caller.actor_id}' lacks ops-approver role",
            )
        if caller.actor_id == requested_by:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason="Separation of duties violation: requester cannot approve their own change",
            )
        return PolicyResult(decision=PolicyDecision.ALLOW, reason="Approver authorized")
