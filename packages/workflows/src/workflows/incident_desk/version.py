"""Workflow release_version hash and checkpoint_compat constant (spec §5.5)."""

from __future__ import annotations

import hashlib
import inspect
import json
import os

from core import policy as policy_module
from core.tool_contract import TOOL_CONTRACTS

PACKAGE_SEMVER = "0.1.0"
CHECKPOINT_COMPAT = 1

GRAPH_TOPOLOGY_SPEC = {
    "nodes": [
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
    ],
    "prompts": {
        "investigate": (
            "You are an IT/Ops incident investigator. Use ONLY the provided read-only tools "
            "to inspect service health, recent deploys, and incident tickets. Never fabricate "
            "authorization or attempt writes."
        ),
        "propose_plan": (
            "Produce a structured ChangePlan grounded in the retrieved runbooks and tool outputs."
        ),
    },
}


def get_checkpoint_compat() -> int:
    override = os.getenv("CHECKPOINT_COMPAT_OVERRIDE")
    if override:
        return int(override)
    return CHECKPOINT_COMPAT


def compute_release_version() -> str:
    """Compute `<semver>+<12-hex sha256(topology + prompts + tool contracts + policy)>`."""
    payload = {
        "topology": GRAPH_TOPOLOGY_SPEC,
        "tools": {k: v.model_dump(mode="json") for k, v in sorted(TOOL_CONTRACTS.items())},
        "policy_source": inspect.getsource(policy_module),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"{PACKAGE_SEMVER}+{digest}"


RELEASE_VERSION = compute_release_version()
