"""Workflow release_version hash and checkpoint_compat constant for refund_ops."""

from __future__ import annotations

import hashlib
import inspect
import json
import os

from core import policy as policy_module
from core.tool_contract import TOOL_CONTRACTS
from workflows.spec_constants import (
    GOVERNED_GRAPH_EDGES,
    GOVERNED_GRAPH_NODES,
    PROPOSE_PLAN_PROMPT_TEMPLATE,
)

PACKAGE_SEMVER = "0.1.0"
CHECKPOINT_COMPAT = 1

INVESTIGATE_SYSTEM_PROMPT = (
    "You are a Billing & Refund Operations investigator. Inspect customer "
    "account status using read-only tools only. Never fabricate authorization "
    "or attempt refunds. Summarize findings concisely."
)

GRAPH_TOPOLOGY_SPEC = {
    "workflow": "refund_ops",
    "nodes": list(GOVERNED_GRAPH_NODES),
    "edges": [list(e) for e in GOVERNED_GRAPH_EDGES],
    "prompts": {
        "investigate": INVESTIGATE_SYSTEM_PROMPT,
        "propose_plan": PROPOSE_PLAN_PROMPT_TEMPLATE,
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
