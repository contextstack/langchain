"""Workflow release_version hash and checkpoint_compat for incident_desk."""

from __future__ import annotations

from core.workflow_catalog import WORKFLOW_CATALOG

_META = WORKFLOW_CATALOG["incident_desk"]
PACKAGE_SEMVER = "0.1.0"
CHECKPOINT_COMPAT = _META.checkpoint_compat
INVESTIGATE_SYSTEM_PROMPT = _META.investigate_prompt
GRAPH_TOPOLOGY_SPEC = _META.topology_spec
RELEASE_VERSION = _META.release_version


def get_checkpoint_compat() -> int:
    return _META.current_checkpoint_compat()
