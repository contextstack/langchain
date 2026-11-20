"""Calibrated citation-faithfulness & triage-quality evaluators (spec §9.4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def score_citation_faithfulness(
    *,
    cited_doc_ids: list[str],
    retrieved_doc_ids: list[str],
) -> bool:
    """Groundedness check: non-empty citations that are a subset of retrieved runbooks."""
    if not cited_doc_ids:
        return False
    return set(cited_doc_ids).issubset(set(retrieved_doc_ids))


def compute_calibration_agreement(calibration_path: Path) -> dict[str, Any]:
    """Compare automated faithfulness judge against hand-labeled calibration.jsonl."""
    total = 0
    agreed = 0
    for line in calibration_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pred = score_citation_faithfulness(
            cited_doc_ids=row["cited_doc_ids"],
            retrieved_doc_ids=row["retrieved_doc_ids"],
        )
        total += 1
        if pred == bool(row["human_faithful"]):
            agreed += 1
    rate = (agreed / total) if total else 0.0
    return {"samples": total, "agreed": agreed, "agreement_rate": round(rate, 3)}
