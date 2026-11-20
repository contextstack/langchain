"""Calibrated citation-faithfulness & triage-quality LLM-as-judge evaluators (spec §9.4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.config import Settings, get_settings
from gateway.factory import get_chat_model


class TriageJudgeResult(BaseModel, frozen=True):
    faithful: bool
    triage_score: int = Field(ge=1, le=5)
    rationale: str


def score_citation_faithfulness(
    *,
    cited_doc_ids: list[str],
    retrieved_doc_ids: list[str],
) -> bool:
    """Groundedness check: non-empty citations that are a subset of retrieved runbooks."""
    if not cited_doc_ids:
        return False
    return set(cited_doc_ids).issubset(set(retrieved_doc_ids))


def _heuristic_triage_score(
    *,
    summary: str,
    cited_doc_ids: list[str],
    retrieved_doc_ids: list[str],
) -> TriageJudgeResult:
    """Deterministic $0 rubric fallback used when MODEL_MODE=mock."""
    faithful = score_citation_faithfulness(
        cited_doc_ids=cited_doc_ids, retrieved_doc_ids=retrieved_doc_ids
    )
    lower = summary.lower()
    if not faithful:
        score = 1 if (cited_doc_ids and not set(cited_doc_ids).issubset(retrieved_doc_ids)) else 2
        return TriageJudgeResult(
            faithful=False,
            triage_score=score,
            rationale="Citations missing or not grounded in retrieved runbooks.",
        )
    has_actionable = any(w in lower for w in ("roll back", "rollback", "restart", "v2.4.0"))
    score = 5 if has_actionable else 4
    return TriageJudgeResult(
        faithful=True,
        triage_score=score,
        rationale="Grounded in retrieved runbooks with clear operational summary.",
    )


async def evaluate_triage_and_faithfulness(
    *,
    summary: str,
    cited_doc_ids: list[str],
    retrieved_doc_ids: list[str],
    settings: Settings | None = None,
) -> TriageJudgeResult:
    """Score faithfulness + triage quality (1-5) via LLM judge in gateway mode, rubric in mock."""
    cfg = settings or get_settings()
    if cfg.model_mode == "mock":
        return _heuristic_triage_score(
            summary=summary,
            cited_doc_ids=cited_doc_ids,
            retrieved_doc_ids=retrieved_doc_ids,
        )

    judge_model = get_chat_model(alias="ops-judge", settings=cfg)
    structured = judge_model.with_structured_output(TriageJudgeResult)
    return await structured.ainvoke(
        "Grade the IT/Ops incident plan on citation faithfulness (bool) and triage quality (1-5).\n"
        f"Summary: {summary}\n"
        f"Cited docs: {cited_doc_ids}\n"
        f"Retrieved docs: {retrieved_doc_ids}"
    )


def compute_calibration_agreement(calibration_path: Path) -> dict[str, Any]:
    """Compare faithfulness & triage-quality scoring against hand-labeled calibration.jsonl."""
    total = 0
    faithful_agreed = 0
    triage_within_1 = 0
    for line in calibration_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        res = _heuristic_triage_score(
            summary=row["summary"],
            cited_doc_ids=row["cited_doc_ids"],
            retrieved_doc_ids=row["retrieved_doc_ids"],
        )
        total += 1
        if res.faithful == bool(row["human_faithful"]):
            faithful_agreed += 1
        if abs(res.triage_score - int(row["human_triage_score"])) <= 1:
            triage_within_1 += 1
    rate = (faithful_agreed / total) if total else 0.0
    triage_rate = (triage_within_1 / total) if total else 0.0
    return {
        "samples": total,
        "agreed": faithful_agreed,
        "agreement_rate": round(rate, 3),
        "triage_within_1_rate": round(triage_rate, 3),
    }
