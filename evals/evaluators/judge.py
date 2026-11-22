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
    retrieved_runbooks: list[dict[str, Any]],
    tool_observations: list[str] | None = None,
    settings: Settings | None = None,
) -> TriageJudgeResult:
    """Score faithfulness + triage quality (1-5) against runbooks and tool outputs via LLM judge."""
    cfg = settings or get_settings()
    retrieved_ids = [str(r["doc_id"]) for r in retrieved_runbooks]
    if cfg.model_mode == "mock":
        return _heuristic_triage_score(
            summary=summary,
            cited_doc_ids=cited_doc_ids,
            retrieved_doc_ids=retrieved_ids,
        )

    judge_model = get_chat_model(alias="ops-judge", settings=cfg)
    structured = judge_model.with_structured_output(TriageJudgeResult)
    runbook_excerpts = "\n\n".join(
        f"[{r['doc_id']}] {r.get('title', '')}\n{r.get('content', '')}" for r in retrieved_runbooks
    )
    obs_block = ""
    if tool_observations:
        obs_lines = "\n".join(f"- {obs}" for obs in tool_observations)
        obs_block = f"\n\nRead-Only Tool Observations:\n{obs_lines}"
    return await structured.ainvoke(
        "Grade the IT/Ops or Billing plan on:\n"
        "1. faithful (bool): Return true when Cited Doc IDs is non-empty, every cited doc_id "
        "appears in Retrieved Runbooks, and the Plan Summary is consistent with (does not "
        "contradict or fabricate beyond) the cited runbook text and/or Read-Only Tool "
        "Observations. A concise summary is still faithful even if it omits secondary runbook "
        "details. Return false if citations are empty, cite a nonexistent doc_id, or make "
        "claims that contradict the runbook policy.\n"
        "2. triage_score (1-5): Operational clarity and actionability (5 = clear, faithful, "
        "actionable; 4 = clear, faithful read-only/no-action summary; 1-2 = unfaithful, "
        "contradictory, or missing citations).\n\n"
        f"Plan Summary: {summary}\n"
        f"Cited Doc IDs: {cited_doc_ids}\n\n"
        f"Retrieved Runbooks:\n{runbook_excerpts}"
        f"{obs_block}"
    )


async def compute_calibration_agreement(
    calibration_path: Path,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Compare active judge (LLM in gateway mode, rubric in mock) against calibration.jsonl."""
    total = 0
    faithful_agreed = 0
    triage_within_1 = 0
    for line in calibration_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        runbooks = row.get("retrieved_runbooks") or [
            {"doc_id": d, "title": "", "content": ""} for d in row.get("retrieved_doc_ids", [])
        ]
        res = await evaluate_triage_and_faithfulness(
            summary=row["summary"],
            cited_doc_ids=row["cited_doc_ids"],
            retrieved_runbooks=runbooks,
            settings=settings,
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
