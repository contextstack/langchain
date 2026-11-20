"""Offline evaluation runner & CI regression gate (`uv run ops-cli eval`, AC-7)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from core.models import RunContext
from evals.evaluators.deterministic import (
    eval_end_state_matches,
    eval_idempotency_preserved,
    eval_injection_followed,
    eval_no_unauthorized_writes,
)
from evals.evaluators.judge import (
    compute_calibration_agreement,
    evaluate_triage_and_faithfulness,
)
from mcp_servers.domain_store import InMemoryDomainStore
from workflows.incident_desk.graph import WorkflowDeps, build_incident_desk_graph


async def run_eval_suite(
    dataset_path: Path,
    calibration_path: Path | None = None,
) -> dict[str, Any]:
    cases = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results: list[dict[str, Any]] = []
    all_passed = True

    for case in cases:
        store = InMemoryDomainStore()
        deps = WorkflowDeps(domain_store=store, use_postgres=False)
        graph = build_incident_desk_graph(deps=deps, checkpointer=InMemorySaver())
        ctx = RunContext(
            tenant_id=case["tenant_id"],
            requester_id=case["requester_id"],
            requester_roles=frozenset(case["roles"]),
            thread_id=f"eval-{case['case_id']}",
            run_id=f"run-{case['case_id']}",
            request_id=f"req-{case['case_id']}",
        )
        cfg = {"configurable": {"thread_id": ctx.thread_id}}

        final = await graph.ainvoke(
            {"request_text": case["request_text"]},
            config=cfg,
            context=ctx,
            durability="sync",
        )
        snap = await graph.aget_state(cfg)
        if snap.interrupts and case.get("decision") is not None:
            dec = dict(case["decision"])
            if dec["decision"] == "approve":
                dec["approved_plan_hash"] = snap.interrupts[0].value["plan_hash"]
            final = await graph.ainvoke(
                Command(resume=dec),
                config=cfg,
                context=ctx,
                durability="sync",
            )

        ok_end, msg_end = eval_end_state_matches(case=case, final_state=final, store=store)
        ok_auth, msg_auth = eval_no_unauthorized_writes(
            case=case,
            store=store,
            approvals=deps.approvals_memory,
            audits=deps.audit_memory,
        )
        ok_idem, msg_idem = eval_idempotency_preserved(case=case, store=store)
        inj_metric = eval_injection_followed(case=case, final_state=final)

        plan = final.get("change_plan")
        cited_ids = (
            [c.doc_id for c in plan.citations]
            if hasattr(plan, "citations")
            else [c["doc_id"] for c in (plan or {}).get("citations", [])]
        )
        retrieved_ids = [r["doc_id"] for r in final.get("retrieved_runbooks", [])]
        plan_summary = (
            plan.summary if hasattr(plan, "summary") else str((plan or {}).get("summary", ""))
        )
        judge_res = await evaluate_triage_and_faithfulness(
            summary=plan_summary,
            cited_doc_ids=cited_ids,
            retrieved_doc_ids=retrieved_ids,
        )

        case_passed = ok_end and ok_auth and ok_idem
        if not case_passed:
            all_passed = False

        results.append(
            {
                "case_id": case["case_id"],
                "passed": case_passed,
                "end_state": (ok_end, msg_end),
                "no_unauthorized_writes": (ok_auth, msg_auth),
                "idempotency": (ok_idem, msg_idem),
                "injection_followed": inj_metric,
                "citation_faithful": judge_res.faithful,
                "triage_score": judge_res.triage_score,
            }
        )

    calibration = (
        compute_calibration_agreement(calibration_path)
        if calibration_path and calibration_path.exists()
        else None
    )
    return {
        "passed": all_passed,
        "total": len(results),
        "cases": results,
        "calibration": calibration,
    }
