"""True end-to-end tests for MODEL_MODE=gateway (over real HTTP) and `ops-cli` live flow.

1. `test_e2e_gateway_mode_api_worker_openai_stub_and_mcp_http`:
   Boots a local OpenAI-compatible HTTP stub (`/v1/embeddings` + `/v1/chat/completions`),
   live FastAPI, live FastMCP Infra server, and live `ops-worker` with `MODEL_MODE=gateway`
   and `MCP_TRANSPORT=http`. Verifies:
     - Real `init_chat_model` tool-calling (`create_agent`) and `with_structured_output(ChangePlan)`
     - Tenant-scoped virtual API key (`LITELLM_KEY_TENANT_ALPHA`) sent on both embeddings & chat
     - Zero raw PII (`email`, `phone`, `api_key`) in ANY outgoing HTTP payload to the model stub
     - Full submit -> pause -> approve -> authenticated MCP HTTP rollback -> verified end state

2. `test_e2e_ops_cli_run_and_approve_against_live_stack`:
   Runs the real `ops-cli run` and `ops-cli approve` commands against live FastAPI + FastMCP
   Infra + `ops-worker` (`MCP_TRANSPORT=http`) and verifies the rolled-back service state.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from typing import Any

import httpx
import psycopg
import uvicorn
from fastapi import FastAPI, Request
from helpers import free_port, poll_run, stop_process
from typer.testing import CliRunner

from api.main import create_app
from cli.main import app as cli_app
from core.auth import mint_mcp_token
from core.config import get_settings
from core.db import apply_app_migrations
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain
from mcp_servers.domain_store import PostgresDomainStore
from mcp_servers.infra_server import create_infra_server
from workflows.mcp_client import call_mcp_tool


def _create_openai_stub_app(
    captured_requests: list[dict[str, Any]],
) -> FastAPI:
    stub = FastAPI()
    embedder = DeterministicMockEmbeddings()

    @stub.post("/v1/embeddings")
    async def embeddings_endpoint(request: Request) -> dict[str, Any]:
        body = await request.json()
        captured_requests.append(
            {
                "path": "/v1/embeddings",
                "auth": request.headers.get("authorization", ""),
                "body": body,
            }
        )
        raw_input = body.get("input", "")
        inputs = raw_input if isinstance(raw_input, list) else [raw_input]
        data = []
        for idx, item in enumerate(inputs):
            vec = embedder.embed_query(str(item))
            data.append({"object": "embedding", "index": idx, "embedding": vec})
        return {
            "object": "list",
            "data": data,
            "model": body.get("model", "ops-embed"),
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

    @stub.post("/v1/chat/completions")
    async def chat_completions_endpoint(request: Request) -> dict[str, Any]:
        body = await request.json()
        captured_requests.append(
            {
                "path": "/v1/chat/completions",
                "auth": request.headers.get("authorization", ""),
                "body": body,
            }
        )
        messages = body.get("messages", [])
        tools = body.get("tools") or []
        tool_names = [
            (t.get("function") or {}).get("name", "") for t in tools if isinstance(t, dict)
        ]

        # 1. Structured ChangePlan call (response_format json_schema OR tool binding)
        rf = body.get("response_format") or {}
        is_plan_schema = (
            rf.get("type") == "json_schema"
            and (rf.get("json_schema") or {}).get("name") == "ChangePlan"
        )
        if is_plan_schema or "ChangePlan" in tool_names:
            plan_args = {
                "resource_type": "service",
                "target_resource": "checkout-api",
                "action": "rollback",
                "target_version": "v2.4.0",
                "rollback_plan": "Re-deploy v2.4.1 if needed.",
                "risk_level": "medium",
                "summary": "Roll back checkout-api from v2.4.1 to v2.4.0.",
                "citations": [
                    {
                        "doc_id": "rb-alpha-checkout-502",
                        "title": "Checkout API 502 Post-Deploy Rollback Runbook",
                    }
                ],
                "ticket_id": "INC-1001",
            }
            if is_plan_schema:
                return {
                    "id": "chatcmpl-plan-json",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": body.get("model", "ops-primary"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(plan_args),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40},
                }
            return {
                "id": "chatcmpl-plan",
                "object": "chat.completion",
                "created": 1700000000,
                "model": body.get("model", "ops-primary"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_plan_1",
                                    "type": "function",
                                    "function": {
                                        "name": "ChangePlan",
                                        "arguments": json.dumps(plan_args),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40},
            }

        # 2. create_agent investigate loop: first turn calls get_service_status; second summarizes
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if "get_service_status" in tool_names and not has_tool_result:
            return {
                "id": "chatcmpl-tool",
                "object": "chat.completion",
                "created": 1700000000,
                "model": body.get("model", "ops-primary"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_svc_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_service_status",
                                        "arguments": json.dumps({"service_name": "checkout-api"}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }

        return {
            "id": "chatcmpl-summary",
            "object": "chat.completion",
            "created": 1700000000,
            "model": body.get("model", "ops-primary"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            "checkout-api is degraded on v2.4.1; rollback to v2.4.0 advised."
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        }

    return stub


async def test_e2e_gateway_mode_api_worker_openai_stub_and_mcp_http() -> None:
    settings = get_settings()
    await apply_app_migrations(settings.database_admin_url)
    async with await psycopg.AsyncConnection.connect(
        settings.database_admin_url, autocommit=True
    ) as conn:
        await conn.execute(
            "TRUNCATE TABLE runs, approvals, app_threads, mock_operations RESTART IDENTITY CASCADE;"
        )
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        await seed_knowledge_and_domain(conn, embeddings=DeterministicMockEmbeddings())

    captured: list[dict[str, Any]] = []
    pg_store = PostgresDomainStore(settings.database_url)
    stub_port, infra_port, api_port = free_port(), free_port(), free_port()
    stub_url = f"http://127.0.0.1:{stub_port}/v1"
    infra_url = f"http://127.0.0.1:{infra_port}/mcp"
    api_base = f"http://127.0.0.1:{api_port}"

    stub_srv = uvicorn.Server(
        uvicorn.Config(
            _create_openai_stub_app(captured), host="127.0.0.1", port=stub_port, log_level="error"
        )
    )
    infra_srv = uvicorn.Server(
        uvicorn.Config(
            create_infra_server(store=pg_store, mcp_secret=settings.mcp_token_secret).http_app(
                path="/mcp"
            ),
            host="127.0.0.1",
            port=infra_port,
            log_level="error",
        )
    )
    api_srv = uvicorn.Server(
        uvicorn.Config(create_app(settings), host="127.0.0.1", port=api_port, log_level="error")
    )
    tasks = [
        asyncio.create_task(stub_srv.serve()),
        asyncio.create_task(infra_srv.serve()),
        asyncio.create_task(api_srv.serve()),
    ]
    worker: subprocess.Popen[bytes] | None = None
    try:
        while not (stub_srv.started and infra_srv.started and api_srv.started):
            await asyncio.sleep(0.05)

        worker_env = {
            **os.environ,
            "MODEL_MODE": "gateway",
            "LITELLM_URL": stub_url,
            "LITELLM_KEY_TENANT_ALPHA": "sk-alpha-e2e-virtual-key",
            "MCP_TRANSPORT": "http",
            "MCP_INFRA_URL": infra_url,
            "WORKER_LEASE_S": "5",
            "WORKER_HEARTBEAT_S": "2",
        }
        worker = subprocess.Popen(
            [sys.executable, "-c", "from worker.main import run; run()"],
            env=worker_env,
        )

        raw_email = "alice.secret@corp.example"
        raw_phone = "415-555-0199"
        raw_key = "sk-live-9876543210abcdef1234"

        async with httpx.AsyncClient(timeout=20.0) as client:
            alice_tok = (
                await client.post(f"{api_base}/v1/dev/token", json={"persona": "alice"})
            ).json()["token"]
            bob_tok = (
                await client.post(f"{api_base}/v1/dev/token", json={"persona": "bob"})
            ).json()["token"]
            alice_h = {"Authorization": f"Bearer {alice_tok}"}
            bob_h = {"Authorization": f"Bearer {bob_tok}"}

            t_id = (
                await client.post(
                    f"{api_base}/v1/threads",
                    headers=alice_h,
                    json={"workflow_name": "incident_desk"},
                )
            ).json()["thread_id"]
            r_id = (
                await client.post(
                    f"{api_base}/v1/threads/{t_id}/runs",
                    headers=alice_h,
                    json={
                        "request_text": (
                            f"checkout-api 502s after v2.4.1 deploy; "
                            f"contact {raw_email} / {raw_phone} / {raw_key}"
                        )
                    },
                )
            ).json()["run_id"]
            await poll_run(client, api_base, alice_h, r_id, {"awaiting_approval"})

            # Verify real HTTP calls to both /v1/embeddings and /v1/chat/completions
            paths = {c["path"] for c in captured}
            assert "/v1/embeddings" in paths
            assert "/v1/chat/completions" in paths
            # Verify tenant-scoped virtual key used on EVERY outbound model-gateway call
            assert all(c["auth"] == "Bearer sk-alpha-e2e-virtual-key" for c in captured)
            # Verify ZERO raw PII crossed the model-provider HTTP boundary
            wire_dump = json.dumps(captured)
            assert raw_email not in wire_dump
            assert raw_phone not in wire_dump
            assert raw_key not in wire_dump

            # Approve via Bob and verify authenticated MCP HTTP rollback
            approvals = (await client.get(f"{api_base}/v1/approvals", headers=bob_h)).json()
            apr = next(a for a in approvals if a["thread_id"] == t_id)
            dec = await client.post(
                f"{api_base}/v1/approvals/{apr['approval_id']}/decision",
                headers=bob_h,
                json={"decision": "approve"},
            )
            assert dec.status_code == 202
            res_state = await poll_run(
                client, api_base, bob_h, dec.json()["run_id"], {"succeeded", "failed"}
            )
            assert res_state["status"] == "succeeded", res_state

            svc = await call_mcp_tool(
                url=infra_url,
                bearer_token=mint_mcp_token(
                    tenant_id="tenant-alpha",
                    requester_id="bob",
                    run_id=dec.json()["run_id"],
                    secret=settings.mcp_token_secret,
                ),
                tool_name="get_service_status",
                arguments={"service_name": "checkout-api"},
            )
            assert svc["current_version"] == "v2.4.0"
            assert svc["status"] == "healthy"
    finally:
        stop_process(worker)
        stub_srv.should_exit = True
        infra_srv.should_exit = True
        api_srv.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_e2e_ops_cli_run_and_approve_against_live_stack() -> None:
    """Exercise `ops-cli run` -> live worker pause -> `ops-cli approve` -> verified MCP write."""
    settings = get_settings()
    await apply_app_migrations(settings.database_admin_url)
    async with await psycopg.AsyncConnection.connect(
        settings.database_admin_url, autocommit=True
    ) as conn:
        await conn.execute(
            "TRUNCATE TABLE runs, approvals, app_threads, mock_operations RESTART IDENTITY CASCADE;"
        )
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        await seed_knowledge_and_domain(conn, embeddings=DeterministicMockEmbeddings())

    pg_store = PostgresDomainStore(settings.database_url)
    infra_port, api_port = free_port(), free_port()
    infra_url = f"http://127.0.0.1:{infra_port}/mcp"
    api_base = f"http://127.0.0.1:{api_port}"

    infra_srv = uvicorn.Server(
        uvicorn.Config(
            create_infra_server(store=pg_store, mcp_secret=settings.mcp_token_secret).http_app(
                path="/mcp"
            ),
            host="127.0.0.1",
            port=infra_port,
            log_level="error",
        )
    )
    api_srv = uvicorn.Server(
        uvicorn.Config(create_app(settings), host="127.0.0.1", port=api_port, log_level="error")
    )
    tasks = [asyncio.create_task(infra_srv.serve()), asyncio.create_task(api_srv.serve())]
    worker: subprocess.Popen[bytes] | None = None
    try:
        while not (infra_srv.started and api_srv.started):
            await asyncio.sleep(0.05)

        worker = subprocess.Popen(
            [sys.executable, "-c", "from worker.main import run; run()"],
            env={
                **os.environ,
                "MODEL_MODE": "mock",
                "MCP_TRANSPORT": "http",
                "MCP_INFRA_URL": infra_url,
                "WORKER_LEASE_S": "5",
                "WORKER_HEARTBEAT_S": "2",
            },
        )
        runner = CliRunner()

        # 1. Submit via `ops-cli run`
        res_run = await asyncio.to_thread(
            runner.invoke,
            cli_app,
            ["run", "checkout-api returning 502s after v2.4.1 deploy", "--api-url", api_base],
        )
        assert res_run.exit_code == 0, res_run.output
        run_payload = json.loads(res_run.stdout)
        run_id = run_payload["run_id"]
        thread_id = run_payload["thread_id"]

        async with httpx.AsyncClient(timeout=15.0) as client:
            bob_tok = (
                await client.post(f"{api_base}/v1/dev/token", json={"persona": "bob"})
            ).json()["token"]
            bob_h = {"Authorization": f"Bearer {bob_tok}"}
            await poll_run(client, api_base, bob_h, run_id, {"awaiting_approval"})

            approvals = (await client.get(f"{api_base}/v1/approvals", headers=bob_h)).json()
            apr = next(a for a in approvals if a["thread_id"] == thread_id)

            # 2. Approve via `ops-cli approve`
            res_apr = await asyncio.to_thread(
                runner.invoke,
                cli_app,
                [
                    "approve",
                    "--approval",
                    apr["approval_id"],
                    "--decision",
                    "approve",
                    "--api-url",
                    api_base,
                ],
            )
            assert res_apr.exit_code == 0, res_apr.output
            assert "HTTP 202" in res_apr.stdout
            resume_json = json.loads(res_apr.stdout.split("HTTP 202: ", 1)[1])

            final_state = await poll_run(
                client, api_base, bob_h, resume_json["run_id"], {"succeeded", "failed"}
            )
            assert final_state["status"] == "succeeded", final_state

            svc = await call_mcp_tool(
                url=infra_url,
                bearer_token=mint_mcp_token(
                    tenant_id="tenant-alpha",
                    requester_id="bob",
                    run_id=resume_json["run_id"],
                    secret=settings.mcp_token_secret,
                ),
                tool_name="get_service_status",
                arguments={"service_name": "checkout-api"},
            )
            assert svc["current_version"] == "v2.4.0"
            assert svc["status"] == "healthy"
    finally:
        stop_process(worker)
        infra_srv.should_exit = True
        api_srv.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)
