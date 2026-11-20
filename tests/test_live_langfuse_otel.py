"""Live self-hosted Langfuse v4 OTLP ingestion + PII masking verification (M0 #5, #8, AC-11).

Auto-skips cleanly when `--profile full` (Langfuse on localhost:3000) is not running.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import httpx
import pytest

from core.config import Settings
from core.models import RunContext
from telemetry.tracing import build_trace_attributes, configure_tracer_provider


async def _langfuse_healthy(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/public/health")
            return resp.status_code == 200
    except Exception:
        return False


async def test_live_langfuse_v4_otlp_ingestion_and_redaction() -> None:
    settings = Settings(
        telemetry_backend="langfuse_otel",
        langfuse_base_url="http://127.0.0.1:3000",
        langfuse_public_key="pk-lf-dev-local",
        langfuse_secret_key="sk-lf-dev-local",
    )
    if not await _langfuse_healthy(settings.langfuse_base_url):
        pytest.skip("Self-hosted Langfuse v4 (--profile full) is not running on localhost:3000")

    unique_run = f"run-live-otel-{uuid.uuid4().hex[:10]}"
    secret_email = "live.leak@enterprise.example"
    secret_phone = "415-555-0199"
    secret_key = "sk-live-9876543210abcdef1234"

    provider = configure_tracer_provider(settings=settings, service_name="live-otel-smoke")
    tracer = provider.get_tracer("live-smoke")
    ctx = RunContext(
        tenant_id="tenant-alpha",
        requester_id="alice",
        requester_roles=frozenset({"engineer"}),
        thread_id=f"thr-{unique_run}",
        run_id=unique_run,
        request_id=f"req-{unique_run}",
    )
    attrs = build_trace_attributes(ctx, workflow_version="0.1.0+live", operation_id="op-live-1")
    with tracer.start_as_current_span(f"incident_desk.{unique_run}", attributes=attrs) as span:
        span.set_attribute(
            "input.value",
            f"Live OTLP test for {secret_email} / {secret_phone} / {secret_key}",
        )
        span.set_attribute("output.value", "Completed safely")

    assert provider.force_flush(timeout_millis=10000) is True

    # Poll Langfuse v4 v2 observations API for the ingested OTLP span (ClickHouse async worker)
    deadline = time.monotonic() + 20.0
    matched_obs: dict[str, object] | None = None
    auth = (settings.langfuse_public_key, settings.langfuse_secret_key)

    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            resp = await client.get(
                f"{settings.langfuse_base_url}/api/public/v2/observations",
                params={"limit": 20, "fields": "core,io,metadata"},
                auth=auth,
            )
            assert resp.status_code == 200, resp.text
            for item in resp.json().get("data", []):
                if unique_run in str(item):
                    matched_obs = item
                    break
            if matched_obs is not None:
                break
            await asyncio.sleep(1.0)

    assert matched_obs is not None, f"OTLP span for {unique_run} did not appear in Langfuse v4"
    serialized = str(matched_obs)
    assert "[REDACTED_EMAIL]" in serialized
    assert "[REDACTED_PHONE]" in serialized
    assert "[REDACTED_API_KEY]" in serialized
    for secret in (secret_email, secret_phone, secret_key):
        assert secret not in serialized
