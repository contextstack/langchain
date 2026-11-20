"""OpenTelemetry, self-hosted Langfuse v4, and LangSmith seam configuration (spec §9.1-9.2)."""

from __future__ import annotations

import base64
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter

from core.config import Settings, get_settings
from core.models import RunContext
from telemetry.masking import PIISpanProcessor, mask_export_payload


def build_trace_attributes(
    ctx: RunContext,
    *,
    workflow_version: str,
    operation_id: str | None = None,
) -> dict[str, str]:
    attrs = {
        "tenant.id": ctx.tenant_id,
        "actor.id": ctx.requester_id,
        "thread.id": ctx.thread_id,
        "run.id": ctx.run_id,
        "workflow.version": workflow_version,
    }
    if operation_id:
        attrs["operation.id"] = operation_id
    return attrs


def build_langfuse_otlp_headers(public_key: str, secret_key: str) -> dict[str, str]:
    creds = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
    return {
        "Authorization": f"Basic {creds}",
        "x-langfuse-ingestion-version": "4",
    }


def configure_tracer_provider(
    *,
    settings: Settings | None = None,
    exporter: SpanExporter | None = None,
    service_name: str = "contextstack-worker",
) -> TracerProvider:
    """Configure OTel TracerProvider with mandatory PIISpanProcessor for all telemetry modes."""
    cfg = settings or get_settings()
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(PIISpanProcessor())

    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        return provider

    if cfg.telemetry_backend == "langfuse_otel":
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
        os.environ["LANGSMITH_OTEL_ONLY"] = "true"
        endpoint = f"{cfg.langfuse_base_url.rstrip('/')}/api/public/otel/v1/traces"
        headers = build_langfuse_otlp_headers(cfg.langfuse_public_key, cfg.langfuse_secret_key)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
        )
    elif cfg.telemetry_backend == "langsmith":
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
        os.environ["LANGSMITH_OTEL_ONLY"] = "false"

    trace.set_tracer_provider(provider)
    return provider


def build_langchain_callbacks(settings: Settings | None = None) -> list[Any]:
    """Return LangChain callbacks for the active telemetry backend (e.g. langfuse_callback)."""
    cfg = settings or get_settings()
    if cfg.telemetry_backend != "langfuse_callback":
        return []
    try:
        from langfuse.langchain import CallbackHandler  # noqa: PLC0415

        return [CallbackHandler()]
    except Exception:
        return []


def prepare_langsmith_export_record(record: dict[str, Any]) -> dict[str, Any]:
    """Apply shared PII masking before LangSmith export (used by redaction hooks & AC-11 tests)."""
    masked = mask_export_payload(record)
    assert isinstance(masked, dict)
    return masked
