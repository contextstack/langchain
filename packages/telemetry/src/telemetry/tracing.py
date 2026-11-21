"""OpenTelemetry, self-hosted Langfuse v4, and LangSmith seam configuration (spec §9.1-9.3)."""

from __future__ import annotations

import base64
import os
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter

from core.config import Settings, get_settings
from core.models import RunContext
from core.pii import redact_text
from telemetry.masking import PIISpanProcessor, mask_export_payload


def _redact_error(error: BaseException) -> BaseException:
    masked = redact_text(str(error))
    try:
        return type(error)(masked)
    except Exception:
        return RuntimeError(masked)


def _redact_llm_result(response: Any) -> Any:
    if not isinstance(response, LLMResult):
        return mask_export_payload(response)
    masked_gens: list[list[Any]] = []
    for gen_list in response.generations:
        row: list[Any] = []
        for g in gen_list:
            if hasattr(g, "message") and getattr(g, "message", None) is not None:
                msg = g.message
                new_content = mask_export_payload(getattr(msg, "content", ""))
                new_msg = msg.model_copy(update={"content": new_content})
                row.append(
                    g.model_copy(
                        update={"text": redact_text(getattr(g, "text", "")), "message": new_msg}
                    )
                )
            elif isinstance(g, Generation):
                row.append(g.model_copy(update={"text": redact_text(g.text)}))
            else:
                row.append(mask_export_payload(g))
        masked_gens.append(row)
    return LLMResult(
        generations=masked_gens,
        llm_output=mask_export_payload(response.llm_output),
        run=response.run,
    )


class RedactingCallbackProxy(BaseCallbackHandler):
    """Wraps a LangChain callback handler and masks PII before forwarding events."""

    def __init__(self, inner: BaseCallbackHandler) -> None:
        super().__init__()
        self._inner = inner

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        return self._inner.on_chain_start(
            mask_export_payload(serialized),
            mask_export_payload(inputs),
            **kwargs,
        )

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> Any:
        return self._inner.on_chain_end(mask_export_payload(outputs), **kwargs)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> Any:
        return self._inner.on_llm_start(
            mask_export_payload(serialized),
            mask_export_payload(prompts),
            **kwargs,
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> Any:
        return self._inner.on_tool_start(
            mask_export_payload(serialized),
            mask_export_payload(input_str),
            **kwargs,
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        return self._inner.on_tool_end(mask_export_payload(output), **kwargs)

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        return self._inner.on_llm_end(_redact_llm_result(response), **kwargs)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        return self._inner.on_llm_error(_redact_error(error), **kwargs)

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> Any:
        return self._inner.on_chain_error(_redact_error(error), **kwargs)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> Any:
        return self._inner.on_tool_error(_redact_error(error), **kwargs)


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
        # Keep LANGSMITH_TRACING=false so LangChain's CallbackManager does NOT
        # auto-install a second, unmasked default LangChainTracer alongside our
        # PII-redacting LangChainTracer from build_langchain_callbacks().
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGSMITH_OTEL_ENABLED"] = "false"
        os.environ["LANGSMITH_OTEL_ONLY"] = "false"

    trace.set_tracer_provider(provider)
    return provider


def build_langchain_callbacks(settings: Settings | None = None) -> list[BaseCallbackHandler]:
    """Return PII-masked LangChain callbacks for the active telemetry backend."""
    cfg = settings or get_settings()
    if cfg.telemetry_backend == "langfuse_callback":
        try:
            from langfuse.langchain import CallbackHandler  # noqa: PLC0415

            return [RedactingCallbackProxy(CallbackHandler())]
        except Exception:
            return []
    if cfg.telemetry_backend == "langsmith":
        try:
            from langchain_core.tracers import LangChainTracer  # noqa: PLC0415
            from langsmith import Client  # noqa: PLC0415

            client = Client(
                hide_inputs=mask_export_payload,
                hide_outputs=mask_export_payload,
            )
            return [RedactingCallbackProxy(LangChainTracer(client=client))]
        except Exception:
            return []
    return []


def prepare_langsmith_export_record(record: dict[str, Any]) -> dict[str, Any]:
    """Apply shared PII masking before LangSmith export."""
    masked = mask_export_payload(record)
    assert isinstance(masked, dict)
    return masked
