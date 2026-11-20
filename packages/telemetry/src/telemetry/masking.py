"""PII span masking processor and callback payload redactor (spec §9.3, AC-11)."""

from __future__ import annotations

from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from core.pii import redact_obj, redact_text


def _mask_attribute_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, (list, tuple)):
        return type(value)(redact_text(v) if isinstance(v, str) else v for v in value)
    return value


class PIISpanProcessor(SpanProcessor):
    """OTel SpanProcessor that scrubs PII from span attributes before export."""

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        self._scrub_span(span)

    def _scrub_span(self, span: Span | ReadableSpan) -> None:
        attrs = getattr(span, "_attributes", None)
        if attrs is None:
            return
        backing = getattr(attrs, "_dict", attrs)
        if isinstance(backing, dict):
            for key in list(backing.keys()):
                backing[key] = _mask_attribute_value(backing[key])

    def on_end(self, span: ReadableSpan) -> None:
        self._scrub_span(span)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def mask_export_payload(payload: Any) -> Any:
    """Mask PII in payloads exported via Langfuse callback or LangSmith writers."""
    return redact_obj(payload)
