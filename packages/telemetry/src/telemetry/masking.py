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

    def _scrub_attrs(self, attrs: Any) -> None:
        if attrs is None:
            return
        backing = getattr(attrs, "_dict", attrs)
        if isinstance(backing, dict):
            for key in list(backing.keys()):
                backing[key] = _mask_attribute_value(backing[key])

    def _scrub_span(self, span: Span | ReadableSpan) -> None:
        raw_name = getattr(span, "_name", None)
        if isinstance(raw_name, str):
            span._name = redact_text(raw_name)  # type: ignore[union-attr]
        self._scrub_attrs(getattr(span, "_attributes", None))
        events = getattr(span, "_events", None) or ()
        for ev in events:
            ev_name = getattr(ev, "_name", None)
            if isinstance(ev_name, str):
                ev._name = redact_text(ev_name)
            self._scrub_attrs(getattr(ev, "_attributes", None))

    def on_end(self, span: ReadableSpan) -> None:
        self._scrub_span(span)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def mask_export_payload(payload: Any) -> Any:
    """Mask PII in payloads exported via Langfuse callback or LangSmith writers."""
    return redact_obj(payload)
