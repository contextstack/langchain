"""Shared PII patterns and redaction helpers (spec §4.2, §9.3).

Defined once in `core` and reused by both `PIIMiddleware` (model boundary)
and `packages/telemetry` (trace export boundary).
"""

from __future__ import annotations

import re
from typing import Any

EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
JWT_PATTERN = r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
SLACK_TOKEN_PATTERN = r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"
API_KEY_PATTERN = r"\b(?:sk|pk|rk|ghp|gho|AKIA)[A-Za-z0-9_\-]{12,}\b"
SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"
CREDIT_CARD_PATTERN = r"\b(?:\d{4}[\s-]?){3}\d{4}\b"
PHONE_PATTERN = (
    r"(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}"
    r"|(?<!\d)(?:\+1)?[2-9]\d{9}(?!\d)"
)

PII_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("email", re.compile(EMAIL_PATTERN), "[REDACTED_EMAIL]"),
    ("jwt", re.compile(JWT_PATTERN), "[REDACTED_JWT]"),
    ("slack_token", re.compile(SLACK_TOKEN_PATTERN), "[REDACTED_SLACK_TOKEN]"),
    ("api_key", re.compile(API_KEY_PATTERN), "[REDACTED_API_KEY]"),
    ("ssn", re.compile(SSN_PATTERN), "[REDACTED_SSN]"),
    ("credit_card", re.compile(CREDIT_CARD_PATTERN), "[REDACTED_CARD]"),
    ("phone_number", re.compile(PHONE_PATTERN), "[REDACTED_PHONE]"),
]


def redact_text(text: str) -> str:
    masked = text
    for _, pattern, replacement in PII_RULES:
        masked = pattern.sub(replacement, masked)
    return masked


def redact_obj(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_obj(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_obj(v) for v in value)
    if hasattr(value, "model_dump"):
        return redact_obj(value.model_dump(mode="json"))
    return value
