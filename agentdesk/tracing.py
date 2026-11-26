"""Structured tracing: every agent step becomes a JSON event with PII redacted."""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from typing import Any

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def redact(value: Any) -> Any:
    if isinstance(value, str):
        value = _EMAIL.sub("[email]", value)
        value = _IBAN.sub("[iban]", value)
        return _PHONE.sub("[phone]", value)
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    return value


class Tracer:
    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self._events: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def new_trace_id() -> str:
        return uuid.uuid4().hex[:12]

    def emit(self, trace_id: str, event: str, **fields: Any) -> None:
        record = {"trace_id": trace_id, "event": event, "ts": round(time.time(), 3), **redact(fields)}
        with self._lock:
            self._events.setdefault(trace_id, []).append(record)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get(self, trace_id: str) -> list[dict]:
        with self._lock:
            return list(self._events.get(trace_id, []))
