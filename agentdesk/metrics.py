"""Minimal in-process metrics with a Prometheus text endpoint (no extra dependency)."""
from __future__ import annotations

import threading
from collections import Counter, deque


class Metrics:
    def __init__(self, latency_window: int = 1000) -> None:
        self._counters: Counter = Counter()
        self._latencies: deque = deque(maxlen=latency_window)
        self._lock = threading.Lock()

    def inc(self, name: str, **labels: str) -> None:
        with self._lock:
            self._counters[(name, tuple(sorted(labels.items())))] += 1

    def observe_latency(self, ms: float) -> None:
        with self._lock:
            self._latencies.append(ms)

    @staticmethod
    def _quantile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 2)

    def snapshot(self) -> dict:
        with self._lock:
            counters = {}
            for (name, labels), value in self._counters.items():
                key = name + ("".join(f"[{k}={v}]" for k, v in labels))
                counters[key] = value
            lat = list(self._latencies)
        return {"counters": counters,
                "latency_ms": {"count": len(lat), "p50": self._quantile(lat, 0.5),
                               "p95": self._quantile(lat, 0.95)}}

    def render_prometheus(self) -> str:
        with self._lock:
            items = list(self._counters.items())
            lat = list(self._latencies)
        lines = ["# TYPE agentdesk_events_total counter"]
        for (name, labels), value in sorted(items):
            label_text = ",".join(f'{k}="{v}"' for k, v in labels)
            lines.append(f'agentdesk_{name}_total{{{label_text}}} {value}' if label_text
                         else f"agentdesk_{name}_total {value}")
        lines.append("# TYPE agentdesk_chat_latency_ms summary")
        for q in (0.5, 0.95):
            lines.append(f'agentdesk_chat_latency_ms{{quantile="{q}"}} {self._quantile(lat, q)}')
        lines.append(f"agentdesk_chat_latency_ms_count {len(lat)}")
        return "\n".join(lines) + "\n"
