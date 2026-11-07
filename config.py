from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class Settings:
    """Runtime settings. All values can be overridden with environment variables."""

    refund_auto_limit: float = 50.0
    refund_window_days: int = 30
    max_steps: int = 6
    today: date = field(default_factory=date.today)
    trace_path: str | None = None
    reviewer_key: str = "demo-reviewer"


def load_settings() -> Settings:
    today = os.getenv("AGENTDESK_TODAY")
    return Settings(
        refund_auto_limit=float(os.getenv("REFUND_AUTO_LIMIT", "50")),
        refund_window_days=int(os.getenv("REFUND_WINDOW_DAYS", "30")),
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "6")),
        today=date.fromisoformat(today) if today else date.today(),
        trace_path=os.getenv("TRACE_PATH") or None,
        reviewer_key=os.getenv("REVIEWER_KEY", "demo-reviewer"),
    )
