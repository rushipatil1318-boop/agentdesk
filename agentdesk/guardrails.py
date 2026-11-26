"""Policy enforcement point: every tool call is checked here before it runs.

Design choice: the LLM *proposes*, the guard *decides*. Rules are plain Python so they can be
unit-tested, versioned and audited, instead of hoping a prompt keeps the model in line.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Any

from .config import Settings

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above)\s+(instructions|rules)",
    r"disregard\s+(the\s+|all\s+|any\s+)?(.*?)(policy|policies|rules|instructions)",
    r"you\s+are\s+now\s+(in\s+)?(admin|developer|system|dan)",
    r"\b(admin|developer|system)\s+mode\b",
    r"reveal\s+(your\s+|the\s+)?(system\s+)?prompt",
    r"without\s+(any\s+)?approval",
    r"override\s+(the\s+)?(policy|policies|limit)",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.I)
SAFE_REFUSAL = (
    "I can't follow instructions that try to change or bypass our support rules. "
    "I can still help with order status, refunds within policy, or connect you with a human agent."
)


@dataclass(frozen=True)
class Decision:
    verdict: str  # "allow" | "deny" | "require_approval"
    rule: str
    reason: str = ""


@dataclass(frozen=True)
class InputCheck:
    flagged: bool
    reason: str = ""


class Guard:
    def __init__(self, conn: sqlite3.Connection, settings: Settings, enabled: bool = True,
                 allowed_tools: set[str] | None = None) -> None:
        self.conn = conn
        self.settings = settings
        self.enabled = enabled
        self.allowed_tools = allowed_tools

    # --- input ---------------------------------------------------------------
    def check_input(self, text: str) -> InputCheck:
        if _INJECTION_RE.search(text):
            return InputCheck(True, "possible prompt injection")
        return InputCheck(False)

    # --- tool calls ----------------------------------------------------------
    def _order(self, order_id: Any) -> sqlite3.Row | None:
        if not order_id:
            return None
        return self.conn.execute("SELECT * FROM orders WHERE id = ?", (str(order_id).upper(),)).fetchone()

    def check_tool(self, name: str, args: dict, customer_id: str) -> Decision:
        if self.allowed_tools is not None and name not in self.allowed_tools:
            return Decision("deny", "tool.not_allowed", "That action is not available.")
        order = self._order(args.get("order_id"))
        if order is not None and order["customer_id"] != customer_id:
            return Decision("deny", "authz.order_owner", "You can only access your own orders.")
        if name != "issue_refund":
            return Decision("allow", "default.allow")
        return self._check_refund(order, args)

    def _check_refund(self, order: sqlite3.Row | None, args: dict) -> Decision:
        if order is None:
            return Decision("deny", "refund.order_not_found", "I couldn't find that order.")
        if order["status"] != "delivered" or not order["delivered_on"]:
            return Decision("deny", "refund.not_delivered",
                            "Refunds are only possible after the order has been delivered.")
        try:
            amount = float(args.get("amount", 0))
        except (TypeError, ValueError):
            return Decision("deny", "refund.invalid_amount", "The refund amount is not valid.")
        if amount <= 0:
            return Decision("deny", "refund.invalid_amount", "The refund amount must be positive.")
        if amount > float(order["total"]) + 1e-9:
            return Decision("deny", "refund.amount_exceeds_total",
                            "A refund cannot be higher than the amount paid.")
        age = (self.settings.today - date.fromisoformat(order["delivered_on"])).days
        if age > self.settings.refund_window_days:
            return Decision("deny", "refund.window_expired",
                            f"The {self.settings.refund_window_days}-day refund window has passed.")
        existing = self.conn.execute(
            "SELECT 1 FROM refunds WHERE order_id = ? AND status IN ('issued', 'pending_approval')",
            (order["id"],),
        ).fetchone()
        if existing:
            return Decision("deny", "refund.duplicate", "This order already has a refund.")
        if amount > self.settings.refund_auto_limit:
            return Decision("require_approval", "refund.above_auto_limit",
                            "Refunds above the automatic limit need human approval.")
        return Decision("allow", "refund.within_policy")

    # --- tool output ---------------------------------------------------------
    def sanitize_output(self, value: Any) -> Any:
        """Strip instruction-like text from tool results (indirect prompt injection)."""
        if isinstance(value, str):
            return "[removed: possible prompt injection]" if _INJECTION_RE.search(value) else value
        if isinstance(value, list):
            return [self.sanitize_output(v) for v in value]
        if isinstance(value, dict):
            return {k: self.sanitize_output(v) for k, v in value.items()}
        return value
