"""Tools the agent may call. They are intentionally thin: the policy decision lives in the guard,
which sits in front of every call (a single enforcement point that is easy to test and audit)."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Callable

from .config import Settings
from .db import add_refund
from .knowledge import PolicyKnowledgeBase


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    risk: str  # "low" | "high"


SPECS = [
    ToolSpec(
        "lookup_order",
        "Look up the status and details of an order by its order id (format ORD-1234).",
        {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
        "low",
    ),
    ToolSpec(
        "search_policy",
        "Search the support policy documents and return the most relevant passages.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "low",
    ),
    ToolSpec(
        "issue_refund",
        "Refund an order (full or partial). High risk: money leaves the company.",
        {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["order_id", "amount", "reason"],
        },
        "high",
    ),
    ToolSpec(
        "escalate_to_human",
        "Create a ticket for a human support agent when the request cannot be handled automatically.",
        {
            "type": "object",
            "properties": {"summary": {"type": "string"}, "order_id": {"type": "string"}},
            "required": ["summary"],
        },
        "low",
    ),
]


class Toolbox:
    def __init__(self, conn: sqlite3.Connection, kb: PolicyKnowledgeBase, settings: Settings) -> None:
        self.conn = conn
        self.kb = kb
        self.settings = settings
        self._handlers: dict[str, Callable[..., dict]] = {
            "lookup_order": self.lookup_order,
            "search_policy": self.search_policy,
            "issue_refund": self.issue_refund,
            "escalate_to_human": self.escalate_to_human,
        }

    @property
    def names(self) -> set[str]:
        return set(self._handlers)

    def specs(self) -> list[ToolSpec]:
        return list(SPECS)

    def execute(self, name: str, args: dict, customer_id: str) -> dict:
        if name not in self._handlers:
            raise KeyError(f"unknown tool: {name}")
        if name == "escalate_to_human":
            args = {**args, "customer_id": customer_id}
        return self._handlers[name](**args)

    # --- tool implementations -------------------------------------------------
    def lookup_order(self, order_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM orders WHERE id = ?", (str(order_id).upper(),)).fetchone()
        if row is None:
            return {"error": "not_found", "order_id": order_id}
        return {"order": dict(row)}

    def search_policy(self, query: str) -> dict:
        return {"passages": self.kb.search(str(query))}

    def issue_refund(self, order_id: str, amount: float, reason: str) -> dict:
        refund_id = add_refund(self.conn, str(order_id).upper(), float(amount), str(reason),
                               "issued", self.settings.today)
        return {"status": "issued", "refund_id": refund_id, "order_id": str(order_id).upper(),
                "amount": float(amount)}

    def request_refund_approval(self, order_id: str, amount: float, reason: str) -> dict:
        refund_id = add_refund(self.conn, str(order_id).upper(), float(amount), str(reason),
                               "pending_approval", self.settings.today)
        return {"status": "pending_approval", "approval_id": refund_id,
                "order_id": str(order_id).upper(), "amount": float(amount)}

    def escalate_to_human(self, summary: str, customer_id: str, order_id: str | None = None) -> dict:
        cur = self.conn.execute(
            "INSERT INTO tickets (customer_id, order_id, summary, created_on) VALUES (?, ?, ?, ?)",
            (customer_id, order_id, str(summary)[:500], self.settings.today.isoformat()),
        )
        self.conn.commit()
        return {"status": "ticket_created", "ticket_id": int(cur.lastrowid)}

    # --- human review ---------------------------------------------------------
    def list_approvals(self, status: str = "pending_approval") -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM refunds WHERE status = ? ORDER BY id", (status,)
        ).fetchall()
        return [dict(r) for r in rows]

    def decide_refund(self, refund_id: int, decision: str, reviewer: str) -> dict:
        row = self.conn.execute("SELECT * FROM refunds WHERE id = ?", (refund_id,)).fetchone()
        if row is None:
            raise LookupError("refund not found")
        if row["status"] != "pending_approval":
            raise ValueError(f"refund is already {row['status']}")
        new_status = "issued" if decision == "approve" else "rejected"
        self.conn.execute("UPDATE refunds SET status = ?, decided_by = ? WHERE id = ?",
                          (new_status, reviewer, refund_id))
        self.conn.commit()
        return {"refund_id": refund_id, "status": new_status, "decided_by": reviewer}
