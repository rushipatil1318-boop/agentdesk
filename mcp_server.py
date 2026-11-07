"""MCP server exposing AgentDesk's READ-ONLY tools to any MCP client (Claude Desktop, IDE agents, ...).

Run (stdio transport):   python -m agentdesk.mcp_server

Design notes
* Only read tools are exposed. Money-moving actions stay behind the approval API.
* The same Guard protects these tools (an order of another customer is denied).
* Limitation: in this demo the MCP client passes customer_id itself. In a real deployment the identity comes
  from the MCP authorization layer, not from a tool argument.
"""
from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

from . import db
from .config import Settings, load_settings
from .guardrails import Guard
from .knowledge import PolicyKnowledgeBase
from .tools import Toolbox

READ_ONLY_TOOLS = {"lookup_order", "search_policy"}


def build_server(settings: Settings | None = None, conn=None) -> MCPServer:
    settings = settings or load_settings()
    if conn is None:
        conn = db.connect(os.getenv("DATABASE_PATH", ":memory:"))
        db.init_db(conn, settings.today)
    toolbox = Toolbox(conn, PolicyKnowledgeBase(), settings)
    guard = Guard(conn, settings, True, READ_ONLY_TOOLS)
    server = MCPServer("agentdesk")

    @server.tool()
    def lookup_order(customer_id: str, order_id: str) -> dict:
        """Look up the status and details of one of the customer's own orders (format ORD-1234)."""
        decision = guard.check_tool("lookup_order", {"order_id": order_id}, customer_id)
        if decision.verdict != "allow":
            return {"error": "blocked_by_policy", "rule": decision.rule, "reason": decision.reason}
        return guard.sanitize_output(toolbox.lookup_order(order_id))

    @server.tool()
    def search_policy(query: str) -> dict:
        """Search the support policy documents (refunds, shipping, privacy) and return relevant passages."""
        return guard.sanitize_output(toolbox.search_policy(query))

    return server


if __name__ == "__main__":
    build_server().run()
