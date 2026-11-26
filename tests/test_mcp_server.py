import asyncio
import json
import sys

from agentdesk.config import Settings
from agentdesk.mcp_server import build_server
from tests.conftest import TODAY


def call(server, name, args):
    """Call a tool in-process and return the JSON payload the client would receive."""
    result = asyncio.run(server.call_tool(name, args))
    return json.loads(result.content[0].text)


def test_only_read_only_tools_are_exposed():
    server = build_server(Settings(today=TODAY))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"lookup_order", "search_policy"}


def test_lookup_enforces_ownership():
    server = build_server(Settings(today=TODAY))
    own = call(server, "lookup_order", {"customer_id": "C001", "order_id": "ORD-1001"})
    assert own["order"]["status"] == "delivered"
    other = call(server, "lookup_order", {"customer_id": "C002", "order_id": "ORD-1001"})
    assert other["error"] == "blocked_by_policy"


def test_search_policy_returns_passages():
    server = build_server(Settings(today=TODAY))
    result = call(server, "search_policy", {"query": "refund window"})
    assert result["passages"]


def test_mcp_protocol_over_stdio():
    """Start the server as a real subprocess and talk to it with an MCP client."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def scenario():
        params = StdioServerParameters(command=sys.executable, args=["-m", "agentdesk.mcp_server"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                result = await session.call_tool("lookup_order", {"customer_id": "C001", "order_id": "ORD-1001"})
                return tools, result

    tools, result = asyncio.run(asyncio.wait_for(scenario(), timeout=60))
    assert {t.name for t in tools.tools} == {"lookup_order", "search_policy"}
    assert "ORD-1001" in str(result.content)
