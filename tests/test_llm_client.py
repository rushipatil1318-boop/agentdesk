import json

import httpx
import pytest

from agentdesk.llm import Final, LLMError, OpenAICompatibleLLM, ToolCall
from agentdesk.tools import SPECS


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_tool_call_response_is_parsed():
    def handler(request):
        body = json.loads(request.content)
        assert body["tools"][0]["function"]["name"] == "lookup_order"
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [
            {"function": {"name": "lookup_order", "arguments": "{\"order_id\": \"ORD-1001\"}"}}]}}]})

    llm = OpenAICompatibleLLM("http://x/v1", "k", "m", client=_client(handler))
    action = llm.next([{"role": "user", "content": "hi"}], SPECS)
    assert action == ToolCall("lookup_order", {"order_id": "ORD-1001"})


def test_text_response_is_final():
    llm = OpenAICompatibleLLM("http://x/v1", "k", "m", client=_client(
        lambda r: httpx.Response(200, json={"choices": [{"message": {"content": " Hello "}}]})))
    assert llm.next([{"role": "user", "content": "hi"}], SPECS) == Final("Hello")


def test_http_error_becomes_llm_error():
    llm = OpenAICompatibleLLM("http://x/v1", "k", "m", client=_client(lambda r: httpx.Response(500)), sleep=lambda s: None)
    with pytest.raises(LLMError):
        llm.next([{"role": "user", "content": "hi"}], SPECS)


def test_history_is_converted_with_matching_tool_call_ids():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_call": {"name": "lookup_order", "args": {"order_id": "ORD-1"}}},
        {"role": "tool", "name": "lookup_order", "content": {"error": "not_found"}},
    ]
    converted = OpenAICompatibleLLM._to_openai(messages)
    assert converted[2]["tool_calls"][0]["id"] == converted[3]["tool_call_id"] == "call_1"


def _reply(content=None, tool_call=None):
    message = {"content": content}
    if tool_call is not None:
        message["tool_calls"] = [{"function": tool_call}]
    return httpx.Response(200, json={"choices": [{"message": message}]})


def _llm(handler, **kwargs):
    return OpenAICompatibleLLM("http://x/v1", "k", "m", client=_client(handler), sleep=lambda s: None, **kwargs)


@pytest.mark.parametrize("text", [
    '{"name": "lookup_order", "arguments": {"order_id": "ORD-1001"}}',
    '```json\n{"tool": "lookup_order", "args": {"order_id": "ORD-1001"}}\n```',
    'Sure! <tool_call>{"name": "lookup_order", "arguments": "{\\"order_id\\": \\"ORD-1001\\"}"}</tool_call>',
])
def test_tool_call_written_as_text_is_recovered(text):
    llm = _llm(lambda r: _reply(content=text))
    assert llm.next([{"role": "user", "content": "hi"}], SPECS) == ToolCall("lookup_order", {"order_id": "ORD-1001"})


def test_invalid_tool_call_is_corrected_once():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body["messages"][-1]["content"])
        if len(calls) == 1:
            return _reply(tool_call={"name": "lookup_order", "arguments": "{}"})  # missing order_id
        return _reply(tool_call={"name": "lookup_order", "arguments": "{\"order_id\": \"ORD-1001\"}"})

    action = _llm(handler).next([{"role": "user", "content": "hi"}], SPECS)
    assert action == ToolCall("lookup_order", {"order_id": "ORD-1001"})
    assert "missing required argument" in calls[1] and len(calls) == 2


def test_unknown_tool_and_bad_json_are_retried_then_fail():
    llm = _llm(lambda r: _reply(tool_call={"name": "drop_tables", "arguments": "not json"}), max_retries=1)
    with pytest.raises(LLMError, match="invalid tool calls"):
        llm.next([{"role": "user", "content": "hi"}], SPECS)


def test_server_errors_are_retried_with_backoff():
    sleeps, attempts = [], []

    def handler(request):
        attempts.append(1)
        return httpx.Response(503) if len(attempts) < 3 else _reply(content="done")

    llm = OpenAICompatibleLLM("http://x/v1", "k", "m", client=_client(handler), sleep=sleeps.append)
    assert llm.next([{"role": "user", "content": "hi"}], SPECS) == Final("done")
    assert len(attempts) == 3 and sleeps == [0.5, 1.0]


def test_client_errors_are_not_retried():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(401)

    with pytest.raises(LLMError):
        _llm(handler).next([{"role": "user", "content": "hi"}], SPECS)
    assert len(attempts) == 1


def test_plain_json_that_is_not_a_tool_call_stays_text():
    llm = _llm(lambda r: _reply(content='{"answer": 42}'))
    assert isinstance(llm.next([{"role": "user", "content": "hi"}], SPECS), Final)
