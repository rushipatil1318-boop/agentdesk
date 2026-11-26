import pytest
from fastapi.testclient import TestClient

from agentdesk.api import create_app
from agentdesk.config import Settings
from agentdesk.security import TokenService
from tests.conftest import TODAY

CUSTOMERS = {"C001", "C002", "C003"}


def make_client(**kwargs):
    tokens = TokenService(CUSTOMERS, secret="api-test-secret-0123456789-abcdefghijkl", customer_password="cp", reviewer_password="rp")
    return TestClient(create_app(settings=Settings(today=TODAY), tokens=tokens, **kwargs))


def login(client, username, password):
    response = client.post("/auth/token", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_health_is_public():
    assert make_client().get("/health").json()["status"] == "ok"


def test_chat_requires_a_token():
    client = make_client()
    assert client.post("/chat", json={"message": "hi"}).status_code == 401
    assert client.post("/chat", json={"message": "hi"}, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_bad_credentials_rejected():
    client = make_client()
    assert client.post("/auth/token", json={"username": "C001", "password": "wrong"}).status_code == 401
    assert client.post("/auth/token", json={"username": "ghost", "password": "cp"}).status_code == 401


def test_customer_identity_comes_from_the_token():
    client = make_client()
    headers = login(client, "C002", "cp")  # C002 asks about C001's order
    chat = client.post("/chat", json={"message": "Where is ORD-1001?"}, headers=headers).json()
    assert chat["outcome"] == "blocked_policy"
    own = client.post("/chat", json={"message": "Where is ORD-1005?"}, headers=headers).json()
    assert own["outcome"] == "answered"


def test_chat_validation_error():
    client = make_client()
    assert client.post("/chat", json={"message": ""}, headers=login(client, "C001", "cp")).status_code == 422


def test_customer_cannot_use_reviewer_endpoints():
    client = make_client()
    assert client.get("/approvals", headers=login(client, "C001", "cp")).status_code == 403
    assert client.get("/approvals").status_code == 401


def test_approval_flow_end_to_end():
    client = make_client()
    customer, reviewer = login(client, "C001", "cp"), login(client, "reviewer", "rp")
    chat = client.post("/chat", json={"message": "Refund ORD-1002 please"}, headers=customer).json()
    assert chat["outcome"] == "pending_approval"
    pending = client.get("/approvals", headers=reviewer).json()
    assert pending[0]["id"] == chat["pending_approval_id"]
    body = {"decision": "approve", "reviewer": "alice"}
    assert client.post(f"/approvals/{pending[0]['id']}/decision", json=body, headers=reviewer).json()["status"] == "issued"
    assert client.post(f"/approvals/{pending[0]['id']}/decision", json=body, headers=reviewer).status_code == 409
    assert client.post("/approvals/999/decision", json=body, headers=reviewer).status_code == 404


def test_trace_is_protected_and_redacted():
    client = make_client()
    customer, reviewer = login(client, "C001", "cp"), login(client, "reviewer", "rp")
    chat = client.post("/chat", json={"message": "Where is ORD-1001? mail me at anna@example.com"},
                       headers=customer).json()
    assert client.get(f"/traces/{chat['trace_id']}", headers=customer).status_code == 403
    events = client.get(f"/traces/{chat['trace_id']}", headers=reviewer).json()
    assert "anna@example.com" not in str(events) and "[email]" in str(events)


def test_rate_limit_returns_429():
    client = make_client(rate_limit_per_min=3)
    headers = login(client, "C001", "cp")
    codes = [client.post("/chat", json={"message": "Where is ORD-1001?"}, headers=headers).status_code
             for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]


def test_login_is_rate_limited():
    client = make_client()
    codes = [client.post("/auth/token", json={"username": "C001", "password": "bad"}).status_code
             for _ in range(12)]
    assert codes[:10] == [401] * 10 and codes[10:] == [429, 429]


def test_metrics_and_stats_for_reviewer_only():
    client = make_client()
    customer, reviewer = login(client, "C001", "cp"), login(client, "reviewer", "rp")
    client.post("/chat", json={"message": "Refund ORD-1003 please"}, headers=customer)
    assert client.get("/metrics", headers=customer).status_code == 403
    text = client.get("/metrics", headers=reviewer).text
    assert 'agentdesk_chat_requests_total{outcome="blocked_policy"} 1' in text
    assert 'rule="refund.window_expired"' in text
    stats = client.get("/stats", headers=reviewer).json()
    assert stats["latency_ms"]["count"] == 1
