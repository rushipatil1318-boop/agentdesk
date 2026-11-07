import jwt
import pytest

SECRET = "unit-test-secret-0123456789-abcdefghij"

from agentdesk.metrics import Metrics
from agentdesk.security import AuthError, RateLimiter, TokenService


def service(**kwargs):
    return TokenService({"C001"}, secret=SECRET, customer_password="cp", reviewer_password="rp", **kwargs)


def test_login_and_verify_roundtrip():
    svc = service()
    token, role = svc.login("C001", "cp")
    claims = svc.verify(token)
    assert (claims.subject, claims.role, role) == ("C001", "customer", "customer")
    assert svc.verify(svc.login("reviewer", "rp")[0]).role == "reviewer"


@pytest.mark.parametrize("user,pw", [("C001", "x"), ("reviewer", "cp"), ("ghost", "cp"), ("C001", "")])
def test_wrong_credentials(user, pw):
    with pytest.raises(AuthError):
        service().login(user, pw)


def test_expired_token_is_rejected():
    now = [1000.0]
    svc = service(ttl_seconds=60, clock=lambda: now[0])
    token, _ = svc.login("C001", "cp")
    now[0] += 3600
    with pytest.raises(AuthError):
        svc.verify(token)


def test_tampered_and_foreign_tokens_are_rejected():
    svc = service()
    token, _ = svc.login("C001", "cp")
    with pytest.raises(AuthError):
        svc.verify(token[:-2] + "xx")
    forged = jwt.encode({"sub": "C001", "role": "reviewer", "exp": 9999999999}, "another-test-secret-0123456789-abcdefgh", algorithm="HS256")
    with pytest.raises(AuthError):
        svc.verify(forged)
    none_alg = jwt.encode({"sub": "C001", "role": "reviewer", "exp": 9999999999}, None, algorithm="none")
    with pytest.raises(AuthError):
        svc.verify(none_alg)


def test_unknown_role_is_rejected():
    svc = service()
    bad = jwt.encode({"sub": "x", "role": "admin", "exp": 9999999999}, SECRET, algorithm="HS256")
    with pytest.raises(AuthError):
        svc.verify(bad)


def test_default_credentials_are_flagged():
    assert TokenService({"C001"}).default_credentials_in_use
    assert not service().default_credentials_in_use


def test_rate_limiter_sliding_window():
    now = [0.0]
    limiter = RateLimiter(2, 10.0, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    assert limiter.allow("b")  # separate key
    now[0] = 11.0
    assert limiter.allow("a")


def test_metrics_render():
    m = Metrics()
    m.inc("chat_requests", outcome="answered")
    m.inc("chat_requests", outcome="answered")
    m.observe_latency(10.0)
    m.observe_latency(30.0)
    text = m.render_prometheus()
    assert 'agentdesk_chat_requests_total{outcome="answered"} 2' in text
    assert "agentdesk_chat_latency_ms_count 2" in text
    assert m.snapshot()["latency_ms"]["count"] == 2


def test_short_secret_is_refused():
    with pytest.raises(ValueError):
        TokenService({"C001"}, secret="too-short")
