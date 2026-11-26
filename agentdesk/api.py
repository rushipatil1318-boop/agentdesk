"""FastAPI service (v2): JWT auth with roles, rate limiting, metrics.
Interactive docs (Swagger UI) at /docs: log in via POST /auth/token, then click 'Authorize' and paste the token."""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from . import db
from .agent import Agent
from .config import Settings, load_settings
from .guardrails import Guard
from .knowledge import PolicyKnowledgeBase
from .llm import LLM, NaivePlanner, OpenAICompatibleLLM
from .metrics import Metrics
from .security import AuthError, Claims, RateLimiter, TokenService
from .tools import Toolbox
from .tracing import Tracer

log = logging.getLogger("agentdesk")


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=40, examples=["C001"])
    password: str = Field(min_length=1, max_length=200, examples=["demo-pass"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000, examples=["Where is my order ORD-1001?"])


class StepOut(BaseModel):
    tool: str
    verdict: str
    rule: str
    executed: bool


class ChatResponse(BaseModel):
    reply: str
    outcome: str
    trace_id: str
    steps: list[StepOut]
    pending_approval_id: int | None = None


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str = Field(min_length=1, max_length=60)


def create_app(llm: LLM | None = None, guard_enabled: bool = True, db_path: str = ":memory:",
               settings: Settings | None = None, tokens: TokenService | None = None,
               rate_limit_per_min: int | None = None) -> FastAPI:
    settings = settings or load_settings()
    if llm is None:
        llm = OpenAICompatibleLLM.from_env() if os.getenv("LLM_PROVIDER") == "openai" else NaivePlanner()
    conn = db.connect(db_path)
    db.init_db(conn, settings.today)
    customer_ids = {r["id"] for r in conn.execute("SELECT id FROM customers")}
    tokens = tokens or TokenService.from_env(customer_ids)
    if tokens.default_credentials_in_use:
        log.warning("Default demo passwords are in use. Set DEMO_CUSTOMER_PASSWORD, REVIEWER_PASSWORD and "
                    "JWT_SECRET before exposing this service publicly.")
    toolbox = Toolbox(conn, PolicyKnowledgeBase(), settings)
    guard = Guard(conn, settings, guard_enabled, toolbox.names)
    tracer = Tracer(settings.trace_path)
    agent = Agent(llm, toolbox, guard, tracer, settings)
    metrics = Metrics()
    limit = rate_limit_per_min or int(os.getenv("RATE_LIMIT_PER_MIN", "20"))
    chat_limiter = RateLimiter(limit, 60.0)
    login_limiter = RateLimiter(10, 60.0)
    lock = threading.Lock()  # one shared SQLite connection -> serialize agent runs

    app = FastAPI(title="AgentDesk", version="2.0.0",
                  description="Governed tool-using support agent: JWT auth, rate limiting, human approval, "
                              "tracing, metrics and trajectory evaluation.")
    bearer = HTTPBearer(auto_error=False)

    def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> Claims:
        if credentials is None:
            raise HTTPException(401, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"})
        try:
            return tokens.verify(credentials.credentials)
        except AuthError as exc:
            raise HTTPException(401, str(exc), headers={"WWW-Authenticate": "Bearer"})

    def customer_user(user: Claims = Depends(current_user)) -> Claims:
        if user.role != "customer":
            raise HTTPException(403, "Customer token required")
        return user

    def reviewer_user(user: Claims = Depends(current_user)) -> Claims:
        if user.role != "reviewer":
            raise HTTPException(403, "Reviewer token required")
        return user

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": "2.0.0", "guard_enabled": guard_enabled, "llm": type(llm).__name__}

    @app.post("/auth/token", response_model=TokenResponse)
    def login(body: LoginRequest) -> TokenResponse:
        if not login_limiter.allow(body.username):
            metrics.inc("rate_limited", endpoint="auth")
            raise HTTPException(429, "Too many login attempts. Try again in a minute.")
        try:
            token, role = tokens.login(body.username, body.password)
        except AuthError:
            metrics.inc("login_failed")
            raise HTTPException(401, "Invalid username or password")
        metrics.inc("login_ok", role=role)
        return TokenResponse(access_token=token, role=role)

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, user: Claims = Depends(customer_user)) -> ChatResponse:
        if not chat_limiter.allow(user.subject):
            metrics.inc("rate_limited", endpoint="chat")
            raise HTTPException(429, "Rate limit exceeded. Please slow down.")
        started = time.perf_counter()
        with lock:  # the customer id comes from the signed token, never from the body
            result = agent.run(user.subject, body.message)
        metrics.observe_latency((time.perf_counter() - started) * 1000)
        metrics.inc("chat_requests", outcome=result.outcome)
        for step in result.steps:
            if step.verdict != "allow":
                metrics.inc("guard_decisions", verdict=step.verdict, rule=step.rule)
        return ChatResponse(
            reply=result.reply, outcome=result.outcome, trace_id=result.trace_id,
            steps=[StepOut(tool=s.tool, verdict=s.verdict, rule=s.rule, executed=s.executed)
                   for s in result.steps],
            pending_approval_id=result.pending_approval_id,
        )

    @app.get("/approvals")
    def approvals(status: str = "pending_approval", _: Claims = Depends(reviewer_user)) -> list[dict]:
        with lock:
            return toolbox.list_approvals(status)

    @app.post("/approvals/{refund_id}/decision")
    def decide(refund_id: int, body: DecisionRequest, _: Claims = Depends(reviewer_user)) -> dict:
        with lock:
            try:
                result = toolbox.decide_refund(refund_id, body.decision, body.reviewer)
            except LookupError:
                raise HTTPException(404, "Refund not found")
            except ValueError as exc:
                raise HTTPException(409, str(exc))
        metrics.inc("approval_decisions", decision=body.decision)
        return result

    @app.get("/traces/{trace_id}")
    def trace(trace_id: str, _: Claims = Depends(reviewer_user)) -> list[dict]:
        events = tracer.get(trace_id)
        if not events:
            raise HTTPException(404, "Trace not found")
        return events

    @app.get("/stats")
    def stats(_: Claims = Depends(reviewer_user)) -> dict:
        return metrics.snapshot()

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus(_: Claims = Depends(reviewer_user)) -> str:
        return metrics.render_prometheus()

    return app


app = create_app()
