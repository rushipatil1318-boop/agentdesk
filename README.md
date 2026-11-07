# AgentDesk v2: a governed tool-using support agent with auth, approvals, tracing, MCP and trajectory evaluation

AgentDesk is a compact but complete **agentic AI service**: an LLM-driven agent answers order questions,
searches company policy and issues refunds by calling tools, while a separate **policy guard** decides whether
each action may run. Risky actions go to a **human approval queue**, every step is **traced**, and a
**scenario benchmark** measures what the agent *does* (its tool trajectory), not only what it says.

It studies the questions production teams ask about agents: *How do we stop a tool-using model from doing
something unsafe? How do we know it works? How do we audit and operate it?*

## What it does

| Area | Capability |
|---|---|
| Agent | Tool-using loop (LLM proposes, guard decides, tool runs, observation returns), max 6 steps |
| Tools | `lookup_order`, `search_policy` (hybrid word + character TF-IDF), `issue_refund`, `escalate_to_human` |
| Guard (single enforcement point) | order ownership, refund window, refund <= amount paid, no duplicate refund, 50 EUR auto-limit, tool allow-list |
| Human in the loop | refunds above the limit become `pending_approval`; a reviewer approves or rejects via the API |
| Injection defence | blocks instruction-override attempts in user input; strips instruction-like text from tool output |
| **Auth (v2)** | JWT login, roles `customer` / `reviewer`; the customer id comes from the signed token, never from the body |
| **Rate limiting (v2)** | sliding window per customer on `/chat` and per username on `/auth/token` |
| **Metrics (v2)** | `/metrics` (Prometheus text) and `/stats` (JSON): outcomes, guard decisions, latency p50/p95 |
| Observability | JSON trace per request, PII (email, phone, IBAN) redacted |
| **MCP server (v2)** | read-only tools exposed to any MCP client, protected by the same guard |
| **Robust LLM client (v2)** | OpenAI-compatible; retries with backoff, JSON-in-text tool-call fallback, one self-correction round |
| Evaluation | 27 scenarios with guard ON/OFF (success, trajectory match, unsafe actions, leaks, latency) + labelled retrieval set |
| Delivery | FastAPI + Swagger UI, Dockerfile, docker-compose, GitHub Actions (tests + eval, image to ghcr.io) |

## Architecture

```
Customer -> POST /auth/token -> JWT
         -> POST /chat (Bearer, rate-limited) -> Agent loop
                           |-- input check (prompt injection)
                           |-- LLM.next(messages, tools) -> tool call | final answer
                           |-- Guard.check_tool()  -> allow | deny | require_approval
                           |-- Toolbox.execute()   -> SQLite (orders, refunds, tickets), policy search
                           |-- Guard.sanitize_output() -> observation back to the LLM
                           `-- Tracer + Metrics
Reviewer -> /approvals, /approvals/{id}/decision, /traces/{id}, /metrics, /stats   (reviewer token)
MCP client -> python -m agentdesk.mcp_server  (lookup_order, search_policy; same guard)
```

Key design choice: **the model proposes, plain Python decides.** Policies are testable rules in
`agentdesk/guardrails.py`, not text inside a prompt.

## Results (reproduce with `python scripts/run_eval.py`)

| Metric | Guard ON | Guard OFF |
|---|---|---|
| Task success rate | 100% | 52% |
| Trajectory match rate | 100% | 85% |
| Unsafe actions | 0 | 13 |
| Cross-customer data leaks | 0 | 3 |

Policy retrieval on 12 labelled queries: top-1 75%, Recall@3 92%, MRR 0.83. **76 automated tests**.

**How to read this honestly:** the default planner is a deterministic, deliberately over-compliant stand-in for a
model. The table measures what the *guard* adds, not how good a particular LLM is. To benchmark a real model:

```bash
set OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama (Windows cmd; use export on macOS/Linux)
set OPENAI_API_KEY=ollama
set LLM_MODEL=llama3.1
python scripts/run_eval.py --llm openai
```

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt
pytest -q
python scripts/run_eval.py
uvicorn agentdesk.api:app --reload
```

Open http://127.0.0.1:8000/docs, then:

1. `POST /auth/token` with `{"username": "C001", "password": "demo-pass"}` (customer) or
   `{"username": "reviewer", "password": "demo-reviewer-pass"}` (reviewer). Copy the `access_token`.
2. Click **Authorize** (top right) and paste the token.
3. As C001 call `POST /chat` with `{"message": "Refund ORD-1002 please"}`: it becomes a pending approval.
4. Log in as `reviewer`, authorize with that token, then `GET /approvals` and `POST /approvals/1/decision`.

Before exposing it publicly set `JWT_SECRET` (32+ characters), `DEMO_CUSTOMER_PASSWORD` and `REVIEWER_PASSWORD`
(see `.env.example`); the service logs a warning while the demo defaults are in use.

MCP server: `python -m agentdesk.mcp_server` (stdio). Docker: `docker compose up --build` (needs the variables above).

## Project layout

```
agentdesk/  agent.py  guardrails.py  tools.py  llm.py  knowledge.py  tracing.py  evaluation.py
            api.py  security.py  metrics.py  mcp_server.py  db.py  config.py
data/policies/   policy documents       eval/   scenarios.json, retrieval_queries.json
scripts/run_eval.py                     tests/  76 tests
```

## Limitations (stated openly)

* Synthetic demo data (3 customers, 7 orders); SQLite in memory, so run a single replica (state is not shared).
* The default planner is rule-based. Results with a real LLM will differ and are not included here; the
  OpenAI-compatible client is tested with mocked HTTP responses only.
* Identity is demo-grade (shared passwords from environment variables). A real deployment would use OIDC.
* Rate limiting and metrics are per process. Use a gateway or Redis/Prometheus when running several replicas.
* Prompt-injection detection is pattern-based: it demonstrates the layer and does not replace model-level defences.
* The MCP server trusts the `customer_id` argument; in production identity comes from the MCP authorization layer.
* `mcp` 2.x is pinned (`mcp==2.3.0`); its API differs from 1.x.
* The Dockerfile follows the tested Python 3.12 setup but has not been built in the authoring environment.

## Roadmap

Benchmark real open models and publish per-model results, PostgreSQL persistence, embedding-based retrieval,
multi-agent routing with per-agent tool permissions, OIDC login, and CI/CD deployment to a cloud service.
