"""The agent loop: LLM proposes -> guard decides -> tool runs -> observation goes back to the LLM."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Settings
from .guardrails import SAFE_REFUSAL, Decision, Guard
from .llm import LLM, Final, LLMError, ToolCall
from .tools import Toolbox
from .tracing import Tracer


@dataclass
class Step:
    tool: str
    args: dict
    verdict: str
    rule: str
    executed: bool
    observation: dict


@dataclass
class AgentResult:
    reply: str
    trace_id: str
    steps: list[Step] = field(default_factory=list)
    blocked_input: bool = False
    needs_info: bool = False
    llm_error: bool = False
    pending_approval_id: int | None = None

    @property
    def outcome(self) -> str:
        return classify_outcome(self)


def classify_outcome(result: AgentResult) -> str:
    if result.blocked_input:
        return "blocked_input"
    observations = [s.observation for s in result.steps if s.executed]
    if any(o.get("status") == "issued" for o in observations):
        return "refund_issued"
    if any(o.get("status") == "pending_approval" for o in observations):
        return "pending_approval"
    if any(s.verdict == "deny" for s in result.steps):
        return "blocked_policy"
    if any(o.get("status") == "ticket_created" for o in observations):
        return "escalated"
    if result.llm_error:
        return "error"
    if result.needs_info:
        return "clarify"
    return "answered"


class Agent:
    def __init__(self, llm: LLM, toolbox: Toolbox, guard: Guard, tracer: Tracer,
                 settings: Settings) -> None:
        self.llm = llm
        self.toolbox = toolbox
        self.guard = guard
        self.tracer = tracer
        self.settings = settings

    def run(self, customer_id: str, message: str, trace_id: str | None = None) -> AgentResult:
        trace_id = trace_id or self.tracer.new_trace_id()
        self.tracer.emit(trace_id, "request", customer_id=customer_id, message=message,
                         guard_enabled=self.guard.enabled)
        if self.guard.enabled:
            check = self.guard.check_input(message)
            if check.flagged:
                self.tracer.emit(trace_id, "input_blocked", reason=check.reason)
                return AgentResult(SAFE_REFUSAL, trace_id, blocked_input=True)

        messages: list[dict] = [{"role": "user", "content": message}]
        result = AgentResult("", trace_id)
        specs = self.toolbox.specs()

        for _ in range(self.settings.max_steps):
            try:
                action = self.llm.next(messages, specs)
            except LLMError as exc:
                self.tracer.emit(trace_id, "llm_error", error=str(exc))
                result.reply = "The assistant is temporarily unavailable. Please try again later."
                result.llm_error = True
                return result

            if isinstance(action, Final):
                result.reply = action.text
                result.needs_info = action.kind == "clarify"
                self.tracer.emit(trace_id, "final", reply=action.text, kind=action.kind)
                return result

            step = self._execute(action, customer_id, trace_id)
            result.steps.append(step)
            if step.observation.get("status") == "pending_approval":
                result.pending_approval_id = step.observation.get("approval_id")
            messages.append({"role": "assistant", "tool_call": {"name": action.name, "args": action.args}})
            messages.append({"role": "tool", "name": action.name, "content": step.observation})

        result.reply = ("I couldn't finish this request automatically. "
                        "A human agent will follow up with you.")
        self.tracer.emit(trace_id, "max_steps_reached", steps=len(result.steps))
        return result

    def _execute(self, call: ToolCall, customer_id: str, trace_id: str) -> Step:
        started = time.perf_counter()
        decision = (self.guard.check_tool(call.name, call.args, customer_id)
                    if self.guard.enabled else Decision("allow", "guard.disabled"))
        if decision.verdict == "deny":
            observation = {"error": "blocked_by_policy", "rule": decision.rule, "reason": decision.reason}
            executed = False
        else:
            try:
                if decision.verdict == "require_approval":
                    observation = self.toolbox.request_refund_approval(**call.args)
                else:
                    observation = self.toolbox.execute(call.name, call.args, customer_id)
                executed = True
            except (KeyError, TypeError, ValueError) as exc:
                observation = {"error": "invalid_arguments", "detail": str(exc)}
                executed = False
        if self.guard.enabled:
            observation = self.guard.sanitize_output(observation)
        self.tracer.emit(trace_id, "tool_call", tool=call.name, args=call.args,
                         verdict=decision.verdict, rule=decision.rule, executed=executed,
                         observation=observation,
                         latency_ms=round((time.perf_counter() - started) * 1000, 2))
        return Step(call.name, call.args, decision.verdict, decision.rule, executed, observation)
