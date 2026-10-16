"""Agent plumbing: typed tools, trace log, metrics.

Every agent call is recorded as an :class:`AgentTrace` (who decided what, why,
how long it took). Tools are typed pydantic models so the same registry can
be exposed to an LLM as function-calling tools (Azure OpenAI / AI Foundry)
or called directly by the deterministic fallback agents.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from pitchlens.schema import AgentTrace


@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    fn: Callable[[BaseModel], Any]

    def openai_schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                  "parameters": self.args.model_json_schema()}}


class ToolRegistry:
    def __init__(self, trace: TraceLog):
        self.tools: dict[str, Tool] = {}
        self.trace = trace

    def register(self, name: str, description: str, args: type[BaseModel]):
        def deco(fn: Callable[[BaseModel], Any]):
            self.tools[name] = Tool(name, description, args, fn)
            return fn

        return deco

    def call(self, agent: str, name: str, t_match: float, **kwargs: Any) -> Any:
        tool = self.tools[name]
        args = tool.args(**kwargs)
        t0 = time.perf_counter()
        out = tool.fn(args)
        self.trace.add(agent, t_match, f"tool:{name}", f"{name}({args.model_dump_json()})",
                       output={"result": summarize(out)}, latency_ms=(time.perf_counter() - t0) * 1000)
        return out

    def schemas(self) -> list[dict]:
        return [t.openai_schema() for t in self.tools.values()]


def summarize(x: Any, limit: int = 240) -> Any:
    if isinstance(x, BaseModel):
        x = x.model_dump()
    s = repr(x)
    return x if len(s) <= limit else s[:limit] + "..."


@dataclass
class TraceLog:
    traces: list[AgentTrace] = field(default_factory=list)
    metrics: Counter = field(default_factory=Counter)
    keep_tool_calls: bool = True

    def add(self, agent: str, t_match: float, action: str, summary: str, output: dict | None = None,
            latency_ms: float = 0.0, ok: bool = True, input_ref: str | None = None) -> AgentTrace:
        if action.startswith("tool:") and not self.keep_tool_calls:
            self.metrics["tool_calls"] += 1
            return AgentTrace(seq=-1, agent=agent, t_match=t_match, action=action, summary=summary)
        tr = AgentTrace(seq=len(self.traces), agent=agent, t_match=round(t_match, 2), action=action, summary=summary,
                        input_ref=input_ref, output=output or {}, latency_ms=round(latency_ms, 3), ok=ok)
        self.traces.append(tr)
        return tr

    def for_insight(self, insight_id: str) -> list[AgentTrace]:
        return [t for t in self.traces if t.input_ref == insight_id]
