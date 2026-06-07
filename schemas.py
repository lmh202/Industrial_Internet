"""Shared JSON-compatible data structures for planning and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlanStep:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": dict(self.args)}


@dataclass(frozen=True)
class Plan:
    plan_name: str
    steps: list[PlanStep]
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "plan_name": self.plan_name,
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.assumptions:
            payload["assumptions"] = list(self.assumptions)
        return payload


@dataclass(frozen=True)
class ToolResult:
    index: int
    tool: str
    args: dict[str, Any]
    ok: bool
    message: str = ""
    state_delta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "tool": self.tool,
            "args": dict(self.args),
            "ok": self.ok,
            "message": self.message,
        }
        if self.state_delta is not None:
            payload["state_delta"] = self.state_delta
        return payload


@dataclass(frozen=True)
class ExecutionReport:
    plan_name: str
    ok: bool
    steps: list[ToolResult]
    planning_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "plan_name": self.plan_name,
            "ok": self.ok,
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.planning_source:
            payload["planning_source"] = self.planning_source
        return payload
