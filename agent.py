"""Compatibility exports for the production planning agent."""

from __future__ import annotations

import json
from typing import Any

from planner import ProductionAgent, ProductionPlanner, SYSTEM_PROMPT
from process_compiler import DEFAULT_LINE_PART, LINE_PARTS
from tool_registry import (
    ALLOWED_TOOLS,
    CROSS_LINE_TOOLS,
    HOLD_TOOLS,
    INSPECT_TOOLS,
    LOAD_TOOLS,
    PLACE_TOOLS,
    PLAN_TRANSPORT_ROUTES,
    SUPPORTED_PARTS,
    TOOL_REGISTRY,
    TRANSPORT_TOOL_ROUTES,
    TRANSPORT_TOOLS,
    UNLOAD_TOOLS,
)
from validator import PlanValidationError, TaskParseError, validate_plan, validate_plan_sequence


def plan_to_json(plan: dict[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False)


def tasks_to_json(plan: dict[str, Any]) -> str:
    return plan_to_json(plan)


__all__ = [
    "ALLOWED_TOOLS",
    "CROSS_LINE_TOOLS",
    "DEFAULT_LINE_PART",
    "HOLD_TOOLS",
    "INSPECT_TOOLS",
    "LINE_PARTS",
    "LOAD_TOOLS",
    "PLACE_TOOLS",
    "PLAN_TRANSPORT_ROUTES",
    "PlanValidationError",
    "ProductionAgent",
    "ProductionPlanner",
    "SUPPORTED_PARTS",
    "SYSTEM_PROMPT",
    "TOOL_REGISTRY",
    "TRANSPORT_TOOL_ROUTES",
    "TRANSPORT_TOOLS",
    "TaskParseError",
    "UNLOAD_TOOLS",
    "plan_to_json",
    "tasks_to_json",
    "validate_plan",
    "validate_plan_sequence",
]
