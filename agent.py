"""Compatibility exports for the production planning agent."""

from __future__ import annotations

import json
from typing import Any

from planner import (
    OPERATION_SUBAGENT_PROMPT,
    SYSTEM_PROMPT,
    OperationPlanningSubAgent,
    ProductionAgent,
    ProductionPlanner,
)
from process_compiler import (
    compile_process_operations,
    compile_process_plan,
    validate_process_plan,
)
from process_knowledge import PRODUCT_SPECS, build_process_knowledge_prompt
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
    "HOLD_TOOLS",
    "INSPECT_TOOLS",
    "LOAD_TOOLS",
    "PLACE_TOOLS",
    "PLAN_TRANSPORT_ROUTES",
    "PRODUCT_SPECS",
    "PlanValidationError",
    "OPERATION_SUBAGENT_PROMPT",
    "OperationPlanningSubAgent",
    "ProductionAgent",
    "ProductionPlanner",
    "SUPPORTED_PARTS",
    "SYSTEM_PROMPT",
    "TOOL_REGISTRY",
    "TRANSPORT_TOOL_ROUTES",
    "TRANSPORT_TOOLS",
    "TaskParseError",
    "UNLOAD_TOOLS",
    "build_process_knowledge_prompt",
    "compile_process_operations",
    "compile_process_plan",
    "plan_to_json",
    "tasks_to_json",
    "validate_plan",
    "validate_plan_sequence",
    "validate_process_plan",
]
