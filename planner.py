"""LLM-backed production planner with deterministic guardrails."""

from __future__ import annotations

from typing import Any

from config import API_KEY, BASE_MODEL, BASE_URL, LLM_KEEP_ALIVE, LLM_TIMEOUT
from llm_client import LLMError, OpenAICompatibleClient
from process_compiler import compile_process_plan, validate_process_plan
from process_knowledge import build_process_knowledge_prompt
from rules import rule_process_plan_from_prompt
from validator import PlanValidationError


SYSTEM_PROMPT = f"""
Return one JSON object only. No markdown, prose, code, coordinates, robot names,
CoppeliaSim handles, tool names, loops, or macros.

You are the top-level production intent planner. Output a high-level action
plan with this shape:
{{
  "plan_name": "short_snake_case",
  "strategy": "sequential" | "parallel_start",
  "actions": [
    {{"action": "produce_car" | "produce_phone"}},
    {{"action": "reset_status", "line": "A" | "B"}},
    {{"action": "move_to_output", "line": "A" | "B", "part": "car_base"}}
  ],
  "assumptions": ["optional short assumption"]
}}

Use produce_car and produce_phone for product manufacturing. Use move_to_output
only when the user asks to move a specific line part to output. Do not use
quantity fields, loops, or product macros. If the user requests multiple
products, expand them into repeated high-level actions. After a product is
unloaded on a line, insert reset_status for that line before producing another
product on the same line. Example: two phones is produce_phone, reset_status B,
produce_phone. If the user asks to start car and phone simultaneously, use
strategy "parallel_start"; otherwise use "sequential". Preserve the user's
product order in actions.

{build_process_knowledge_prompt()}
""".strip()


OPERATION_SUBAGENT_PROMPT = f"""
Return one JSON object only. No markdown, prose, code, coordinates, robot names,
CoppeliaSim handles, tool names, loops, macros, or low-level tool calls.

You are the operation-planning subagent. Convert a high-level action plan into an
ordered operation plan. Do not use product-level shortcuts. Every product must
be expressed as explicit operations.

Output shape:
{{
  "plan_name": "same_or_clear_name",
  "strategy": "sequential" | "parallel_start",
  "operations": [
    {{"op": "load_base", "line": "A" | "B", "part": "car_base"}},
    {{
      "op": "assemble",
      "line": "A" | "B",
      "base": "car_base",
      "part": "car_frame",
      "supplier": "A2" | "B2",
      "layer": 1
    }},
    {{"op": "inspect", "line": "A" | "B"}},
    {{"op": "unload", "line": "A" | "B", "part": "car_base"}},
    {{"op": "reset_status", "line": "A" | "B"}},
    {{"op": "move_to_output", "line": "A" | "B", "part": "screen"}}
  ],
  "assumptions": ["optional short assumption"]
}}

Allowed operation names: load_base, assemble, inspect, unload, reset_status,
move_to_output.
For car: load car_base, assemble car_frame supplied by A2, inspect A, unload
car_base.
For phone: load phone_base, assemble screen supplied by B2 before
camera_module, assemble camera_module supplied by B2, inspect B, unload
phone_base.
Translate reset_status actions to reset_status operations. For repeated
products, keep reset_status between runs on the same line. For parallel_start,
interleave independent A-line and B-line operations while preserving each line's
internal order.

{build_process_knowledge_prompt()}
""".strip()


class OperationPlanningSubAgent:
    def __init__(self, llm_client: OpenAICompatibleClient):
        self.llm_client = llm_client

    def run(self, high_level_plan: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "High-level action plan JSON:\n"
            f"{high_level_plan}\n\n"
            "Return the explicit operation plan JSON only."
        )
        try:
            payload = self.llm_client.chat_json(OPERATION_SUBAGENT_PROMPT, prompt)
        except LLMError as exc:
            raise PlanValidationError(
                f"Operation subagent cannot create operation plan: {exc}"
            ) from exc
        process_plan = validate_process_plan(payload)
        if "operations" not in process_plan:
            raise PlanValidationError("Operation subagent must return operations.")
        return process_plan


class ProductionPlanner:
    def __init__(self, llm_client: OpenAICompatibleClient | None = None):
        self.llm_client = llm_client or OpenAICompatibleClient(
            BASE_MODEL,
            BASE_URL,
            API_KEY,
            timeout=LLM_TIMEOUT,
            keep_alive=LLM_KEEP_ALIVE,
        )

    def run(self, prompt: str) -> dict[str, Any]:
        """Plan user text into a validated tool-call JSON object."""
        if not prompt.strip():
            raise PlanValidationError("Prompt is empty.")

        rule_process_plan = rule_process_plan_from_prompt(prompt)
        if rule_process_plan is not None:
            return compile_process_plan(rule_process_plan)

        if not self.llm_client.is_configured:
            raise PlanValidationError("LLM is not configured.")

        planning_prompt = prompt
        last_error: PlanValidationError | None = None
        operation_subagent = OperationPlanningSubAgent(self.llm_client)
        for attempt in range(3):
            try:
                payload = self.llm_client.chat_json(SYSTEM_PROMPT, planning_prompt)
            except LLMError as exc:
                raise PlanValidationError(f"Cannot create tool plan: {exc}") from exc
            try:
                process_plan = validate_process_plan(payload)
                if "actions" in process_plan or "jobs" in process_plan:
                    process_plan = operation_subagent.run(process_plan)
                return compile_process_plan(process_plan)
            except PlanValidationError as exc:
                last_error = exc
                if attempt == 2:
                    break
                planning_prompt = (
                    f"{prompt}\n\n"
                    f"Previous process JSON plan was invalid: {exc}. "
                    "Return a corrected JSON object only. Do not explain."
                )
        raise last_error or PlanValidationError("Cannot create a valid tool plan.")


ProductionAgent = ProductionPlanner
