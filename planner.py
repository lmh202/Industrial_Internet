"""Top-planner, operation-agent, and compiler production planning stack."""

from __future__ import annotations

import json
from typing import Any

from config import (
    API_KEY,
    BASE_MODEL,
    BASE_URL,
    LLM_KEEP_ALIVE,
    LLM_OPERATION_PLAN_TOKENS,
    LLM_TIMEOUT,
    LLM_TOP_PLAN_TOKENS,
)
from llm_client import LLMError, OpenAICompatibleClient
from process_compiler import compile_process_plan, validate_process_plan
from validator import PlanValidationError


SYSTEM_PROMPT = """
Return one compact JSON object only.
You are Top Planner. Convert the user request to high-level actions, not tools.
Infer the user's intent directly; there is no deterministic natural-language
pre-parser before this planner.
Schema:
{"plan_name":"short_snake_case","strategy":"sequential|parallel_start","actions":[]}
Allowed actions:
{"action":"produce_car"}
{"action":"produce_phone"}
{"action":"reset_status","line":"A|B"}
{"action":"move_to_output","line":"A|B","part":"car_base|phone_base|screen|camera_module|car_frame"}
Mapping: car/vehicle/汽车/车 -> produce_car. phone/手机 -> produce_phone.
Expand quantities into repeated actions. One car means exactly one produce_car.
Use reset_status only between two products on the same line: car line A, phone line B.
Use move_to_output only if the user explicitly asks to move/send a part to output.
For a move-to-output request without production words, output exactly one
move_to_output action. If the user says a phone-line part without naming it,
use line B and part phone_base. If the user says a car-line part without naming
it, use line A and part car_base.
Use parallel_start only when the user asks simultaneous/parallel/同时 start of car and phone.
No tool names, no operations, no explanations.
""".strip()


OPERATION_SUBAGENT_PROMPT = """
Return one compact JSON object only.
You are Operation Agent. Expand Top Planner actions into explicit operations.
Schema:
{"plan_name":"same_name","strategy":"sequential|parallel_start","operations":[]}
Templates:
produce_car -> [{"op":"load_base","line":"A","part":"car_base"},{"op":"assemble","line":"A","base":"car_base","part":"car_frame","supplier":"A2","layer":1},{"op":"inspect","line":"A"},{"op":"unload","line":"A","part":"car_base"}]
produce_phone -> [{"op":"load_base","line":"B","part":"phone_base"},{"op":"assemble","line":"B","base":"phone_base","part":"screen","supplier":"B2","layer":1},{"op":"assemble","line":"B","base":"phone_base","part":"camera_module","supplier":"B2","layer":2},{"op":"inspect","line":"B"},{"op":"unload","line":"B","part":"phone_base"}]
reset_status -> {"op":"reset_status","line":"same_line"}
move_to_output -> {"op":"move_to_output","line":"same_line","part":"same_part"}
For parallel_start, A-line and B-line operations may be interleaved, but each line order must stay unchanged.
No tool names, no explanations.
""".strip()


class OperationPlanningSubAgent:
    def __init__(self, llm_client: OpenAICompatibleClient):
        self.llm_client = llm_client
        self.last_operation_plan: dict[str, Any] | None = None

    def run(self, high_level_plan: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "High-level action plan JSON:\n"
            f"{json.dumps(high_level_plan, ensure_ascii=False)}\n\n"
            "Return the explicit operation plan JSON only."
        )
        last_error: PlanValidationError | None = None
        for attempt in range(3):
            try:
                payload = self.llm_client.chat_json(
                    OPERATION_SUBAGENT_PROMPT,
                    prompt,
                    max_tokens=LLM_OPERATION_PLAN_TOKENS,
                )
            except LLMError as exc:
                raise PlanValidationError(
                    f"Operation subagent cannot create operation plan: {exc}"
                ) from exc
            try:
                process_plan = validate_process_plan(payload)
                if "operations" not in process_plan:
                    raise PlanValidationError("Operation subagent must return operations.")
                _validate_operation_plan_against_top(process_plan, high_level_plan)
                self.last_operation_plan = process_plan
                return process_plan
            except PlanValidationError as exc:
                last_error = exc
                if attempt == 2:
                    break
                prompt = (
                    "High-level action plan JSON:\n"
                    f"{json.dumps(high_level_plan, ensure_ascii=False)}\n\n"
                    f"Previous operation plan was invalid: {exc}. "
                    "Return a corrected operation JSON object only."
                )
        raise last_error or PlanValidationError("Operation subagent failed.")


class ProductionPlanner:
    def __init__(self, llm_client: OpenAICompatibleClient | None = None):
        self.llm_client = llm_client or OpenAICompatibleClient(
            BASE_MODEL,
            BASE_URL,
            API_KEY,
            timeout=LLM_TIMEOUT,
            keep_alive=LLM_KEEP_ALIVE,
        )
        self.last_top_level_plan: dict[str, Any] | None = None
        self.last_subagent_plan: dict[str, Any] | None = None

    def run(self, prompt: str) -> dict[str, Any]:
        """Plan user text into a validated tool-call JSON object."""
        if not prompt.strip():
            raise PlanValidationError("Prompt is empty.")
        if not self.llm_client.is_configured:
            raise PlanValidationError("LLM is not configured.")

        self.last_top_level_plan = None
        self.last_subagent_plan = None
        operation_subagent = OperationPlanningSubAgent(self.llm_client)
        planning_prompt = prompt
        last_error: PlanValidationError | None = None

        for attempt in range(3):
            try:
                payload = self.llm_client.chat_json(
                    SYSTEM_PROMPT,
                    planning_prompt,
                    max_tokens=LLM_TOP_PLAN_TOKENS,
                )
            except LLMError as exc:
                raise PlanValidationError(f"Cannot create process plan: {exc}") from exc
            try:
                process_plan = validate_process_plan(payload)
                _validate_top_plan_against_prompt(process_plan, prompt)
                self.last_top_level_plan = process_plan
                if "actions" in process_plan or "jobs" in process_plan:
                    process_plan = operation_subagent.run(process_plan)
                    _validate_operation_plan_against_top(process_plan, self.last_top_level_plan)
                    self.last_subagent_plan = process_plan
                plan = compile_process_plan(process_plan)
                plan["planning_source"] = "top_planner_operation_agent_compiler"
                plan["planning_attempts"] = attempt + 1
                return plan
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


def _validate_top_plan_against_prompt(plan: dict[str, Any], prompt: str) -> None:
    if "actions" not in plan:
        raise PlanValidationError("Top planner must return high-level actions.")
    actions = plan["actions"]
    if _prompt_requests_move_to_output(prompt) and not _prompt_requests_production(prompt):
        for index, action in enumerate(actions, start=1):
            if action["action"].startswith("produce_"):
                raise PlanValidationError(
                    f"Top action {index} adds production not requested by the user."
                )
    if not _prompt_requests_move_to_output(prompt):
        for index, action in enumerate(actions, start=1):
            if action["action"] == "move_to_output":
                raise PlanValidationError(
                    f"Top action {index} adds move_to_output not requested by the user."
                )
    for index, action in enumerate(actions):
        if action["action"] != "reset_status":
            continue
        if index == len(actions) - 1:
            raise PlanValidationError("reset_status cannot be the last top action.")
        if index == 0:
            raise PlanValidationError("reset_status cannot be the first top action.")
        previous_line = _action_product_line(actions[index - 1])
        next_line = _action_product_line(actions[index + 1])
        if previous_line != action["line"] or next_line != action["line"]:
            raise PlanValidationError(
                "reset_status is only allowed between products on the same line."
            )


def _validate_operation_plan_against_top(
    operation_plan: dict[str, Any],
    top_plan: dict[str, Any] | None,
) -> None:
    if top_plan is None or "actions" not in top_plan:
        return
    top_actions = {action["action"] for action in top_plan["actions"]}
    operations = operation_plan.get("operations", [])
    if "move_to_output" not in top_actions:
        for index, operation in enumerate(operations, start=1):
            if operation["op"] == "move_to_output":
                raise PlanValidationError(
                    f"Operation {index} adds move_to_output not present in top plan."
                )
    if "reset_status" not in top_actions:
        for index, operation in enumerate(operations, start=1):
            if operation["op"] == "reset_status":
                raise PlanValidationError(
                    f"Operation {index} adds reset_status not present in top plan."
                )
    _validate_operation_sequence_matches_top(operation_plan, top_plan)


def _validate_operation_sequence_matches_top(
    operation_plan: dict[str, Any],
    top_plan: dict[str, Any],
) -> None:
    expected = _expected_operations_from_top(top_plan)
    actual = operation_plan.get("operations", [])
    if len(actual) != len(expected):
        raise PlanValidationError(
            f"Operation plan length {len(actual)} does not match expected {len(expected)}."
        )
    if top_plan.get("strategy") != "parallel_start":
        if actual != expected:
            raise PlanValidationError("Operation plan does not match top-plan actions.")
        return

    for line in ("A", "B"):
        expected_line = [op for op in expected if op.get("line") == line]
        actual_line = [op for op in actual if op.get("line") == line]
        if actual_line != expected_line:
            raise PlanValidationError(
                f"Operation plan changes internal order on line {line}."
            )
    if sorted(_operation_key(op) for op in actual) != sorted(
        _operation_key(op) for op in expected
    ):
        raise PlanValidationError("Operation plan contains unexpected operations.")


def _expected_operations_from_top(top_plan: dict[str, Any]) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for action in top_plan["actions"]:
        name = action["action"]
        if name == "produce_car":
            expected.extend([
                {"op": "load_base", "line": "A", "part": "car_base"},
                {
                    "op": "assemble",
                    "line": "A",
                    "base": "car_base",
                    "part": "car_frame",
                    "supplier": "A2",
                    "layer": 1,
                },
                {"op": "inspect", "line": "A"},
                {"op": "unload", "line": "A", "part": "car_base"},
            ])
        elif name == "produce_phone":
            expected.extend([
                {"op": "load_base", "line": "B", "part": "phone_base"},
                {
                    "op": "assemble",
                    "line": "B",
                    "base": "phone_base",
                    "part": "screen",
                    "supplier": "B2",
                    "layer": 1,
                },
                {
                    "op": "assemble",
                    "line": "B",
                    "base": "phone_base",
                    "part": "camera_module",
                    "supplier": "B2",
                    "layer": 2,
                },
                {"op": "inspect", "line": "B"},
                {"op": "unload", "line": "B", "part": "phone_base"},
            ])
        elif name == "reset_status":
            expected.append({
                "op": "reset_status",
                "line": action["line"],
            })
        elif name == "move_to_output":
            expected.append({
                "op": "move_to_output",
                "line": action["line"],
                "part": action["part"],
            })
    return expected


def _operation_key(operation: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, str(value)) for key, value in operation.items()))


def _action_product_line(action: dict[str, Any]) -> str | None:
    if action["action"] == "produce_car":
        return "A"
    if action["action"] == "produce_phone":
        return "B"
    return None


def _prompt_requests_move_to_output(prompt: str) -> bool:
    text = prompt.lower().replace(" ", "")
    has_output = any(word in text for word in (
        "output",
        "\u8f93\u51fa",
        "\u51fa\u6599",
        "\u8f93\u51fa\u533a",
        "\u51fa\u6599\u533a",
        "\u6210\u54c1\u533a",
    ))
    has_move = any(word in text for word in (
        "move",
        "transport",
        "send",
        "\u79fb\u52a8",
        "\u79fb\u5230",
        "\u79fb\u81f3",
        "\u642c\u5230",
        "\u9001\u5230",
        "\u9001\u81f3",
        "\u653e\u5230",
        "\u653e\u5165",
        "\u8fd0\u5230",
    ))
    return has_output and has_move


def _prompt_requests_production(prompt: str) -> bool:
    text = prompt.lower().replace(" ", "")
    return any(word in text for word in (
        "produce",
        "production",
        "manufacture",
        "make",
        "\u751f\u4ea7",
        "\u5236\u9020",
        "\u9020",
        "\u88c5\u914d",
        "\u7ec4\u88c5",
    ))
