"""Top-planner, operation-agent, and compiler production planning stack."""

from __future__ import annotations

import json
from collections.abc import Callable
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
from rules import (
    RuleParseError,
    parse_operation_plan,
    parse_top_plan,
    top_plan_to_operation_plan,
)
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
No markdown, prose, code, coordinates, robot names, CoppeliaSim handles, tool
names, loops, macros, or low-level tool calls.

You are Operation Agent. Convert a high-level action plan into an ordered
operation plan. Do not use product-level shortcuts. Each product must be
expanded into process operations.
The operation list must be a pure expansion of the given action list. Do not
add any production, reset_status, move_to_output, line, or part that is absent
from the Top Planner actions or product knowledge.

Output shape:
{{
  "plan_name": "same_or_clear_name",
  "strategy": "sequential" | "parallel_start",
  "operations": []
}}

Allowed operation names: load_base, assemble, inspect, unload, reset_status,
move_to_output.
Every operation object must use the key "op" for the operation name; never use
"operation", "action", or "type" as the operation-name key.
Operation fields: load_base uses line and part; assemble uses line, base, part,
supplier, layer; inspect uses line; unload uses line and part; reset_status
uses line; move_to_output uses line and part.
For car: load car_base on line A, assemble car_frame from supplier A2 at layer
1, inspect line A, unload car_base. Car uses only car_base and car_frame.
For phone: load phone_base on line B, assemble screen from supplier B2 at layer
1, assemble camera_module from supplier B2 at layer 2, inspect line B, unload
phone_base. Phone uses only phone_base, screen, and camera_module. screen must
be before camera_module.
Translate reset_status and move_to_output actions directly into matching
operations. Do not invent reset_status after unload; reset_status is allowed
only when the Top Planner action list already contains reset_status.
move_to_output is allowed only when the Top Planner action list already
contains move_to_output. For parallel_start, interleave independent A-line and
B-line work while preserving each line's internal order.
""".strip()

SOURCE_LAYOUT_PROMPT = """
Updated material-source correction:
- Source line A contains car_base and two phone_base stock slots.
- Source line B contains car_frame, screen, and camera_module.
- Product assembly remains fixed: cars are assembled on product line A, phones
  are assembled on product line B.
- In move_to_output, line means source/output side: car_base and phone_base use
  line A; car_frame, screen, and camera_module use line B.
""".strip()

SYSTEM_PROMPT = f"{SYSTEM_PROMPT}\n\n{SOURCE_LAYOUT_PROMPT}"

OPERATION_SOURCE_PROMPT = """
Correct operation schema and process language:
- load_base uses line, source_line, and part. It means mount the product base
  from its source side onto the product line shuttle; the compiler transfers it
  across lines if source_line differs from line.
- assemble uses line, source_line, base, part, supplier, and layer. It means
  fetch a component from its source side, transfer it if needed, then install
  it onto the base on the product line.
- inspect, unload, reset_status, and move_to_output keep their existing fields.
- For car: load car_base on product line A from source_line A; assemble
  car_frame from source_line B at layer 1; inspect A; unload car_base.
- For phone: load phone_base on product line B from source_line A; assemble
  screen from source_line B at layer 1; assemble camera_module from source_line
  B at layer 2; inspect B; unload phone_base.
- Do not use A2 for car_frame in new operation plans. B2 is only the phone
  product-line auxiliary supplier for screen and camera_module after they are
  identified as source_line B components.
""".strip()

OPERATION_SUBAGENT_PROMPT = (
    f"{OPERATION_SUBAGENT_PROMPT}\n\n{SOURCE_LAYOUT_PROMPT}\n\n"
    f"{OPERATION_SOURCE_PROMPT}"
)


SYSTEM_PROMPT = """
Return one compact JSON object only.
You are Top Planner. Convert Chinese or English user requests to high-level
production actions, not tool calls and not detailed operations.

Schema:
{"plan_name":"short_snake_case","strategy":"sequential|parallel_start","actions":[]}

Allowed actions:
{"action":"produce_car"}
{"action":"produce_phone"}
{"action":"reset_status","line":"A|B"}
{"action":"move_to_output","line":"A|B","part":"car_base|phone_base|screen|camera_module|car_frame"}

Never add line or part fields to produce_car or produce_phone.

Chinese mapping:
- 手机 / 电话 / phone -> produce_phone
- 汽车 / 小车 / 车 / car / vehicle -> produce_car
- 一 / 1 / 一部 / 一辆 -> one action
- 二 / 两 / 2 / 两部 / 两辆 -> two actions
- 按序 / 依次 / 连续 -> sequential
- 同时 / 同步 / 并行 -> parallel_start

Product lines:
- Cars are assembled on product line A.
- Phones are assembled on product line B.

Raw material source layout:
- Source A contains car_base, phone_base, phone_base.
- Source B contains car_frame, screen, camera_module.

After every produce_car, produce_phone, or move_to_output action, immediately
append reset_status for the same line to recalibrate the shuttle before any
next action or before ending the plan. phone uses reset_status line B; car uses
reset_status line A.
Use move_to_output only when the user explicitly asks to move a raw part to output.
For move_to_output, line means source/output side: car_base and phone_base use
line A; car_frame, screen, and camera_module use line B.
Expand quantities into repeated actions. Do not use loops or quantity fields.
Use parallel_start only when the user asks to start car and phone production
simultaneously.

Examples:
User: 按序生产两部手机和一部车
Output: {"plan_name":"produce_2_phones_1_car","strategy":"sequential","actions":[{"action":"produce_phone"},{"action":"reset_status","line":"B"},{"action":"produce_phone"},{"action":"reset_status","line":"B"},{"action":"produce_car"},{"action":"reset_status","line":"A"}]}
User: 同时生产一部手机和一辆汽车
Output: {"plan_name":"parallel_start_1_phone_1_car","strategy":"parallel_start","actions":[{"action":"produce_phone"},{"action":"reset_status","line":"B"},{"action":"produce_car"},{"action":"reset_status","line":"A"}]}

Output JSON only. No tools, no operations, no explanations.
""".strip()


OPERATION_SUBAGENT_PROMPT = """
Return one compact JSON object only.
No markdown, prose, code, coordinates, robot names, CoppeliaSim handles, tool
names, loops, macros, or low-level tool calls.

You are Operation Agent. Convert the Top Planner action list into an ordered
operation plan. Each product must be expanded into concrete process operations;
do not use product-level shortcuts.

Output shape:
{
  "plan_name": "same_or_clear_name",
  "strategy": "sequential" | "parallel_start",
  "operations": []
}

Allowed operation names:
- load_base
- assemble
- inspect
- unload
- reset_status
- move_to_output

Operation fields:
- load_base: {"op":"load_base","line":"A|B","source_line":"A|B","part":"..."}
- assemble: {"op":"assemble","line":"A|B","source_line":"A|B","base":"...","part":"...","supplier":"A|B|B2","layer":1}
- inspect: {"op":"inspect","line":"A|B"}
- unload: {"op":"unload","line":"A|B","part":"..."}
- reset_status: {"op":"reset_status","line":"A|B"}
- move_to_output: {"op":"move_to_output","line":"A|B","part":"..."}

The field line is the product line for load_base, assemble, inspect, unload,
and reset_status. The field source_line is the raw material source side.

Raw material source layout:
- Source A contains car_base and phone_base.
- Source B contains car_frame, screen, and camera_module.

Required car process:
1. Fetch car_frame from source_line B first.
2. Transfer car_frame from the B shuttle to the A-side auxiliary shuttle at
   the cross-line transfer position.
3. Return the original B shuttle to its pick position.
4. Move the A-side auxiliary shuttle directly under Assemble_A and immediately
   let the A assembly arm hold car_frame.
5. Return the A-side auxiliary shuttle to its clear position.
6. Mount car_base from source_line A onto the A main product shuttle and move
   that shuttle directly under Assemble_A.
7. Assemble the held car_frame onto car_base on product line A at layer 1.
8. Inspect product line A and unload car_base.

Therefore car operations must be exactly:
load_base line A source_line A part car_base;
assemble line A source_line B base car_base part car_frame supplier B layer 1;
inspect line A;
unload line A part car_base.

Required phone process:
1. Mount phone_base on the A source shuttle at pick, then move it directly
   from pick to the cross-line transfer position. Do not move phone_base to
   Assemble_A before transfer.
2. Move the B main product shuttle to the other side of the same transfer
   position, without overlapping the transfer robot base, and transfer
   phone_base to B in one clean transfer.
3. Move the B main product shuttle with phone_base to the forward side of
   Assemble_B.
4. Use the B auxiliary shuttle to load both camera_module and screen from
   source_line B in one supply trip. One of these parts must use a local_offset
   so the two raw parts do not overlap on the auxiliary shuttle.
5. Move the B auxiliary shuttle to the rear side of Assemble_B.
6. The low-level compiler schedules camera_module first and screen second from
   the auxiliary shuttle onto phone_base, keeping the two shuttles close enough
   for stable IK. At the operation-plan level, still output the canonical two
   assemble operations listed below; the compiler handles the physical order.
7. Reset the empty auxiliary shuttle, inspect product line B, and unload
   phone_base.

Therefore phone operations must be exactly:
load_base line B source_line A part phone_base;
assemble line B source_line B base phone_base part screen supplier B2 layer 1;
assemble line B source_line B base phone_base part camera_module supplier B2 layer 2;
inspect line B;
unload line B part phone_base.

Translate reset_status and move_to_output actions directly into matching
operations. Do not invent reset_status after unload; reset_status is allowed
only when the Top Planner action list already contains reset_status.
move_to_output is allowed only when the Top Planner action list already
contains move_to_output.
For parallel_start, interleave independent product-line work only when it does
not violate the raw material source layout. Preserve each product line's
internal order.
""".strip()


class OperationPlanningSubAgent:
    def __init__(self, llm_client: OpenAICompatibleClient):
        self.llm_client = llm_client
        self.last_operation_plan: dict[str, Any] | None = None

    def run(self, high_level_plan: dict[str, Any]) -> dict[str, Any]:
        if "actions" not in high_level_plan:
            return self._run_whole_plan(high_level_plan)

        operations: list[dict[str, Any]] = []
        cache: dict[str, list[dict[str, Any]]] = {}
        for action in high_level_plan["actions"]:
            cache_key = json.dumps(action, ensure_ascii=False, sort_keys=True)
            if cache_key not in cache:
                cache[cache_key] = self._plan_action_operations(
                    high_level_plan,
                    action,
                )
            operations.extend([dict(operation) for operation in cache[cache_key]])

        process_plan = validate_process_plan({
            "plan_name": high_level_plan["plan_name"],
            "strategy": high_level_plan["strategy"],
            "operations": operations,
        })
        _validate_operation_plan_against_top(process_plan, high_level_plan)
        self.last_operation_plan = process_plan
        return process_plan

    def _plan_action_operations(
        self,
        high_level_plan: dict[str, Any],
        action: dict[str, Any],
    ) -> list[dict[str, Any]]:
        action_name = action["action"]
        single_action_top_plan = {
            "plan_name": f"{high_level_plan['plan_name']}_{action_name}",
            "strategy": "sequential",
            "actions": [action],
        }
        if action_name in {"reset_status", "move_to_output"}:
            return top_plan_to_operation_plan(single_action_top_plan)["operations"]

        process_plan = self._run_single_action_plan(single_action_top_plan)
        return process_plan["operations"]

    def _run_single_action_plan(self, single_action_top_plan: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "Single high-level action JSON:\n"
            f"{json.dumps(single_action_top_plan, ensure_ascii=False)}\n\n"
            "Return the explicit operation plan JSON for this one action only."
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
                _validate_operation_plan_against_top(process_plan, single_action_top_plan)
                return process_plan
            except PlanValidationError as exc:
                last_error = exc
                if attempt == 2:
                    break
                prompt = (
                    "Single high-level action JSON:\n"
                    f"{json.dumps(single_action_top_plan, ensure_ascii=False)}\n\n"
                    f"Previous operation plan was invalid: {exc}. "
                    "Return a corrected operation JSON object for this one action only."
                )
        raise last_error or PlanValidationError("Operation subagent failed.")

    def _run_whole_plan(self, high_level_plan: dict[str, Any]) -> dict[str, Any]:
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
    def __init__(
        self,
        llm_client: OpenAICompatibleClient | None = None,
        on_top_plan: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.llm_client = llm_client or OpenAICompatibleClient(
            BASE_MODEL,
            BASE_URL,
            API_KEY,
            timeout=LLM_TIMEOUT,
            keep_alive=LLM_KEEP_ALIVE,
        )
        self.on_top_plan = on_top_plan
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
                return self._run_rule_fallback(
                    prompt,
                    f"top planner LLM request failed: {exc}",
                )
            try:
                process_plan = validate_process_plan(payload)
                _validate_top_plan_against_prompt(process_plan, prompt)
                self.last_top_level_plan = process_plan
                self._emit_top_plan(process_plan)
                if "actions" in process_plan or "jobs" in process_plan:
                    try:
                        process_plan = operation_subagent.run(process_plan)
                        _validate_operation_plan_against_top(
                            process_plan,
                            self.last_top_level_plan,
                        )
                    except PlanValidationError as exc:
                        process_plan = top_plan_to_operation_plan(self.last_top_level_plan)
                        self.last_subagent_plan = process_plan
                        plan = compile_process_plan(process_plan)
                        plan["planning_source"] = "top_planner_rules_operation_compiler"
                        plan["planning_fallback_reason"] = (
                            f"operation subagent failed after top plan: {exc}"
                        )
                        plan["planning_attempts"] = attempt + 1
                        return plan
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
        return self._run_rule_fallback(
            prompt,
            str(last_error or PlanValidationError("Cannot create a valid tool plan.")),
        )

    def _run_rule_fallback(self, prompt: str, reason: str) -> dict[str, Any]:
        try:
            top_plan = parse_top_plan(prompt)
            process_plan = parse_operation_plan(prompt)
            if "operations" not in process_plan:
                raise RuleParseError("Rule fallback must return operations.")
            plan = compile_process_plan(process_plan)
        except PlanValidationError as exc:
            raise PlanValidationError(
                f"LLM planning failed ({reason}); rule fallback also failed: {exc}"
            ) from exc
        self.last_top_level_plan = top_plan
        self._emit_top_plan(top_plan)
        self.last_subagent_plan = process_plan
        plan["planning_source"] = "rules_fallback_compiler"
        plan["planning_fallback_reason"] = reason
        return plan

    def _emit_top_plan(self, plan: dict[str, Any]) -> None:
        if self.on_top_plan is not None:
            self.on_top_plan(plan)


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
        if action["action"] == "reset_status":
            if index == 0:
                raise PlanValidationError("reset_status cannot be the first top action.")
            previous_line = _action_output_line(actions[index - 1])
            if previous_line != action["line"]:
                raise PlanValidationError(
                    "reset_status must immediately follow an output action on the same line."
                )
            continue

        output_line = _action_output_line(action)
        if output_line is None:
            continue
        expected_reset = {"action": "reset_status", "line": output_line}
        if index + 1 >= len(actions) or actions[index + 1] != expected_reset:
            raise PlanValidationError(
                "Every produce_car, produce_phone, or move_to_output top action "
                "must be followed by reset_status on the same line."
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
                {
                    "op": "load_base",
                    "line": "A",
                    "source_line": "A",
                    "part": "car_base",
                },
                {
                    "op": "assemble",
                    "line": "A",
                    "source_line": "B",
                    "base": "car_base",
                    "part": "car_frame",
                    "supplier": "B",
                    "layer": 1,
                },
                {"op": "inspect", "line": "A"},
                {"op": "unload", "line": "A", "part": "car_base"},
            ])
        elif name == "produce_phone":
            expected.extend([
                {
                    "op": "load_base",
                    "line": "B",
                    "source_line": "A",
                    "part": "phone_base",
                },
                {
                    "op": "assemble",
                    "line": "B",
                    "source_line": "B",
                    "base": "phone_base",
                    "part": "screen",
                    "supplier": "B2",
                    "layer": 1,
                },
                {
                    "op": "assemble",
                    "line": "B",
                    "source_line": "B",
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


def _action_output_line(action: dict[str, Any]) -> str | None:
    product_line = _action_product_line(action)
    if product_line is not None:
        return product_line
    if action["action"] == "move_to_output":
        return action["line"]
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
