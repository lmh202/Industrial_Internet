"""LLM production planner for tool-based factory control."""

from __future__ import annotations

import json
from typing import Any

from config import API_KEY, BASE_MODEL, BASE_URL, LLM_KEEP_ALIVE, LLM_TIMEOUT
from llm_client import LLMError, OpenAICompatibleClient


SUPPORTED_PARTS = {
    "car_base",
    "car_frame",
    "phone_base",
    "screen",
    "camera_module",
}

TRANSPORT_TOOLS = {
    "Transport_A_Pick_Assemble",
    "Transport_A_Assemble_Pick",
    "Transport_A_Assemble_Camera",
    "Transport_A_Camera_Output",
    "Transport_A_Output_Pick",
    "Transport_B_Pick_Assemble",
    "Transport_B_Assemble_Pick",
    "Transport_B_Assemble_Clear",
    "Transport_B_Clear_Assemble",
    "Transport_B_Assemble_Camera",
    "Transport_B_Camera_Output",
    "Transport_B_Output_Pick",
    "Transport_B2_Clear_Pick",
    "Transport_B2_Pick_Assemble",
    "Transport_B2_Assemble_Clear",
}
LOAD_TOOLS = {"Load_A_Pick", "Load_B_Pick", "Load_B2_Pick"}
HOLD_TOOLS = {"Hold_A_Assemble", "Hold_B_Assemble", "Hold_B2_Assemble"}
PLACE_TOOLS = {"Place_A_Assemble", "Place_B_Assemble"}
INSPECT_TOOLS = {"Inspect_A", "Inspect_B"}
UNLOAD_TOOLS = {"Unload_A_Output", "Unload_B_Output"}
CROSS_LINE_TOOLS = {"Transport_A_B", "Transport_B_A"}

ALLOWED_TOOLS = (
    TRANSPORT_TOOLS
    | LOAD_TOOLS
    | HOLD_TOOLS
    | PLACE_TOOLS
    | INSPECT_TOOLS
    | UNLOAD_TOOLS
    | CROSS_LINE_TOOLS
)

PLAN_TRANSPORT_ROUTES = {
    "Transport_A_Pick_Assemble": ("A", "pick", "assemble"),
    "Transport_A_Assemble_Pick": ("A", "assemble", "pick"),
    "Transport_A_Assemble_Camera": ("A", "assemble", "camera"),
    "Transport_A_Camera_Output": ("A", "camera", "output"),
    "Transport_A_Output_Pick": ("A", "output", "pick"),
    "Transport_B_Pick_Assemble": ("B", "pick", "assemble"),
    "Transport_B_Assemble_Pick": ("B", "assemble", "pick"),
    "Transport_B_Assemble_Clear": ("B", "assemble", "clear"),
    "Transport_B_Clear_Assemble": ("B", "clear", "assemble"),
    "Transport_B_Assemble_Camera": ("B", "assemble", "camera"),
    "Transport_B_Camera_Output": ("B", "camera", "output"),
    "Transport_B_Output_Pick": ("B", "output", "pick"),
    "Transport_B2_Clear_Pick": ("B2", "clear", "pick"),
    "Transport_B2_Pick_Assemble": ("B2", "pick", "assemble"),
    "Transport_B2_Assemble_Clear": ("B2", "assemble", "clear"),
}

LINE_PARTS = {
    "A": {"car_base", "car_frame"},
    "B": {"phone_base", "screen", "camera_module"},
}

DEFAULT_LINE_PART = {
    "A": "car_base",
    "B": "phone_base",
}

PART_ALIASES = {
    "car_base": ("car_base", "part_car_base", "底盘", "车底", "车辆底座", "汽车底座"),
    "car_frame": ("car_frame", "part_car_frame", "车架", "车身", "框架"),
    "phone_base": ("phone_base", "part_phone", "手机机身", "手机底座", "机身"),
    "screen": ("screen", "part_screen", "屏幕", "手机屏幕"),
    "camera_module": (
        "camera_module",
        "part_camera_module",
        "摄像头模组",
        "摄像头",
        "相机模组",
        "相机",
    ),
}


SYSTEM_PROMPT = """
Return one JSON object only: {"plan_name":"...","steps":[{"tool":"...","args":{}}]}.
No markdown, no prose, no code, no loops, no make_car/make_phone.
Use only these tools: Load_A_Pick, Load_B_Pick, Transport_A_Pick_Assemble,
Transport_A_Assemble_Pick, Transport_A_Assemble_Camera,
Transport_A_Camera_Output, Transport_B_Pick_Assemble,
Transport_B_Assemble_Pick, Transport_B_Assemble_Clear,
Transport_B_Clear_Assemble, Transport_B_Assemble_Camera,
Transport_B_Camera_Output, Transport_A_Output_Pick, Transport_B_Output_Pick,
Transport_B2_Clear_Pick,
Transport_B2_Pick_Assemble, Transport_B2_Assemble_Clear,
Hold_A_Assemble, Hold_B_Assemble, Hold_B2_Assemble, Place_A_Assemble,
Place_B_Assemble, Inspect_A, Inspect_B, Unload_A_Output, Unload_B_Output,
Transport_A_B, Transport_B_A.
Parts only: car_base, car_frame, phone_base, screen, camera_module.
Never output coordinates, robot names, handles, or non-tool actions.
Expand quantities into repeated concrete steps.
Screen loading rule: screen must always be loaded with Load_B2_Pick after
Transport_B2_Clear_Pick. Never use Load_B_Pick(screen).
Every repeated product must begin with its main shuttle at pick. If another
product on the same line follows an Unload_*_Output step, insert exactly one
Transport_A_Output_Pick or Transport_B_Output_Pick before the next Load_*_Pick.
For two phones, the first Unload_B_Output(phone_base) must be followed by
Transport_B_Output_Pick before starting the second phone with Load_B_Pick.
For two cars, the first Unload_A_Output(car_base) must be followed by
Transport_A_Output_Pick before starting the second car with Load_A_Pick.

Car plan on A:
Load_A_Pick(car_frame), Transport_A_Pick_Assemble,
Hold_A_Assemble(car_frame), Transport_A_Assemble_Pick,
Load_A_Pick(car_base), Transport_A_Pick_Assemble,
Place_A_Assemble(car_frame, attach_to=car_base, layer=1),
Transport_A_Assemble_Camera, Inspect_A, Transport_A_Camera_Output,
Unload_A_Output(car_base).

Phone plan on B:
Load_B_Pick(camera_module), Transport_B_Pick_Assemble,
Hold_B_Assemble(camera_module), Transport_B_Assemble_Pick,
Load_B_Pick(phone_base), Transport_B_Pick_Assemble,
Place_B_Assemble(camera_module, attach_to=phone_base, layer=1),
Transport_B_Assemble_Clear,
Transport_B2_Clear_Pick, Load_B2_Pick(screen), Transport_B2_Pick_Assemble,
Hold_B2_Assemble(screen), Transport_B2_Assemble_Clear,
Transport_B_Clear_Assemble,
Place_B_Assemble(screen, attach_to=phone_base, layer=2),
Transport_B_Assemble_Camera, Inspect_B, Transport_B_Camera_Output,
Unload_B_Output(phone_base).

Move one phone-line part to output:
- phone_base or camera_module:
Load_B_Pick(part), Transport_B_Pick_Assemble, Transport_B_Assemble_Camera,
Transport_B_Camera_Output, Unload_B_Output(part).
- screen:
Transport_B_Pick_Assemble, Transport_B2_Clear_Pick, Load_B2_Pick(screen),
Transport_B2_Pick_Assemble, Hold_B2_Assemble(screen),
Transport_B2_Assemble_Clear, Place_B_Assemble(screen, layer=0),
Transport_B_Assemble_Camera, Transport_B_Camera_Output, Unload_B_Output(screen).
If the user says an unspecified phone-line part, use phone_base.

Move one car-line part to output:
Load_A_Pick(part), Transport_A_Pick_Assemble, Transport_A_Assemble_Camera,
Transport_A_Camera_Output, Unload_A_Output(part).
If the user says an unspecified car-line part, use car_base.
"""


class PlanValidationError(ValueError):
    """Raised when an LLM plan is not executable by the tool whitelist."""


TaskParseError = PlanValidationError


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

        rule_plan = rule_plan_from_prompt(prompt)
        if rule_plan is not None:
            plan = validate_plan(rule_plan)
            validate_plan_sequence(plan)
            return plan

        if not self.llm_client.is_configured:
            raise PlanValidationError("LLM is not configured.")

        planning_prompt = prompt
        last_error: PlanValidationError | None = None
        for attempt in range(3):
            try:
                payload = self.llm_client.chat_json(SYSTEM_PROMPT, planning_prompt)
            except LLMError as exc:
                raise PlanValidationError(f"Cannot create tool plan: {exc}") from exc
            try:
                plan = validate_plan(payload)
                validate_plan_sequence(plan)
                return plan
            except PlanValidationError as exc:
                last_error = exc
                if attempt == 2:
                    break
                planning_prompt = (
                    f"{prompt}\n\n"
                    f"Previous JSON plan was invalid: {exc}. "
                    "Return a corrected JSON object only. Do not explain."
                )
        raise last_error or PlanValidationError("Cannot create a valid tool plan.")


ProductionAgent = ProductionPlanner


def rule_plan_from_prompt(prompt: str) -> dict[str, Any] | None:
    text = _normalize_prompt(prompt)
    if not _looks_like_move_to_output(text):
        return None

    line = _detect_line(text)
    part = _detect_part(text)
    if line is None and part is not None:
        line = _line_for_part(part)
    if line is None:
        return None
    if part is None:
        part = DEFAULT_LINE_PART[line]
    if part not in LINE_PARTS[line]:
        return None
    return _move_part_to_output_plan(line, part)


def _normalize_prompt(prompt: str) -> str:
    return (
        prompt.strip()
        .lower()
        .replace(" ", "")
        .replace("，", ",")
        .replace("。", ".")
    )


def _looks_like_move_to_output(text: str) -> bool:
    has_move = any(word in text for word in (
        "移动", "移到", "移至", "搬到", "搬运", "送到", "送至", "放到", "放入", "运到"
    ))
    has_output = any(word in text for word in (
        "output", "输出", "出料", "输出区", "出料区", "成品区"
    ))
    has_part_context = any(word in text for word in (
        "零件", "部件", "part", "底盘", "车架", "车身", "屏幕", "摄像头", "机身"
    ))
    return has_move and has_output and has_part_context


def _detect_line(text: str) -> str | None:
    if any(word in text for word in ("手机产线", "手机线", "b线", "b产线", "phone_line")):
        return "B"
    if any(word in text for word in ("车辆产线", "汽车产线", "车产线", "a线", "a产线", "car_line")):
        return "A"
    return None


def _detect_part(text: str) -> str | None:
    for part, aliases in PART_ALIASES.items():
        if any(alias in text for alias in aliases):
            return part
    return None


def _line_for_part(part: str) -> str:
    for line, parts in LINE_PARTS.items():
        if part in parts:
            return line
    raise PlanValidationError(f"Unsupported part: {part}")


def _move_part_to_output_plan(line: str, part: str) -> dict[str, Any]:
    if line == "A":
        steps = [
            {"tool": "Load_A_Pick", "args": {"part": part}},
            {"tool": "Transport_A_Pick_Assemble", "args": {}},
            {"tool": "Transport_A_Assemble_Camera", "args": {}},
            {"tool": "Transport_A_Camera_Output", "args": {}},
            {"tool": "Unload_A_Output", "args": {"part": part}},
        ]
    elif part == "screen":
        steps = [
            {"tool": "Transport_B_Pick_Assemble", "args": {}},
            {"tool": "Transport_B2_Clear_Pick", "args": {}},
            {"tool": "Load_B2_Pick", "args": {"part": "screen"}},
            {"tool": "Transport_B2_Pick_Assemble", "args": {}},
            {"tool": "Hold_B2_Assemble", "args": {"part": "screen"}},
            {"tool": "Transport_B2_Assemble_Clear", "args": {}},
            {"tool": "Place_B_Assemble", "args": {"part": "screen", "layer": 0}},
            {"tool": "Transport_B_Assemble_Camera", "args": {}},
            {"tool": "Transport_B_Camera_Output", "args": {}},
            {"tool": "Unload_B_Output", "args": {"part": "screen"}},
        ]
    else:
        steps = [
            {"tool": "Load_B_Pick", "args": {"part": part}},
            {"tool": "Transport_B_Pick_Assemble", "args": {}},
            {"tool": "Transport_B_Assemble_Camera", "args": {}},
            {"tool": "Transport_B_Camera_Output", "args": {}},
            {"tool": "Unload_B_Output", "args": {"part": part}},
        ]
    return {
        "plan_name": f"move_{line.lower()}_{part}_to_output",
        "steps": steps,
    }


def validate_plan(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PlanValidationError("Plan must be a JSON object.")

    plan_name = payload.get("plan_name")
    if not isinstance(plan_name, str) or not plan_name.strip():
        raise PlanValidationError("Plan must include a non-empty plan_name.")

    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise PlanValidationError("Plan must include a non-empty steps list.")

    clean_steps = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise PlanValidationError(f"Step {index} must be an object.")
        tool = step.get("tool")
        args = step.get("args")
        if not isinstance(tool, str) or tool not in ALLOWED_TOOLS:
            raise PlanValidationError(f"Step {index} has unknown tool: {tool!r}")
        if not isinstance(args, dict):
            raise PlanValidationError(f"Step {index} args must be an object.")
        clean_steps.append({"tool": tool, "args": _validate_tool_args(index, tool, args)})

    return {"plan_name": plan_name.strip(), "steps": clean_steps}


def validate_plan_sequence(plan: dict[str, Any]) -> None:
    """Validate station and holding state without touching CoppeliaSim."""
    station = {"A": "pick", "B": "pick", "B2": "clear"}
    parts = {"A": set(), "B": set(), "B2": set()}
    holding = {"A": None, "B": None, "mid": None}

    for index, step in enumerate(plan["steps"], start=1):
        tool = step["tool"]
        args = step["args"]

        if tool in PLAN_TRANSPORT_ROUTES:
            line, source, target = PLAN_TRANSPORT_ROUTES[tool]
            _require_station(index, tool, line, station[line], source)
            station[line] = target
            continue

        if tool in LOAD_TOOLS:
            line = _line_from_load_tool(tool)
            _require_station(index, tool, line, station[line], "pick")
            parts[line].add(args["part"])
            continue

        if tool in HOLD_TOOLS:
            source_line, arm = _hold_source_and_arm(tool)
            _require_station(index, tool, source_line, station[source_line], "assemble")
            part = args["part"]
            if holding[arm] is not None:
                raise PlanValidationError(
                    f"Step {index} {tool} cannot hold {part}; arm {arm} already holds "
                    f"{holding[arm]}."
                )
            if part not in parts[source_line]:
                raise PlanValidationError(
                    f"Step {index} {tool} cannot hold {part}; line {source_line} shuttle "
                    "does not contain it."
                )
            parts[source_line].remove(part)
            holding[arm] = part
            continue

        if tool in PLACE_TOOLS:
            line, arm = _place_line_and_arm(tool)
            _require_station(index, tool, line, station[line], "assemble")
            part = args["part"]
            if holding[arm] != part:
                raise PlanValidationError(
                    f"Step {index} {tool} cannot place {part}; arm {arm} holds "
                    f"{holding[arm]}."
                )
            attach_to = args.get("attach_to")
            if attach_to is not None and attach_to not in parts[line]:
                raise PlanValidationError(
                    f"Step {index} {tool} cannot attach to {attach_to}; line {line} "
                    "shuttle does not contain it."
                )
            holding[arm] = None
            parts[line].add(part)
            continue

        if tool in INSPECT_TOOLS:
            line = tool.removeprefix("Inspect_")
            _require_station(index, tool, line, station[line], "camera")
            continue

        if tool in UNLOAD_TOOLS:
            line = "A" if tool.startswith("Unload_A_") else "B"
            _require_station(index, tool, line, station[line], "output")
            part = args["part"]
            if part not in parts[line]:
                raise PlanValidationError(
                    f"Step {index} {tool} cannot unload {part}; line {line} shuttle "
                    "does not contain it."
                )
            parts[line].remove(part)
            continue


def _validate_tool_args(index: int, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool in TRANSPORT_TOOLS or tool in INSPECT_TOOLS:
        _reject_unknown_args(index, tool, args, set())
        return {}

    if tool in LOAD_TOOLS:
        _reject_unknown_args(index, tool, args, {"part", "local_offset"})
        part = _required_part(index, args)
        if tool == "Load_B_Pick" and part == "screen":
            raise PlanValidationError(
                f"Step {index} Load_B_Pick cannot load screen; use Load_B2_Pick(screen)."
            )
        if tool == "Load_B2_Pick" and part != "screen":
            raise PlanValidationError(
                f"Step {index} Load_B2_Pick can only load screen, not {part}."
            )
        clean = {"part": part}
        if "local_offset" in args:
            clean["local_offset"] = _offset(index, tool, args["local_offset"])
        return clean

    if tool in HOLD_TOOLS or tool in UNLOAD_TOOLS:
        _reject_unknown_args(index, tool, args, {"part"})
        return {"part": _required_part(index, args)}

    if tool in PLACE_TOOLS:
        _reject_unknown_args(index, tool, args, {"part", "attach_to", "layer", "local_offset"})
        clean = {"part": _required_part(index, args)}
        if "attach_to" in args:
            clean["attach_to"] = _part_value(index, args["attach_to"], "attach_to")
        if "layer" in args:
            layer = args["layer"]
            if isinstance(layer, bool) or not isinstance(layer, (int, float)):
                raise PlanValidationError(f"Step {index} layer must be numeric.")
            if layer < 0:
                raise PlanValidationError(f"Step {index} layer must be non-negative.")
            clean["layer"] = layer
        if "local_offset" in args:
            clean["local_offset"] = _offset(index, tool, args["local_offset"])
        return clean

    if tool in CROSS_LINE_TOOLS:
        _reject_unknown_args(index, tool, args, {"part", "target_offset"})
        clean = {"part": _required_part(index, args)}
        if "target_offset" in args:
            clean["target_offset"] = _offset(index, tool, args["target_offset"])
        return clean

    raise PlanValidationError(f"Step {index} has unsupported tool: {tool}")


def _reject_unknown_args(index: int, tool: str, args: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(args) - allowed
    if unknown:
        raise PlanValidationError(
            f"Step {index} {tool} has unsupported args: {sorted(unknown)}"
        )


def _required_part(index: int, args: dict[str, Any]) -> str:
    if "part" not in args:
        raise PlanValidationError(f"Step {index} missing required part.")
    return _part_value(index, args["part"], "part")


def _part_value(index: int, value: Any, field: str) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_PARTS:
        raise PlanValidationError(f"Step {index} has invalid {field}: {value!r}")
    return value


def _offset(index: int, tool: str, value: Any) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise PlanValidationError(f"Step {index} {tool} offset must be [x, y].")
    return [float(value[0]), float(value[1])]


def _require_station(index: int, tool: str, line: str, current: str, expected: str) -> None:
    if current != expected:
        raise PlanValidationError(
            f"Step {index} {tool} requires line {line} at {expected}, but it is {current}."
        )


def _line_from_load_tool(tool: str) -> str:
    if tool == "Load_A_Pick":
        return "A"
    if tool == "Load_B_Pick":
        return "B"
    return "B2"


def _hold_source_and_arm(tool: str) -> tuple[str, str]:
    if tool == "Hold_A_Assemble":
        return "A", "A"
    if tool == "Hold_B_Assemble":
        return "B", "B"
    return "B2", "B"


def _place_line_and_arm(tool: str) -> tuple[str, str]:
    if tool == "Place_A_Assemble":
        return "A", "A"
    return "B", "B"


def plan_to_json(plan: dict[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False)


def tasks_to_json(plan: dict[str, Any]) -> str:
    return plan_to_json(plan)
