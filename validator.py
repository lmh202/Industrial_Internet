"""Plan shape and sequence validation."""

from __future__ import annotations

from typing import Any

from state import PlanState, PlanStateError
from tool_registry import (
    ALLOWED_TOOLS,
    CROSS_LINE_TOOLS,
    HOLD_TOOLS,
    INSPECT_TOOLS,
    LOAD_TOOLS,
    PLACE_TOOLS,
    SUPPORTED_PARTS,
    TRANSPORT_TOOLS,
    UNLOAD_TOOLS,
)


class PlanValidationError(ValueError):
    """Raised when a plan is not executable by the tool whitelist."""


TaskParseError = PlanValidationError


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

    clean: dict[str, Any] = {"plan_name": plan_name.strip(), "steps": clean_steps}
    assumptions = payload.get("assumptions")
    if assumptions is not None:
        if not isinstance(assumptions, list) or not all(
            isinstance(item, str) and item.strip() for item in assumptions
        ):
            raise PlanValidationError("Plan assumptions must be a list of non-empty strings.")
        clean["assumptions"] = [item.strip() for item in assumptions]
    return clean


def validate_plan_sequence(plan: dict[str, Any]) -> None:
    """Validate station and holding state without touching CoppeliaSim."""
    state = PlanState()
    for index, step in enumerate(plan["steps"], start=1):
        try:
            state.apply_step(index, step["tool"], step["args"])
        except PlanStateError as exc:
            raise PlanValidationError(str(exc)) from exc


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
