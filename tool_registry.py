"""Single source of truth for factory tool metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: str
    args_schema: tuple[str, ...]
    executor_name: str
    description: str
    route: tuple[str, str, str] | None = None
    line: str | None = None
    source_line: str | None = None
    target_line: str | None = None
    arm_line: str | None = None
    prompt_example: str = ""


SUPPORTED_PARTS = {
    "car_base",
    "car_frame",
    "phone_base",
    "screen",
    "camera_module",
}


def _transport(name: str, line: str, source: str, target: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        category="transport",
        args_schema=(),
        executor_name="_execute_transport_tool",
        description=f"Move line {line} shuttle from {source} to {target}.",
        route=(line, source, target),
        line=line,
    )


def _load(name: str, line: str, part_rule: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        category="load",
        args_schema=("part", "local_offset?"),
        executor_name="_execute_load_tool",
        description=f"Load {part_rule} onto line {line} at pick station.",
        line=line,
        prompt_example=f'{name}({part_rule})',
    )


def _hold(name: str, source_line: str, arm_line: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        category="hold",
        args_schema=("part",),
        executor_name="_execute_hold_tool",
        description=f"Pick a part from line {source_line} at assemble station.",
        source_line=source_line,
        arm_line=arm_line,
        prompt_example=f"{name}(part)",
    )


def _place(name: str, line: str, arm_line: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        category="place",
        args_schema=("part", "attach_to?", "layer?", "local_offset?"),
        executor_name="_execute_place_tool",
        description=f"Place a held part onto line {line} at assemble station.",
        line=line,
        arm_line=arm_line,
        prompt_example=f"{name}(part, attach_to?, layer?)",
    )


def _inspect(name: str, line: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        category="inspect",
        args_schema=(),
        executor_name="_execute_inspect_tool",
        description=f"Inspect line {line} at camera station.",
        line=line,
    )


def _unload(name: str, line: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        category="unload",
        args_schema=("part",),
        executor_name="_execute_unload_tool",
        description=f"Unload a finished part from line {line} at output station.",
        line=line,
        prompt_example=f"{name}(part)",
    )


def _cross_line(name: str, source: str, target: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        category="cross_line",
        args_schema=("part", "target_offset?"),
        executor_name="_execute_cross_line_tool",
        description=f"Move a part from line {source} shuttle to line {target} shuttle.",
        source_line=source,
        target_line=target,
        prompt_example=f"{name}(part, target_offset?)",
    )


TOOL_REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        _transport("Transport_A_Pick_Assemble", "A", "pick", "assemble"),
        _transport("Transport_A_Assemble_Pick", "A", "assemble", "pick"),
        _transport("Transport_A_Assemble_Forward", "A", "assemble", "forward"),
        _transport("Transport_A_Forward_Assemble", "A", "forward", "assemble"),
        _transport("Transport_A_Assemble_Camera", "A", "assemble", "camera"),
        _transport("Transport_A_Camera_Output", "A", "camera", "output"),
        _transport("Transport_A_Output_Pick", "A", "output", "pick"),
        _transport("Transport_A2_Clear_Pick", "A2", "clear", "pick"),
        _transport("Transport_A2_Pick_Assemble", "A2", "pick", "assemble"),
        _transport("Transport_A2_Assemble_Clear", "A2", "assemble", "clear"),
        _transport("Transport_B_Pick_Assemble", "B", "pick", "assemble"),
        _transport("Transport_B_Assemble_Pick", "B", "assemble", "pick"),
        _transport("Transport_B_Assemble_Forward", "B", "assemble", "forward"),
        _transport("Transport_B_Forward_Assemble", "B", "forward", "assemble"),
        _transport("Transport_B_Assemble_Clear", "B", "assemble", "clear"),
        _transport("Transport_B_Clear_Assemble", "B", "clear", "assemble"),
        _transport("Transport_B_Assemble_Camera", "B", "assemble", "camera"),
        _transport("Transport_B_Camera_Output", "B", "camera", "output"),
        _transport("Transport_B_Output_Pick", "B", "output", "pick"),
        _transport("Transport_B2_Clear_Pick", "B2", "clear", "pick"),
        _transport("Transport_B2_Pick_Assemble", "B2", "pick", "assemble"),
        _transport("Transport_B2_Assemble_Clear", "B2", "assemble", "clear"),
        _load("Load_A_Pick", "A", "car_base or car_frame"),
        _load("Load_A2_Pick", "A2", "car_frame"),
        _load("Load_B_Pick", "B", "phone_base or camera_module"),
        _load("Load_B2_Pick", "B2", "screen or camera_module"),
        _hold("Hold_A_Assemble", "A", "A"),
        _hold("Hold_A2_Assemble", "A2", "A"),
        _hold("Hold_B_Assemble", "B", "B"),
        _hold("Hold_B2_Assemble", "B2", "B"),
        _place("Place_A_Assemble", "A", "A"),
        _place("Place_B_Assemble", "B", "B"),
        _inspect("Inspect_A", "A"),
        _inspect("Inspect_B", "B"),
        _unload("Unload_A_Output", "A"),
        _unload("Unload_B_Output", "B"),
        _cross_line("Transport_A_B", "A", "B"),
        _cross_line("Transport_B_A", "B", "A"),
    )
}

ALLOWED_TOOLS = set(TOOL_REGISTRY)
TRANSPORT_TOOLS = {name for name, spec in TOOL_REGISTRY.items() if spec.category == "transport"}
LOAD_TOOLS = {name for name, spec in TOOL_REGISTRY.items() if spec.category == "load"}
HOLD_TOOLS = {name for name, spec in TOOL_REGISTRY.items() if spec.category == "hold"}
PLACE_TOOLS = {name for name, spec in TOOL_REGISTRY.items() if spec.category == "place"}
INSPECT_TOOLS = {name for name, spec in TOOL_REGISTRY.items() if spec.category == "inspect"}
UNLOAD_TOOLS = {name for name, spec in TOOL_REGISTRY.items() if spec.category == "unload"}
CROSS_LINE_TOOLS = {name for name, spec in TOOL_REGISTRY.items() if spec.category == "cross_line"}
TRANSPORT_TOOL_ROUTES = {
    name: spec.route
    for name, spec in TOOL_REGISTRY.items()
    if spec.route is not None
}
PLAN_TRANSPORT_ROUTES = TRANSPORT_TOOL_ROUTES


def build_tool_prompt() -> str:
    lines = ["Use only these tools:"]
    for name in sorted(TOOL_REGISTRY):
        spec = TOOL_REGISTRY[name]
        args = ", ".join(spec.args_schema)
        signature = f"{name}({args})" if args else name
        lines.append(f"- {signature}: {spec.description}")
    return "\n".join(lines)
