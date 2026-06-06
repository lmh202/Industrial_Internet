"""In-memory state model used to validate tool-plan sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tool_registry import (
    CROSS_LINE_TOOLS,
    HOLD_TOOLS,
    INSPECT_TOOLS,
    LOAD_TOOLS,
    PLACE_TOOLS,
    TRANSPORT_TOOL_ROUTES,
    UNLOAD_TOOLS,
)


class PlanStateError(ValueError):
    """Raised when a tool cannot be applied to the current abstract state."""


@dataclass
class PlanState:
    station: dict[str, str] = field(default_factory=lambda: {
        "A": "pick",
        "A2": "clear",
        "B": "pick",
        "B2": "clear",
    })
    parts: dict[str, set[str]] = field(default_factory=lambda: {
        "A": set(),
        "A2": set(),
        "B": set(),
        "B2": set(),
    })
    holding: dict[str, str | None] = field(default_factory=lambda: {
        "A": None,
        "B": None,
        "mid": None,
    })

    def apply_step(self, index: int, tool: str, args: dict[str, Any]) -> None:
        if tool in TRANSPORT_TOOL_ROUTES:
            line, source, target = TRANSPORT_TOOL_ROUTES[tool]
            self._require_station(index, tool, line, source)
            self.station[line] = target
            return

        if tool in LOAD_TOOLS:
            line = _line_from_load_tool(tool)
            self._require_station(index, tool, line, "pick")
            self.parts[line].add(args["part"])
            return

        if tool in HOLD_TOOLS:
            source_line, arm = _hold_source_and_arm(tool)
            self._require_station(index, tool, source_line, "assemble")
            part = args["part"]
            if self.holding[arm] is not None:
                raise PlanStateError(
                    f"Step {index} {tool} cannot hold {part}; arm {arm} already holds "
                    f"{self.holding[arm]}."
                )
            if part not in self.parts[source_line]:
                raise PlanStateError(
                    f"Step {index} {tool} cannot hold {part}; line {source_line} shuttle "
                    "does not contain it."
                )
            self.parts[source_line].remove(part)
            self.holding[arm] = part
            return

        if tool in PLACE_TOOLS:
            line, arm = _place_line_and_arm(tool)
            self._require_station(index, tool, line, "assemble")
            part = args["part"]
            if self.holding[arm] != part:
                raise PlanStateError(
                    f"Step {index} {tool} cannot place {part}; arm {arm} holds "
                    f"{self.holding[arm]}."
                )
            attach_to = args.get("attach_to")
            if attach_to is not None and attach_to not in self.parts[line]:
                raise PlanStateError(
                    f"Step {index} {tool} cannot attach to {attach_to}; line {line} "
                    "shuttle does not contain it."
                )
            self.holding[arm] = None
            self.parts[line].add(part)
            return

        if tool in INSPECT_TOOLS:
            line = tool.removeprefix("Inspect_")
            self._require_station(index, tool, line, "camera")
            return

        if tool in UNLOAD_TOOLS:
            line = "A" if tool.startswith("Unload_A_") else "B"
            self._require_station(index, tool, line, "output")
            part = args["part"]
            if part not in self.parts[line]:
                raise PlanStateError(
                    f"Step {index} {tool} cannot unload {part}; line {line} shuttle "
                    "does not contain it."
                )
            self.parts[line].remove(part)
            return

        if tool in CROSS_LINE_TOOLS:
            source, target = _cross_line_source_target(tool)
            part = args["part"]
            if part not in self.parts[source]:
                raise PlanStateError(
                    f"Step {index} {tool} cannot transfer {part}; line {source} shuttle "
                    "does not contain it."
                )
            self.parts[source].remove(part)
            self.parts[target].add(part)
            self.station[source] = "assemble"
            self.station[target] = "assemble"
            return

    def _require_station(self, index: int, tool: str, line: str, expected: str) -> None:
        current = self.station[line]
        if current != expected:
            raise PlanStateError(
                f"Step {index} {tool} requires line {line} at {expected}, but it is {current}."
            )


def _line_from_load_tool(tool: str) -> str:
    if tool == "Load_A_Pick":
        return "A"
    if tool == "Load_A2_Pick":
        return "A2"
    if tool == "Load_B_Pick":
        return "B"
    return "B2"


def _hold_source_and_arm(tool: str) -> tuple[str, str]:
    if tool == "Hold_A_Assemble":
        return "A", "A"
    if tool == "Hold_A2_Assemble":
        return "A2", "A"
    if tool == "Hold_B_Assemble":
        return "B", "B"
    return "B2", "B"


def _place_line_and_arm(tool: str) -> tuple[str, str]:
    if tool == "Place_A_Assemble":
        return "A", "A"
    return "B", "B"


def _cross_line_source_target(tool: str) -> tuple[str, str]:
    if tool == "Transport_A_B":
        return "A", "B"
    return "B", "A"
