"""Keyword-based fallback operation planner.

This module is intentionally narrow: it produces the same process-plan shape
as subagent_plan.json, then lets process_compiler own all low-level tool steps.
"""

from __future__ import annotations

import re
from typing import Any

from process_compiler import validate_process_plan
from validator import PlanValidationError


class RuleParseError(PlanValidationError):
    """Raised when the fallback rules cannot infer a supported instruction."""


_NUMBER_WORDS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "俩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


def parse_operation_plan(prompt: str) -> dict[str, Any]:
    """Parse user text into an operation plan matching subagent_plan.json."""
    top_plan = parse_top_plan(prompt)
    return top_plan_to_operation_plan(top_plan)


def parse_top_plan(prompt: str) -> dict[str, Any]:
    """Parse user text into a top-plan actions object matching top_plan.json."""
    text = _normalize(prompt)
    if not text:
        raise RuleParseError("Rule fallback cannot parse an empty prompt.")

    if _requests_move_to_output(text) and not _requests_production(text):
        move_action = _move_action(text)
        plan = {
            "plan_name": _move_plan_name(text),
            "strategy": "sequential",
            "actions": [move_action, {"action": "reset_status", "line": move_action["line"]}],
        }
        return validate_process_plan(plan)

    car_count = _product_count(text, "car")
    phone_count = _product_count(text, "phone")
    if car_count == 0 and phone_count == 0:
        raise RuleParseError(f"Rule fallback cannot infer a supported product: {prompt}")

    strategy = "parallel_start" if _requests_parallel(text) and car_count and phone_count else "sequential"
    actions: list[dict[str, Any]] = []
    line_has_output = {"A": False, "B": False}

    product_order = _product_order(text, car_count, phone_count)
    for product in product_order:
        line = "A" if product == "car" else "B"
        actions.append({"action": f"produce_{product}"})
        actions.append({"action": "reset_status", "line": line})
        line_has_output[line] = False

    plan = {
        "plan_name": _plan_name(car_count, phone_count, strategy),
        "strategy": strategy,
        "actions": actions,
    }
    return validate_process_plan(plan)


def top_plan_to_operation_plan(top_plan: dict[str, Any]) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    for action in top_plan["actions"]:
        name = action["action"]
        if name == "produce_car":
            operations.extend(_product_operations("car"))
        elif name == "produce_phone":
            operations.extend(_product_operations("phone"))
        elif name == "reset_status":
            operations.append({"op": "reset_status", "line": action["line"]})
        elif name == "move_to_output":
            operations.append({
                "op": "move_to_output",
                "line": action["line"],
                "part": action["part"],
            })
    return validate_process_plan({
        "plan_name": top_plan["plan_name"],
        "strategy": top_plan["strategy"],
        "operations": operations,
    })


def _normalize(prompt: str) -> str:
    return prompt.lower().replace(" ", "").replace("\t", "")


def _requests_parallel(text: str) -> bool:
    return any(word in text for word in ("同时", "同步", "并行", "parallel", "simultaneous"))


def _requests_production(text: str) -> bool:
    return any(word in text for word in (
        "生产", "制造", "装配", "组装", "造", "produce", "make", "manufacture"
    ))


def _requests_move_to_output(text: str) -> bool:
    has_output = any(word in text for word in ("output", "输出", "出料", "输出区", "出料区"))
    has_move = any(word in text for word in (
        "移", "搬", "送", "放", "运", "move", "transport", "send"
    ))
    return has_output and has_move


def _product_count(text: str, product: str) -> int:
    keywords = {
        "car": ("汽车", "小车", "车", "car", "vehicle"),
        "phone": ("手机", "phone"),
    }[product]
    positions = [text.find(keyword) for keyword in keywords if keyword in text]
    if not positions:
        return 0
    position = min(index for index in positions if index >= 0)
    prefix = text[max(0, position - 8):position]
    explicit = _count_near_product(prefix)
    return explicit or 1


def _count_near_product(prefix: str) -> int | None:
    digit_match = re.search(r"(\d+)(?:个|部|辆|台|件)?$", prefix)
    if digit_match:
        value = int(digit_match.group(1))
        return value if value > 0 else None
    for word, value in _NUMBER_WORDS.items():
        if prefix.endswith(word) or prefix.endswith(f"{word}个") or prefix.endswith(f"{word}部") or prefix.endswith(f"{word}辆"):
            return value
    return None


def _product_order(text: str, car_count: int, phone_count: int) -> list[str]:
    entries: list[tuple[int, str, int]] = []
    car_pos = _first_position(text, ("汽车", "小车", "车", "car", "vehicle"))
    phone_pos = _first_position(text, ("手机", "phone"))
    if car_count:
        entries.append((car_pos, "car", car_count))
    if phone_count:
        entries.append((phone_pos, "phone", phone_count))
    entries.sort(key=lambda item: item[0])
    products: list[str] = []
    for _pos, product, count in entries:
        products.extend([product] * count)
    return products


def _first_position(text: str, keywords: tuple[str, ...]) -> int:
    positions = [text.find(keyword) for keyword in keywords if keyword in text]
    return min(positions) if positions else len(text)


def _product_operations(product: str) -> list[dict[str, Any]]:
    if product == "car":
        return [
            {"op": "load_base", "line": "A", "source_line": "A", "part": "car_base"},
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
        ]
    return [
        {"op": "load_base", "line": "B", "source_line": "A", "part": "phone_base"},
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
    ]


def _move_operation(text: str) -> dict[str, Any]:
    part = _part_from_text(text)
    line = "A" if part in {"car_base", "phone_base"} else "B"
    return {"op": "move_to_output", "line": line, "part": part}


def _move_action(text: str) -> dict[str, Any]:
    part = _part_from_text(text)
    line = "A" if part in {"car_base", "phone_base"} else "B"
    return {"action": "move_to_output", "line": line, "part": part}


def _part_from_text(text: str) -> str:
    part_keywords = (
        ("camera_module", ("camera_module", "camera", "相机", "摄像头", "镜头")),
        ("screen", ("screen", "屏幕", "显示屏")),
        ("car_frame", ("car_frame", "车架", "外壳", "车壳")),
        ("car_base", ("car_base", "汽车底座", "车底座", "车基座")),
        ("phone_base", ("phone_base", "手机底座", "手机机身", "手机产线", "手机")),
    )
    for part, keywords in part_keywords:
        if any(keyword in text for keyword in keywords):
            return part
    if "a" in text or "汽车" in text or "车" in text:
        return "car_base"
    if "b" in text or "手机" in text:
        return "phone_base"
    raise RuleParseError("Rule fallback cannot infer which part should move to output.")


def _move_plan_name(text: str) -> str:
    return f"move_{_part_from_text(text)}_to_output"


def _plan_name(car_count: int, phone_count: int, strategy: str) -> str:
    prefix = "parallel_start" if strategy == "parallel_start" else "produce"
    return f"{prefix}_{car_count}_car_{phone_count}_phone"
