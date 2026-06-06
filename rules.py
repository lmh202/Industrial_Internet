"""Deterministic rule planner for common shop-floor commands."""

from __future__ import annotations

from typing import Any

from process_compiler import compile_process_plan
from tool_registry import SUPPORTED_PARTS
from validator import PlanValidationError


LINE_PARTS = {
    "A": {"car_base", "car_frame"},
    "B": {"phone_base", "screen", "camera_module"},
}

DEFAULT_LINE_PART = {
    "A": "car_base",
    "B": "phone_base",
}

PART_ALIASES = {
    "car_base": (
        "car_base",
        "part_car_base",
        "底盘",
        "车底",
        "汽车底座",
        "车辆底座",
        "搴曠洏",
    ),
    "car_frame": (
        "car_frame",
        "part_car_frame",
        "车架",
        "车身",
        "框架",
        "杞︽灦",
        "鏋",
    ),
    "phone_base": (
        "phone_base",
        "part_phone",
        "手机机身",
        "手机底座",
        "机身",
        "鎵嬫満鏈鸿韩",
        "鏈鸿韩",
    ),
    "screen": (
        "screen",
        "part_screen",
        "屏幕",
        "手机屏幕",
        "灞忓箷",
        "睆",
    ),
    "camera_module": (
        "camera_module",
        "part_camera_module",
        "摄像头模块",
        "摄像头",
        "相机模组",
        "相机",
        "鎽勫儚",
        "鐩告満",
    ),
}


def rule_plan_from_prompt(prompt: str) -> dict[str, Any] | None:
    process_plan = rule_process_plan_from_prompt(prompt)
    if process_plan is None:
        return None
    return compile_process_plan(process_plan)


def rule_process_plan_from_prompt(prompt: str) -> dict[str, Any] | None:
    text = _normalize_prompt(prompt)
    production_plan = _production_process_from_text(text)
    if production_plan is not None:
        return production_plan

    if not _looks_like_move_to_output(text):
        return None

    line = _detect_line(text)
    part = _detect_part(text)
    assumptions = []
    if line is None and part is not None:
        line = _line_for_part(part)
    if line is None:
        return None
    if part is None:
        part = DEFAULT_LINE_PART[line]
        assumptions.append(f"Unspecified line {line} part defaults to {part}.")
    if part not in LINE_PARTS[line]:
        return None
    return _move_part_to_output_process(line, part, assumptions)


def _normalize_prompt(prompt: str) -> str:
    return prompt.strip().lower().replace(" ", "")


def _production_process_from_text(text: str) -> dict[str, Any] | None:
    if not _looks_like_production(text):
        return None

    products = []
    car_aliases = ("car", "\u8f66", "\u6c7d\u8f66", "\u8f66\u8f86")
    phone_aliases = ("phone", "\u624b\u673a")
    car_count = _product_count(text, car_aliases)
    phone_count = _product_count(text, phone_aliases)
    if car_count:
        products.append(("car", car_count, _first_product_index(text, car_aliases)))
    if phone_count:
        products.append(("phone", phone_count, _first_product_index(text, phone_aliases)))
    if not products:
        return None

    if _looks_like_parallel_start(text) and car_count and phone_count:
        return {
            "plan_name": f"parallel_start_{car_count}_car_{phone_count}_phone",
            "strategy": "parallel_start",
            "actions": _production_actions(products),
            "assumptions": [
                "Car and phone line steps are interleaved to start both lines early.",
            ],
        }

    name_parts = []
    if car_count:
        name_parts.append(f"{car_count}_car")
    if phone_count:
        name_parts.append(f"{phone_count}_phone")
    return {
        "plan_name": "produce_" + "_".join(name_parts),
        "strategy": "sequential",
        "actions": _production_actions(products),
    }


def _production_actions(products: list[tuple[str, int, int]]) -> list[dict[str, Any]]:
    actions = []
    line_has_output = {"A": False, "B": False}
    for product, quantity, _index in sorted(products, key=lambda item: item[2]):
        line = "A" if product == "car" else "B"
        for _ in range(quantity):
            if line_has_output[line]:
                actions.append({"action": "reset_status", "line": line})
                line_has_output[line] = False
            actions.append({"action": f"produce_{product}"})
            line_has_output[line] = True
    return actions


def _looks_like_parallel_start(text: str) -> bool:
    return any(word in text for word in (
        "同时",
        "同步",
        "一起",
        "并行",
        "同时启动",
        "simultaneously",
        "atthesametime",
        "inparallel",
        "starttogether",
    ))


def _looks_like_production(text: str) -> bool:
    return any(word in text for word in (
        "\u751f\u4ea7",
        "\u5236\u9020",
        "\u7ec4\u88c5",
    ))


def _product_count(text: str, aliases: tuple[str, ...]) -> int:
    for alias in aliases:
        position = text.find(alias)
        if position < 0:
            continue

        after_start = position + len(alias)
        after_window = text[after_start:after_start + 4]
        after_token = _first_quantity_token(after_window)
        if after_token:
            return _quantity_value(after_token)

        before_window = text[max(0, position - 4):position]
        before_token = _last_quantity_token(before_window)
        if before_token:
            return _quantity_value(before_token)
        return 1
    return 0


def _first_quantity_token(text: str) -> str | None:
    token = ""
    for char in text:
        if _is_quantity_char(char):
            token += char
        elif token:
            return token
    return token or None


def _last_quantity_token(text: str) -> str | None:
    token = ""
    for char in reversed(text):
        if _is_quantity_char(char):
            token = char + token
        elif token:
            return token
    return token or None


def _is_quantity_char(char: str) -> bool:
    return char.isdigit() or char in {
        "\u4e00",
        "\u4e8c",
        "\u4e24",
        "\u4e09",
        "\u56db",
        "\u4e94",
        "\u516d",
        "\u4e03",
        "\u516b",
        "\u4e5d",
        "\u5341",
    }


def _quantity_value(token: str) -> int:
    if token.isdigit():
        return max(1, int(token))
    values = {
        "\u4e00": 1,
        "\u4e8c": 2,
        "\u4e24": 2,
        "\u4e09": 3,
        "\u56db": 4,
        "\u4e94": 5,
        "\u516d": 6,
        "\u4e03": 7,
        "\u516b": 8,
        "\u4e5d": 9,
        "\u5341": 10,
    }
    if token == "\u5341":
        return 10
    if token.startswith("\u5341"):
        return 10 + values.get(token[1:], 0)
    if "\u5341" in token:
        left, right = token.split("\u5341", 1)
        return values.get(left, 1) * 10 + values.get(right, 0)
    return values.get(token, 1)


def _first_product_index(text: str, aliases: tuple[str, ...]) -> int:
    indexes = [text.find(alias) for alias in aliases if alias in text]
    return min(indexes) if indexes else len(text)


def _looks_like_move_to_output(text: str) -> bool:
    has_move = any(word in text for word in (
        "move",
        "transport",
        "send",
        "放到",
        "放入",
        "搬到",
        "送到",
        "送至",
        "移动",
        "移到",
        "移至",
        "运到",
        "鎶",
        "绉",
        "閫",
        "杩",
    ))
    has_output = any(word in text for word in (
        "output",
        "输出",
        "出料",
        "成品",
        "杈撳嚭",
        "鍑烘枡",
        "鎴愬搧",
    ))
    has_part_context = any(word in text for word in (
        "part",
        "零件",
        "部件",
        "底盘",
        "车架",
        "屏幕",
        "摄像",
        "机身",
        "闆",
        "浠",
        "灞",
        "鏋",
    ))
    return has_move and has_output and has_part_context


def _detect_line(text: str) -> str | None:
    if any(word in text for word in (
        "phone_line",
        "手机产线",
        "手机线",
        "b线",
        "b产线",
        "鎵嬫満",
        "墜鏈",
        "b绾",
    )):
        return "B"
    if any(word in text for word in (
        "car_line",
        "汽车产线",
        "车辆产线",
        "车产线",
        "a线",
        "a产线",
        "姹借溅",
        "杞",
        "a绾",
    )):
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


def _move_part_to_output_process(
    line: str,
    part: str,
    assumptions: list[str] | None = None,
) -> dict[str, Any]:
    if part not in SUPPORTED_PARTS:
        raise PlanValidationError(f"Unsupported part: {part}")

    plan: dict[str, Any] = {
        "plan_name": f"move_{line.lower()}_{part}_to_output",
        "strategy": "sequential",
        "actions": [{"action": "move_to_output", "line": line, "part": part}],
    }
    if assumptions:
        plan["assumptions"] = assumptions
    return plan
