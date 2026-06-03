"""Natural-language production task parser."""

from __future__ import annotations

import json
import re
from typing import Any

from config import API_KEY, BASE_MODEL, BASE_URL, LLM_TIMEOUT
from llm_client import LLMError, OpenAICompatibleClient


SUPPORTED_PRODUCTS = {"car", "phone"}

SYSTEM_PROMPT = """
You are a manufacturing task parser. Convert the user's Chinese or English
instruction into compact JSON only.

Return exactly one JSON object with this shape:
{"tasks":[{"product":"car","quantity":1}]}

Rules:
- The only valid product strings are "car" and "phone".
- Never output "car|phone" as a product value.
- Use "car" for 车, 车辆, 汽车, vehicle, car, cars.
- Use "phone" for 手机, phone, phones.
- quantity must be a positive integer.
- preserve the order mentioned by the user.
- do not include markdown or explanations.

Examples:
User: 生产一辆车
Output: {"tasks":[{"product":"car","quantity":1}]}
User: 连续生产一辆车和两部手机
Output: {"tasks":[{"product":"car","quantity":1},{"product":"phone","quantity":2}]}
User: produce 2 phones and 1 car
Output: {"tasks":[{"product":"phone","quantity":2},{"product":"car","quantity":1}]}
"""


class TaskParseError(ValueError):
    """Raised when a prompt cannot be converted into production tasks."""


class ProductionAgent:
    def __init__(self, llm_client: OpenAICompatibleClient | None = None,
                 allow_rule_fallback: bool = True):
        self.llm_client = llm_client or OpenAICompatibleClient(
            BASE_MODEL, BASE_URL, API_KEY, timeout=LLM_TIMEOUT)
        self.allow_rule_fallback = allow_rule_fallback

    def run(self, prompt: str) -> list[dict[str, int | str]]:
        """Parse user text into validated production tasks."""
        errors = []
        rule_tasks = None
        if self.allow_rule_fallback:
            try:
                rule_tasks = self._parse_with_rules(prompt)
            except TaskParseError as exc:
                errors.append(str(exc))

        if self.llm_client.is_configured:
            try:
                llm_tasks = self._validate_payload(
                    self.llm_client.chat_json(SYSTEM_PROMPT, prompt))
                if self._same_product_order(rule_tasks, llm_tasks):
                    return rule_tasks
                return llm_tasks
            except (LLMError, TaskParseError) as exc:
                errors.append(str(exc))

        if rule_tasks is not None:
            return rule_tasks

        detail = "; ".join(errors) if errors else "no parser available"
        raise TaskParseError(f"Cannot parse production task: {detail}")

    def _validate_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
            raise TaskParseError('Expected {"tasks": [...]} JSON.')

        tasks = []
        for item in payload["tasks"]:
            if not isinstance(item, dict):
                raise TaskParseError("Each task must be an object.")
            product = str(item.get("product", "")).strip().lower()
            quantity = item.get("quantity", 1)
            if product not in SUPPORTED_PRODUCTS:
                raise TaskParseError(f"Unsupported product: {product!r}")
            if isinstance(quantity, bool):
                raise TaskParseError("Quantity must be an integer.")
            try:
                quantity = int(quantity)
            except (TypeError, ValueError) as exc:
                raise TaskParseError("Quantity must be an integer.") from exc
            if quantity <= 0:
                raise TaskParseError("Quantity must be positive.")
            tasks.append({"product": product, "quantity": quantity})

        if not tasks:
            raise TaskParseError("No production tasks found.")
        return tasks

    def _parse_with_rules(self, prompt: str) -> list[dict[str, int | str]]:
        text = prompt.strip().lower()
        if not text:
            raise TaskParseError("Prompt is empty.")

        matches = []
        product_pattern = (
            r"(车辆|汽车|车|cars|car|手机|phones|phone)"
        )
        product_matches = list(re.finditer(product_pattern, text))
        previous_end = 0
        previous_suffix_consumed = False
        for index, match in enumerate(product_matches):
            product = self._normalize_product(match.group(1))
            next_start = (
                product_matches[index + 1].start()
                if index + 1 < len(product_matches)
                else len(text)
            )
            quantity, suffix_consumed = self._extract_quantity(
                text[previous_end:match.start()],
                text[match.end():next_start],
                prefix_consumed=previous_suffix_consumed,
            )
            matches.append({"product": product, "quantity": quantity})
            previous_end = match.end()
            previous_suffix_consumed = suffix_consumed

        if not matches:
            raise TaskParseError("No supported product keyword found.")
        return self._merge_adjacent(matches)

    def _normalize_product(self, word: str) -> str:
        if word in {"车", "车辆", "汽车", "car", "cars"}:
            return "car"
        return "phone"

    def _extract_quantity(self, prefix: str, suffix: str = "",
                          prefix_consumed: bool = False) -> tuple[int, bool]:
        if not prefix_consumed:
            before = self._quantity_candidates(prefix)
            if before:
                return sorted(before, key=lambda item: item[0])[-1][1], False

        after = self._quantity_candidates(suffix)
        if after and self._starts_with_quantity(suffix, after):
            return sorted(after, key=lambda item: item[0])[0][1], True
        return 1, False

    def _starts_with_quantity(self, text: str, candidates: list[tuple[int, int]]) -> bool:
        if not candidates:
            return False
        stripped = text.lstrip()
        leading_spaces = len(text) - len(stripped)
        first_pos = sorted(candidates, key=lambda item: item[0])[0][0]
        return first_pos == leading_spaces

    def _quantity_candidates(self, text: str) -> list[tuple[int, int]]:
        chinese_numbers = {
            "一": 1,
            "二": 2, "两": 2, "俩": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }
        candidates: list[tuple[int, int]] = []
        for match in re.finditer(r"\d+", text):
            candidates.append((match.start(), int(match.group(0))))
        for word, value in chinese_numbers.items():
            for match in re.finditer(word, text):
                candidates.append((match.start(), value))
        return candidates

    def _merge_adjacent(self, tasks: list[dict[str, int | str]]):
        merged = []
        for task in tasks:
            if merged and merged[-1]["product"] == task["product"]:
                merged[-1]["quantity"] = int(merged[-1]["quantity"]) + int(task["quantity"])
            else:
                merged.append(dict(task))
        return merged

    def _same_product_order(self, first, second) -> bool:
        if first is None or second is None or len(first) != len(second):
            return False
        return all(
            a["product"] == b["product"]
            for a, b in zip(first, second)
        )


def tasks_to_json(tasks: list[dict[str, int | str]]) -> str:
    return json.dumps({"tasks": tasks}, ensure_ascii=False)
