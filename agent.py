"""Natural-language production task parser."""

from __future__ import annotations

import json
import re
from typing import Any

from config import API_KEY, BASE_MODEL, BASE_URL
from llm_client import LLMError, OpenAICompatibleClient


SUPPORTED_PRODUCTS = {"car", "phone"}

SYSTEM_PROMPT = """
You are a manufacturing task parser. Convert the user's Chinese or English
instruction into compact JSON only.

Return schema:
{"tasks":[{"product":"car|phone","quantity":1}]}

Rules:
- product must be "car" for vehicle/car tasks or "phone" for phone tasks.
- quantity must be a positive integer.
- preserve the order mentioned by the user.
- do not include markdown or explanations.
"""


class TaskParseError(ValueError):
    """Raised when a prompt cannot be converted into production tasks."""


class ProductionAgent:
    def __init__(self, llm_client: OpenAICompatibleClient | None = None,
                 allow_rule_fallback: bool = True):
        self.llm_client = llm_client or OpenAICompatibleClient(
            BASE_MODEL, BASE_URL, API_KEY)
        self.allow_rule_fallback = allow_rule_fallback

    def run(self, prompt: str) -> list[dict[str, int | str]]:
        """Parse user text into validated production tasks."""
        errors = []
        if self.llm_client.is_configured:
            try:
                return self._validate_payload(
                    self.llm_client.chat_json(SYSTEM_PROMPT, prompt))
            except (LLMError, TaskParseError) as exc:
                errors.append(str(exc))

        if self.allow_rule_fallback:
            try:
                return self._parse_with_rules(prompt)
            except TaskParseError as exc:
                errors.append(str(exc))

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
        for match in re.finditer(product_pattern, text):
            product = self._normalize_product(match.group(1))
            start = max(0, match.start() - 12)
            end = min(len(text), match.end() + 4)
            quantity = self._extract_quantity(
                text[start:match.start()],
                text[match.end():end],
            )
            matches.append({"product": product, "quantity": quantity})

        if not matches:
            raise TaskParseError("No supported product keyword found.")
        return self._merge_adjacent(matches)

    def _normalize_product(self, word: str) -> str:
        if word in {"车", "车辆", "汽车", "car", "cars"}:
            return "car"
        return "phone"

    def _extract_quantity(self, prefix: str, suffix: str = "") -> int:
        before = self._quantity_candidates(prefix)
        if before:
            return sorted(before, key=lambda item: item[0])[-1][1]
        after = self._quantity_candidates(suffix)
        if after:
            return sorted(after, key=lambda item: item[0])[0][1]
        return 1

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


def tasks_to_json(tasks: list[dict[str, int | str]]) -> str:
    return json.dumps({"tasks": tasks}, ensure_ascii=False)
