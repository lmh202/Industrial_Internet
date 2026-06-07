"""LLM-backed production planner with deterministic guardrails."""

from __future__ import annotations

import json
from typing import Any

from config import (
    AGENT_RULE_FALLBACK,
    AGENT_RULE_HINTS,
    API_KEY,
    BASE_MODEL,
    BASE_URL,
    LLM_KEEP_ALIVE,
    LLM_TIMEOUT,
)
from llm_client import LLMError, OpenAICompatibleClient
from rules import rule_plan_from_prompt
from tool_registry import build_tool_prompt
from validator import PlanValidationError, validate_plan, validate_plan_sequence


SYSTEM_PROMPT = f"""
Return one JSON object only: {{"plan_name":"...","steps":[{{"tool":"...","args":{{}}}}]}}.
No markdown, no prose, no code, no loops, no make_car/make_phone.
Never output coordinates, robot names, handles, or non-tool actions.
Expand quantities into repeated concrete steps.
Parts only: car_base, car_frame, phone_base, screen, camera_module.

{build_tool_prompt()}

Screen loading rule: screen must always be loaded with Load_B2_Pick after
Transport_B2_Clear_Pick. Never use Load_B_Pick(screen).
Auxiliary shuttle rule: do not use the assemble arm as a long-term buffer.
Forward means moving the main shuttle a short distance forward along the
conveyor, not sideways toward the robot base. The auxiliary shuttle then
follows behind to the normal assemble station, so the assemble arm picks from
the original assemble position.
For car assembly, put car_base on A first, move A forward along the conveyor,
use A2 to bring car_frame to the assemble station, hold car_frame briefly,
return A2 to clear, move A back to assemble, then place car_frame on car_base.
For phone assembly, put phone_base on B first. Use B2 to bring screen, place it
on phone_base, then move B forward along the conveyor. Use B2 to bring
camera_module to the assemble station, hold it briefly, return B2 to clear, move B back to
assemble, then place camera_module.
Every repeated product must begin with its main shuttle at pick. If another
product on the same line follows an Unload_*_Output step, insert exactly one
Transport_A_Output_Pick or Transport_B_Output_Pick before the next Load_*_Pick.
If the user asks to start car and phone production simultaneously, interleave
independent A-line and B-line steps so both lines begin early; keep each line's
own internal step order unchanged.
For two phones, the first Unload_B_Output(phone_base) must be followed by
Transport_B_Output_Pick before starting the second phone with Load_B_Pick.
For two cars, the first Unload_A_Output(car_base) must be followed by
Transport_A_Output_Pick before starting the second car with Load_A_Pick.

Car plan on A:
Load_A_Pick(car_base), Transport_A_Pick_Assemble,
Transport_A_Assemble_Forward,
Transport_A2_Clear_Pick, Load_A2_Pick(car_frame),
Transport_A2_Pick_Assemble, Hold_A2_Assemble(car_frame),
Transport_A2_Assemble_Clear, Transport_A_Forward_Assemble,
Place_A_Assemble(car_frame, attach_to=car_base, layer=1),
Transport_A_Assemble_Camera, Inspect_A, Transport_A_Camera_Output,
Unload_A_Output(car_base).

Phone plan on B:
Load_B_Pick(phone_base), Transport_B_Pick_Assemble,
Transport_B_Assemble_Forward,
Transport_B2_Clear_Pick, Load_B2_Pick(screen), Transport_B2_Pick_Assemble,
Hold_B2_Assemble(screen), Transport_B2_Assemble_Clear,
Transport_B_Forward_Assemble,
Place_B_Assemble(screen, attach_to=phone_base, layer=1),
Transport_B_Assemble_Forward,
Transport_B2_Clear_Pick, Load_B2_Pick(camera_module),
Transport_B2_Pick_Assemble, Hold_B2_Assemble(camera_module),
Transport_B2_Assemble_Clear, Transport_B_Forward_Assemble,
Place_B_Assemble(camera_module, attach_to=phone_base, layer=2),
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
""".strip()


APPROVAL_SYSTEM = """
Return one JSON object only: {"approved": true|false, "reason": "..."}.
You are approving or rejecting a candidate factory tool-call plan.
Approve only if the candidate correctly satisfies the user request, uses only
allowed tools, and keeps all tool arguments explicit. Do not return a tool plan.
If the user request is ambiguous and the candidate includes an explicit
assumption or default, approve when that assumption is reasonable and executable.
""".strip()


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

        if not self.llm_client.is_configured:
            return self._rule_plan_or_raise(prompt, "LLM is not configured.")

        rule_plan = rule_plan_from_prompt(prompt) if AGENT_RULE_HINTS else None
        if rule_plan is not None:
            approved = self._approve_candidate_plan(prompt, rule_plan)
            if approved is not None:
                plan = validate_plan(rule_plan)
                validate_plan_sequence(plan)
                plan["planning_source"] = "llm_approved_candidate"
                plan["planning_approval"] = approved
                return plan

        planning_prompt = self._planning_prompt(prompt, rule_plan)
        last_error: PlanValidationError | None = None
        for attempt in range(3):
            try:
                payload = self.llm_client.chat_json(SYSTEM_PROMPT, planning_prompt)
            except LLMError as exc:
                raise PlanValidationError(f"Cannot create tool plan: {exc}") from exc
            try:
                plan = validate_plan(payload)
                validate_plan_sequence(plan)
                plan["planning_source"] = "llm"
                plan["planning_attempts"] = attempt + 1
                if rule_plan is not None:
                    plan["planning_hint"] = "rule_candidate"
                return plan
            except PlanValidationError as exc:
                last_error = exc
                if attempt == 2:
                    break
                planning_prompt = (
                    f"{self._planning_prompt(prompt, rule_plan)}\n\n"
                    f"Previous JSON plan was invalid: {exc}. "
                    "Return a corrected JSON object only. Do not explain."
                )
        if AGENT_RULE_FALLBACK:
            return self._rule_plan_or_raise(
                prompt,
                f"LLM planning failed after validation retries: {last_error}",
            )
        raise last_error or PlanValidationError("Cannot create a valid tool plan.")

    def _rule_plan_or_raise(self, prompt: str, reason: str) -> dict[str, Any]:
        rule_plan = rule_plan_from_prompt(prompt)
        if rule_plan is None:
            raise PlanValidationError(reason)
        plan = validate_plan(rule_plan)
        validate_plan_sequence(plan)
        plan["planning_source"] = "rule_fallback"
        assumptions = list(plan.get("assumptions", []))
        assumptions.append(reason)
        plan["assumptions"] = assumptions
        return plan

    def _planning_prompt(
        self,
        prompt: str,
        rule_plan: dict[str, Any] | None,
    ) -> str:
        if rule_plan is None:
            return prompt
        return (
            f"User request: {prompt}\n\n"
            "A deterministic tool-template candidate is provided as context only. "
            "You must act as the planner: review it against the system tool rules, "
            "fix it if needed, and return the final executable JSON plan yourself. "
            "Do not copy any invalid step.\n"
            f"Candidate JSON: {json.dumps(rule_plan, ensure_ascii=False)}"
        )

    def _approve_candidate_plan(
        self,
        prompt: str,
        rule_plan: dict[str, Any],
    ) -> dict[str, Any] | None:
        user_prompt = (
            f"User request: {prompt}\n"
            f"Candidate JSON: {json.dumps(rule_plan, ensure_ascii=False)}"
        )
        try:
            payload = self.llm_client.chat_json(APPROVAL_SYSTEM, user_prompt)
        except LLMError:
            return None
        if not isinstance(payload, dict):
            return None
        approved = payload.get("approved")
        if not isinstance(approved, bool):
            return None
        if not approved:
            return None
        reason = payload.get("reason", "")
        if not isinstance(reason, str):
            reason = ""
        return {"approved": True, "reason": reason.strip()}


ProductionAgent = ProductionPlanner
