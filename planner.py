"""LLM-backed production planner with deterministic guardrails."""

from __future__ import annotations

from typing import Any

from config import API_KEY, BASE_MODEL, BASE_URL, LLM_KEEP_ALIVE, LLM_TIMEOUT
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
