"""Stress-test planner architectures under complex production instructions.

The optimized path uses the project architecture:
Top Planner -> Operation Agent -> process compiler -> validated tool plan.

The direct baseline asks the same base model to emit the full low-level tool
plan in one JSON response, then validates it with the same validator used by
the runtime.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config import (
    API_KEY,
    BASE_MODEL,
    BASE_URL,
    LLM_KEEP_ALIVE,
    LLM_TIMEOUT,
)
from llm_client import LLMError, OpenAICompatibleClient
from planner import (
    OPERATION_SUBAGENT_PROMPT,
    SYSTEM_PROMPT,
    ProductionPlanner,
)
from process_compiler import compile_process_plan
from rules import parse_top_plan, top_plan_to_operation_plan
from tool_registry import build_tool_prompt
from validator import PlanValidationError, validate_plan, validate_plan_sequence


DEFAULT_PROMPTS = [
    "连续生产两部手机和一辆汽车",
    "按顺序生产三部手机和两辆汽车",
    "先同时启动生产一部手机和一辆汽车，完成后再生产一部手机和一辆汽车",
    "连续生产两部手机、两辆汽车，并把B线的屏幕移动到output",
]


DIRECT_TOOL_PROMPT = f"""
You are a direct low-level factory tool planner.

Return JSON only, with exactly this shape:
{{"plan_name": "short_snake_case_name", "steps": [{{"tool": "ToolName", "args": {{}}}}]}}

Do not output high-level actions, operations, loops, repeated-count notation,
markdown, comments, or explanations. Expand every requested product into every
low-level tool step.

Factory material layout:
- Source area A contains car_base, phone_base, phone_base from left to right.
- Source area B contains car_frame, screen, camera_module from left to right.
- Load_A_Pick can load only car_base or phone_base.
- Load_B_Pick can load car_frame, screen, or camera_module.
- Load_B2_Pick can load screen or camera_module.

Product requirements:
- A car is assembled on line A from car_base and car_frame.
- A phone is assembled on line B from phone_base, screen, and camera_module.
- phone_base must come from source line A and be transferred to line B.
- car_frame must come from source line B and be transferred to line A/A2.
- After every finished product reaches output, include the required reset
  transport before the next product uses that same product line.
- When a Chinese instruction gives quantities, expand each item explicitly.

Use only executable tools from the list below. The returned JSON must pass
stateful station and holding validation.

{build_tool_prompt()}
""".strip()


@dataclass
class LLMCallRecord:
    index: int
    role: str
    max_tokens: int
    elapsed_s: float
    ok: bool
    system_chars: int
    user_chars: int
    output_chars: int = 0
    error: str = ""


@dataclass
class CaseResult:
    prompt: str
    architecture: str
    ok: bool
    elapsed_s: float
    llm_calls: int
    llm_elapsed_s: float
    tool_steps: int = 0
    top_actions: int = 0
    operations: int = 0
    unique_actions: int = 0
    plan_chars: int = 0
    planning_source: str = ""
    error: str = ""
    calls: list[LLMCallRecord] = field(default_factory=list)


class CountingLLMClient:
    """Small wrapper that records timing and payload size for each LLM call."""

    def __init__(self, inner: OpenAICompatibleClient):
        self.inner = inner
        self.calls: list[LLMCallRecord] = []

    @property
    def is_configured(self) -> bool:
        return self.inner.is_configured

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        role = self._role_for_prompt(system_prompt)
        started = time.perf_counter()
        try:
            payload = self.inner.chat_json(system_prompt, user_prompt, max_tokens=max_tokens)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self.calls.append(
                LLMCallRecord(
                    index=len(self.calls) + 1,
                    role=role,
                    max_tokens=max_tokens,
                    elapsed_s=elapsed,
                    ok=False,
                    system_chars=len(system_prompt),
                    user_chars=len(user_prompt),
                    error=str(exc),
                )
            )
            raise

        elapsed = time.perf_counter() - started
        self.calls.append(
            LLMCallRecord(
                index=len(self.calls) + 1,
                role=role,
                max_tokens=max_tokens,
                elapsed_s=elapsed,
                ok=True,
                system_chars=len(system_prompt),
                user_chars=len(user_prompt),
                output_chars=len(json.dumps(payload, ensure_ascii=False)),
            )
        )
        return payload

    def _role_for_prompt(self, system_prompt: str) -> str:
        if system_prompt == SYSTEM_PROMPT:
            return "top_planner"
        if system_prompt == OPERATION_SUBAGENT_PROMPT:
            return "operation_agent"
        if system_prompt == DIRECT_TOOL_PROMPT:
            return "direct_tool_planner"
        return "unknown"


def build_llm_client(model: str, timeout: float) -> CountingLLMClient:
    return CountingLLMClient(
        OpenAICompatibleClient(
            model,
            BASE_URL,
            API_KEY,
            timeout=timeout,
            keep_alive=LLM_KEEP_ALIVE,
        )
    )


def run_optimized(prompt: str, model: str, timeout: float) -> CaseResult:
    client = build_llm_client(model, timeout)
    planner = ProductionPlanner(llm_client=client)
    started = time.perf_counter()
    try:
        plan = planner.run(prompt)
        elapsed = time.perf_counter() - started
        top_plan = planner.last_top_level_plan or {}
        subagent_plan = planner.last_subagent_plan or {}
        actions = top_plan.get("actions", [])
        operations = subagent_plan.get("operations", [])
        return CaseResult(
            prompt=prompt,
            architecture="two_layer_agent",
            ok=True,
            elapsed_s=elapsed,
            llm_calls=len(client.calls),
            llm_elapsed_s=sum(call.elapsed_s for call in client.calls),
            tool_steps=len(plan["steps"]),
            top_actions=len(actions),
            operations=len(operations),
            unique_actions=count_unique_actions(actions),
            plan_chars=len(json.dumps(plan, ensure_ascii=False)),
            planning_source=str(plan.get("planning_source", "")),
            calls=client.calls,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return CaseResult(
            prompt=prompt,
            architecture="two_layer_agent",
            ok=False,
            elapsed_s=elapsed,
            llm_calls=len(client.calls),
            llm_elapsed_s=sum(call.elapsed_s for call in client.calls),
            error=str(exc),
            calls=client.calls,
        )


def run_direct(prompt: str, model: str, timeout: float, max_tokens: int) -> CaseResult:
    client = build_llm_client(model, timeout)
    started = time.perf_counter()
    try:
        payload = client.chat_json(DIRECT_TOOL_PROMPT, prompt, max_tokens=max_tokens)
        plan = validate_plan(payload)
        validate_plan_sequence(plan)
        elapsed = time.perf_counter() - started
        return CaseResult(
            prompt=prompt,
            architecture="direct_tool_json",
            ok=True,
            elapsed_s=elapsed,
            llm_calls=len(client.calls),
            llm_elapsed_s=sum(call.elapsed_s for call in client.calls),
            tool_steps=len(plan["steps"]),
            plan_chars=len(json.dumps(plan, ensure_ascii=False)),
            calls=client.calls,
        )
    except (LLMError, PlanValidationError, Exception) as exc:
        elapsed = time.perf_counter() - started
        output_chars = sum(call.output_chars for call in client.calls)
        return CaseResult(
            prompt=prompt,
            architecture="direct_tool_json",
            ok=False,
            elapsed_s=elapsed,
            llm_calls=len(client.calls),
            llm_elapsed_s=sum(call.elapsed_s for call in client.calls),
            plan_chars=output_chars,
            error=str(exc),
            calls=client.calls,
        )


def run_compiled_oracle(prompt: str) -> dict[str, Any]:
    """Deterministic compiler output used as the direct baseline size target."""

    top_plan = parse_top_plan(prompt)
    operation_plan = top_plan_to_operation_plan(top_plan)
    tool_plan = compile_process_plan(operation_plan)
    return {
        "top_actions": len(top_plan.get("actions", [])),
        "unique_actions": count_unique_actions(top_plan.get("actions", [])),
        "operations": len(operation_plan.get("operations", [])),
        "tool_steps": len(tool_plan.get("steps", [])),
        "top_json_chars": len(json.dumps(top_plan, ensure_ascii=False)),
        "operation_json_chars": len(json.dumps(operation_plan, ensure_ascii=False)),
        "tool_json_chars": len(json.dumps(tool_plan, ensure_ascii=False)),
    }


def count_unique_actions(actions: list[dict[str, Any]]) -> int:
    return len({
        json.dumps(action, ensure_ascii=False, sort_keys=True)
        for action in actions
    })


def summarize_results(results: list[CaseResult], oracle: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Planning Stress Test Results",
        "",
    ]
    if results:
        lines.extend([
            "| prompt | architecture | ok | elapsed_s | llm_calls | llm_elapsed_s | top_actions | unique_actions | operations | tool_steps | plan_chars | source/error |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for result in results:
            source = result.planning_source or result.error.replace("\n", " ")[:120]
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_cell(result.prompt),
                        result.architecture,
                        "yes" if result.ok else "no",
                        f"{result.elapsed_s:.2f}",
                        str(result.llm_calls),
                        f"{result.llm_elapsed_s:.2f}",
                        str(result.top_actions),
                        str(result.unique_actions),
                        str(result.operations),
                        str(result.tool_steps),
                        str(result.plan_chars),
                        md_cell(source),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No LLM timing run was requested; this file contains deterministic complexity targets only.")

    lines.extend([
        "",
        "## Deterministic Complexity Targets",
        "",
        "| prompt | top_actions | unique_actions | operations | compiled_tool_steps | top_json_chars | operation_json_chars | direct_tool_json_chars |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for prompt, item in oracle.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    md_cell(prompt),
                    str(item["top_actions"]),
                    str(item["unique_actions"]),
                    str(item["operations"]),
                    str(item["tool_steps"]),
                    str(item["top_json_chars"]),
                    str(item["operation_json_chars"]),
                    str(item["tool_json_chars"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def md_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_outputs(
    output_dir: Path,
    model: str,
    mode: str,
    results: list[CaseResult],
    oracle: dict[str, dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "mode": mode,
        "results": [
            {
                **asdict(result),
                "calls": [asdict(call) for call in result.calls],
            }
            for result in results
        ],
        "oracle": oracle,
    }
    (output_dir / "planning_stress_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "planning_stress_results.md").write_text(
        summarize_results(results, oracle),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two-layer agent planning with direct low-level JSON planning.",
    )
    parser.add_argument("--model", default=BASE_MODEL, help="Ollama model name.")
    parser.add_argument("--timeout", type=float, default=LLM_TIMEOUT)
    parser.add_argument(
        "--mode",
        choices=("optimized", "direct", "both", "oracle"),
        default="both",
    )
    parser.add_argument("--direct-max-tokens", type=int, default=8192)
    parser.add_argument("--max-prompts", type=int, default=len(DEFAULT_PROMPTS))
    parser.add_argument(
        "--output-dir",
        default="stress_results",
        help="Directory for JSON and Markdown result files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts = DEFAULT_PROMPTS[: max(1, args.max_prompts)]
    results: list[CaseResult] = []
    oracle = {prompt: run_compiled_oracle(prompt) for prompt in prompts}

    for prompt in prompts:
        if args.mode in {"optimized", "both"}:
            results.append(run_optimized(prompt, args.model, args.timeout))
        if args.mode in {"direct", "both"}:
            results.append(
                run_direct(
                    prompt,
                    args.model,
                    args.timeout,
                    args.direct_max_tokens,
                )
            )

    output_dir = Path(args.output_dir)
    write_outputs(output_dir, args.model, args.mode, results, oracle)
    print(summarize_results(results, oracle))
    print(f"Saved results to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
