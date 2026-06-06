"""Command-line entry point for the intelligent manufacturing agent."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import shutil
import time
from pathlib import Path

from agent import ProductionAgent, plan_to_json
from config import (
    API_KEY,
    BASE_MODEL,
    BASE_URL,
    COPPELIASIM_EXE,
    LLM_KEEP_ALIVE,
    LLM_TIMEOUT,
    SCENE_PATH,
    SIM_CONNECT_HOST,
    SIM_CONNECT_PORT,
    SIM_CONNECT_TIMEOUT,
    SIM_DT,
)
from factory_controller import FactoryController
from llm_client import LLMError, OpenAICompatibleClient


PLAN_OUTPUT_PATH = Path(__file__).resolve().parent / "plan.json"


def launch_coppeliasim() -> subprocess.Popen | None:
    exe = Path(COPPELIASIM_EXE)
    if not exe.exists():
        raise FileNotFoundError(f"CoppeliaSim executable not found: {exe}")
    print(f"[Sim] 启动 CoppeliaSim: {exe}")
    return subprocess.Popen([str(exe)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def connect_sim():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    deadline = time.time() + SIM_CONNECT_TIMEOUT
    last_error = None
    while time.time() < deadline:
        try:
            client = RemoteAPIClient(host=SIM_CONNECT_HOST, port=SIM_CONNECT_PORT)
            sim = client.require("sim")
            sim.getSimulationTime()
            return sim
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Cannot connect to CoppeliaSim Remote API: {last_error}")


def configure_scene(sim):
    scene_path = _prepare_scene_path(Path(SCENE_PATH).resolve())
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")
    print(f"[Sim] 加载场景: {scene_path}")
    _stop_simulation_if_needed(sim)
    sim.loadScene(scene_path.as_posix())
    sim.setBoolParam(sim.boolparam_dynamics_handling_enabled, False)
    sim.setStepping(True)
    sim.setFloatParam(sim.floatparam_simulation_time_step, SIM_DT)
    sim.startSimulation()


def _prepare_scene_path(scene_path: Path) -> Path:
    """CoppeliaSim on Windows can fail on non-ASCII scene paths."""
    try:
        str(scene_path).encode("ascii")
        return scene_path
    except UnicodeEncodeError:
        target = Path(r"C:\tmp") / scene_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(scene_path, target)
        print(f"[Sim] 场景路径包含非 ASCII 字符，已复制到: {target}")
        return target


def _stop_simulation_if_needed(sim):
    stopped_state = getattr(sim, "simulation_stopped", 0)
    try:
        state = sim.getSimulationState()
    except Exception:
        return
    if state == stopped_state:
        return
    print("[Sim] 当前仿真未停止，先停止后重新加载场景")
    sim.stopSimulation()
    _wait_for_simulation_stopped(sim, timeout=10)


def _wait_for_simulation_stopped(sim, timeout: float = 10):
    stopped_state = getattr(sim, "simulation_stopped", 0)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if sim.getSimulationState() == stopped_state:
                return
        except Exception:
            return
        time.sleep(0.2)


def _model_help_status() -> str:
    client = OpenAICompatibleClient(
        BASE_MODEL, BASE_URL, API_KEY, timeout=min(LLM_TIMEOUT, 2.0))
    try:
        loaded_models = client.loaded_models(timeout=1.0)
    except LLMError as exc:
        return (
            "模型状态:\n"
            f"  BASE_MODEL: {BASE_MODEL}\n"
            f"  BASE_URL: {BASE_URL}\n"
            f"  LLM_KEEP_ALIVE: {LLM_KEEP_ALIVE}\n"
            f"  当前状态: 无法查询 Ollama ({exc})"
        )

    loaded_names = []
    is_loaded = False
    for model in loaded_models:
        if not isinstance(model, dict):
            continue
        name = str(model.get("name") or model.get("model") or "").strip()
        if name:
            loaded_names.append(name)
        if BASE_MODEL in {str(model.get("name", "")), str(model.get("model", ""))}:
            is_loaded = True

    loaded_text = ", ".join(loaded_names) if loaded_names else "无"
    return (
        "模型状态:\n"
        f"  BASE_MODEL: {BASE_MODEL}\n"
        f"  BASE_URL: {BASE_URL}\n"
        f"  LLM_KEEP_ALIVE: {LLM_KEEP_ALIVE}\n"
        f"  当前状态: {'已加载' if is_loaded else '未加载'}\n"
        f"  Ollama 已加载模型: {loaded_text}"
    )


def _run_prompt_once(sim, agent: ProductionAgent, prompt: str) -> None:
    plan = agent.run(prompt)
    _save_plan(plan)
    print(f"[Planner] 计划书: {plan_to_json(plan)}")
    configure_scene(sim)
    factory = FactoryController(sim)
    report = factory.execute_tool_plan(plan)
    completed = sum(1 for step in report["steps"] if step["ok"])
    print(f"[Executor] completed {completed}/{len(report['steps'])} steps")
    if not report["ok"]:
        failed = next((step for step in report["steps"] if not step["ok"]), None)
        message = failed["message"] if failed else "unknown execution failure"
        raise RuntimeError(message)


def _run_demo_mid_transfer_once(sim) -> None:
    configure_scene(sim)
    factory = FactoryController(sim)
    factory.demo_mid_transfer()


def _save_plan(plan: dict) -> None:
    PLAN_OUTPUT_PATH.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[Planner] 计划已保存: {PLAN_OUTPUT_PATH}")


def _interactive_loop(sim, agent: ProductionAgent) -> None:
    print("[Agent] 会话模式已启动。输入新的生产指令会先复位 CoppeliaSim 场景再执行。")
    print("[Agent] 输入 exit / quit / q 退出；输入 demo-mid-transfer 演示跨产线转运。")
    while True:
        try:
            prompt = input("[Agent] 下一条指令> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", "q"}:
            return

        try:
            if prompt.lower() in {"demo-mid-transfer", "demo"}:
                _run_demo_mid_transfer_once(sim)
            else:
                _run_prompt_once(sim, agent, prompt)
        except Exception as exc:
            print(f"[ERROR] 当前指令执行失败: {exc}")
            print("[Agent] 可以继续输入下一条指令，下一次运行会重新加载场景。")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="智能制造 AI Agent: natural language to CoppeliaSim control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_model_help_status())
    parser.add_argument("prompt", nargs="*", help="生产任务，如：生产一辆车")
    parser.add_argument("--no-launch", action="store_true",
                        help="不自动启动 CoppeliaSim，只连接已运行实例")
    parser.add_argument("--parse-only", action="store_true",
                        help="只生成工具调用计划，不连接仿真")
    parser.add_argument("--demo-mid-transfer", action="store_true",
                        help="只演示 Robot_Put_Mid 跨产线转运，不执行产品生产")
    parser.add_argument("--unload-model", action="store_true",
                        help="卸载当前 Ollama 模型，不连接仿真")
    args = parser.parse_args(argv)

    if args.unload_model:
        if args.prompt:
            parser.error("--unload-model 不需要生产任务文本")
        if args.parse_only or args.demo_mid_transfer:
            parser.error("--unload-model 不能与其他运行模式同时使用")
        client = OpenAICompatibleClient(
            BASE_MODEL, BASE_URL, API_KEY, timeout=LLM_TIMEOUT)
        client.unload_model()
        print(f"[Agent] 已请求卸载模型: {BASE_MODEL}")
        return 0

    if args.demo_mid_transfer and args.prompt:
        parser.error("--demo-mid-transfer 不需要生产任务文本")
    if args.demo_mid_transfer and args.parse_only:
        parser.error("--demo-mid-transfer 不能与 --parse-only 同时使用")
    if args.parse_only and not args.prompt:
        parser.error("--parse-only 需要提供生产任务文本")

    agent = ProductionAgent()
    if args.parse_only:
        prompt = " ".join(args.prompt)
        plan = agent.run(prompt)
        _save_plan(plan)
        print(f"[Planner] 计划书: {plan_to_json(plan)}")
        return 0

    proc = None
    sim = None
    try:
        if not args.no_launch:
            proc = launch_coppeliasim()
            time.sleep(3)
        sim = connect_sim()

        if args.demo_mid_transfer:
            _run_demo_mid_transfer_once(sim)
        elif args.prompt:
            _run_prompt_once(sim, agent, " ".join(args.prompt))

        _interactive_loop(sim, agent)
        return 0
    finally:
        if sim is not None:
            try:
                sim.stopSimulation()
                _wait_for_simulation_stopped(sim, timeout=10)
            except Exception:
                pass
        if proc is not None:
            print("[Sim] CoppeliaSim 仍保持打开，便于查看结果。")


if __name__ == "__main__":
    sys.exit(main())
