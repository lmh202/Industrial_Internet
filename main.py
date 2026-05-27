"""Command-line entry point for the intelligent manufacturing agent."""

from __future__ import annotations

import argparse
import subprocess
import sys
import shutil
import time
from pathlib import Path

from agent import ProductionAgent, tasks_to_json
from config import (
    COPPELIASIM_EXE,
    SCENE_PATH,
    SIM_CONNECT_HOST,
    SIM_CONNECT_PORT,
    SIM_CONNECT_TIMEOUT,
    SIM_DT,
)
from factory_controller import FactoryController


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="智能制造 AI Agent: natural language to CoppeliaSim control")
    parser.add_argument("prompt", nargs="+", help="生产任务，如：生产一辆车")
    parser.add_argument("--no-launch", action="store_true",
                        help="不自动启动 CoppeliaSim，只连接已运行实例")
    parser.add_argument("--parse-only", action="store_true",
                        help="只解析任务，不连接仿真")
    args = parser.parse_args(argv)

    prompt = " ".join(args.prompt)
    agent = ProductionAgent()
    tasks = agent.run(prompt)
    print(f"[Agent] 解析结果: {tasks_to_json(tasks)}")

    if args.parse_only:
        return 0

    proc = None
    sim = None
    try:
        if not args.no_launch:
            proc = launch_coppeliasim()
            time.sleep(3)
        sim = connect_sim()
        configure_scene(sim)
        factory = FactoryController(sim)
        factory.produce_plan(tasks)
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
