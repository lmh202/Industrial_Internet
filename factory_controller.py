"""Deterministic CoppeliaSim factory workflow controller."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import time

from collision_manager import CollisionManager
from config import (
    ASSEMBLY_LAYER_Z,
    DEFAULT_INSPECT_STEPS,
    DEFAULT_SHUTTLE_SPEED,
    SHUTTLE_PART_Z,
    SIM_DT,
    SIM_RENDER_DELAY,
)
from robot_arm import RobotArmController
from scene_config import (
    LINE_A_CENTER_X,
    LINE_B_CENTER_X,
    SEGMENT_HEIGHT,
    SHUTTLE_SAFE_MARGIN,
    SHUTTLE_SIZE_X,
    SHUTTLE_SIZE_Y,
    Y_ASSEM,
    Y_CAMERA,
    Y_POLISH,
    Y_PUT,
)
from shuttle_controller import ShuttleController
from schemas import ExecutionReport, ToolResult
from tool_registry import TOOL_REGISTRY, TRANSPORT_TOOL_ROUTES


class SceneObjectError(RuntimeError):
    """Raised when required scene objects are missing."""


class ProductionStepError(RuntimeError):
    """Raised when a required pick/place/move step cannot be completed."""


STATION_Y = {
    "pick": Y_PUT,
    "assemble": Y_ASSEM,
    "camera": Y_CAMERA,
    "output": Y_POLISH,
}

B_CLEAR_OFFSET_X = SHUTTLE_SIZE_X + 2 * SHUTTLE_SAFE_MARGIN
ASSEMBLE_FOLLOW_GAP_Y = SHUTTLE_SIZE_Y + 2 * SHUTTLE_SAFE_MARGIN


class FactoryController:
    def __init__(self, sim, dt: float = SIM_DT,
                 render_delay: float = SIM_RENDER_DELAY):
        self.sim = sim
        self.dt = dt
        self.render_delay = render_delay
        self.arms: dict[str, RobotArmController] = {}
        self.shuttles: dict[str, ShuttleController] = {}
        self.shuttle_handles: dict[str, int] = {}
        self.shuttle_initial_positions: dict[str, list[float]] = {}
        self.obstacle_handles: dict[str, int] = {}
        self.part_templates: dict[str, int] = {}
        self.part_stock_positions: dict[str, list[float]] = {}
        self.output_bins: dict[str, int] = {}
        self._init_scene()

    def produce_plan(self, tasks: list[dict]):
        queues = self._build_line_queues(tasks)
        print(f"[Agent] line queues: "
              f"A={list(queues['line_a'])}, B={list(queues['line_b'])}")
        self._run_line_scheduler(queues)

    def produce_car(self, quantity: int = 1):
        self.produce_plan([{"product": "car", "quantity": quantity}])

    def produce_phone(self, quantity: int = 1):
        self.produce_plan([{"product": "phone", "quantity": quantity}])

    def demo_mid_transfer(self):
        """Demonstrate Robot_Put_Mid moving one part from line A to line B."""
        print("[Factory] start Robot_Put_Mid transfer demo")
        part = self._fresh_part("car_frame")
        self._load_part_to_shuttle(
            "line_a", self.arms["put_a"], part, self.shuttles["line_a"],
            self.shuttle_handles["line_a"], LINE_A_CENTER_X, Y_PUT,
            SHUTTLE_PART_Z)
        self.transfer_part_between_lines(part, "line_a", "line_b", y=Y_ASSEM)
        print("[Factory] Robot_Put_Mid transfer demo complete")

    def _init_scene(self):
        self.arms = {
            "put_a": self._arm("Robot_Put_A"),
            "put_b": self._arm("Robot_Put_B"),
            "put_mid": self._arm("Robot_Put_Mid"),
            "assemble_car": self._arm("Robot_Assemble_Car"),
            "assemble_phone": self._arm("Robot_Assemble_Phone"),
            "pick_car": self._arm("Robot_Pick_Car"),
            "pick_phone": self._arm("Robot_Pick_Phone"),
        }

        for name, aliases in {
            "line_a": ["Shuttle_A1", "Shuttle_A0", "Shuttle_A"],
            "line_a_aux": ["Shuttle_A2"],
            "line_b": ["Shuttle_B1", "Shuttle_B0", "Shuttle_B"],
            "line_b_aux": ["Shuttle_B2"],
        }.items():
            handle = self._object(aliases)
            self.shuttle_handles[name] = handle
            self.shuttle_initial_positions[name] = self.sim.getObjectPosition(handle, -1)
            self.shuttles[name] = ShuttleController(
                self.sim, handle, name=name, dt=self.dt,
                render_delay=self.render_delay)

        self.part_templates = {
            "car_base": self._object(["Part_Car_Base", "Part_Bottom_A0"]),
            "car_frame": self._object(["Part_Car_Frame", "Part_Top_A0"]),
            "phone_base": self._object(["Part_Phone1", "Part_Phone"]),
            "screen": self._object(["Part_Screen"]),
            "camera_module": self._object(["Part_Camera_Module", "Part_Camear_Module"]),
        }
        self.part_stock_positions = {
            "car_base": [-0.626, 0.803, SEGMENT_HEIGHT + 0.005],
            "car_frame": [-0.626, 0.843, SEGMENT_HEIGHT + 0.004],
            "phone_base": [0.626, 0.803, SEGMENT_HEIGHT + 0.005],
            "screen": [0.626, 0.843, SEGMENT_HEIGHT + 0.004],
            "camera_module": [0.626, 0.883, SEGMENT_HEIGHT + 0.004],
        }
        for name, handle in self.part_templates.items():
            self.sim.setObjectParent(handle, -1, True)
            self.sim.setObjectPosition(handle, -1, self.part_stock_positions[name])
        self.output_bins = {
            "line_a": self._object(["OutputBin_A"]),
            "line_b": self._object(["OutputBin_B"]),
        }

        self.collision = CollisionManager(
            self.sim, {**self.shuttle_handles, **self.obstacle_handles})
        for name, shuttle in self.shuttles.items():
            shuttle.set_collision_checker(self.collision.checker_for(name))

    def execute_tool_plan(self, plan: dict) -> dict:
        self._init_tool_state()
        steps = plan.get("steps", [])
        results = []
        planning_source = plan.get("planning_source")
        print(f"[Planner] execute plan: {plan.get('plan_name', '<unnamed>')}")
        index = 1
        while index <= len(steps):
            step = steps[index - 1]
            next_step = steps[index] if index < len(steps) else None
            if next_step is not None and self._can_parallel_tool_pair(step, next_step):
                print(f"[Tool] {index:02d}-{index + 1:02d}. parallel tools")
                try:
                    self._execute_parallel_pair(step, next_step)
                except ProductionStepError as exc:
                    if "Parallel shuttle transport" not in str(exc):
                        for failed_index, failed_step in (
                            (index, step),
                            (index + 1, next_step),
                        ):
                            results.append(ToolResult(
                                index=failed_index,
                                tool=failed_step["tool"],
                                args=failed_step.get("args", {}),
                                ok=False,
                                message=str(exc),
                            ))
                        print(f"[Tool] {index:02d}-{index + 1:02d}. failed: {exc}")
                        return ExecutionReport(
                            plan_name=plan.get("plan_name", "<unnamed>"),
                            ok=False,
                            steps=results,
                            planning_source=planning_source,
                        ).to_dict()

                    print(
                        f"[Tool] {index:02d}-{index + 1:02d}. "
                        f"parallel unsafe, fallback to serial: {exc}")
                    for serial_index, serial_step in ((index, step), (index + 1, next_step)):
                        tool = serial_step["tool"]
                        args = serial_step.get("args", {})
                        try:
                            message = self._execute_tool(tool, args) or "completed"
                        except Exception as serial_exc:
                            results.append(ToolResult(
                                index=serial_index,
                                tool=tool,
                                args=args,
                                ok=False,
                                message=str(serial_exc),
                            ))
                            print(f"[Tool] {serial_index:02d}. failed: {serial_exc}")
                            return ExecutionReport(
                                plan_name=plan.get("plan_name", "<unnamed>"),
                                ok=False,
                                steps=results,
                                planning_source=planning_source,
                            ).to_dict()
                        results.append(ToolResult(
                            index=serial_index,
                            tool=tool,
                            args=args,
                            ok=True,
                            message=f"{message} (serial fallback)",
                        ))
                    index += 2
                    continue
                for done_index, done_step in ((index, step), (index + 1, next_step)):
                    results.append(ToolResult(
                        index=done_index,
                        tool=done_step["tool"],
                        args=done_step.get("args", {}),
                        ok=True,
                        message="completed in parallel",
                    ))
                index += 2
                continue

            tool = step["tool"]
            args = step.get("args", {})
            print(f"[Tool] {index:02d}. {tool} {args}")
            try:
                message = self._execute_tool(tool, args) or "completed"
            except Exception as exc:
                results.append(ToolResult(
                    index=index,
                    tool=tool,
                    args=args,
                    ok=False,
                    message=str(exc),
                ))
                print(f"[Tool] {index:02d}. failed: {exc}")
                return ExecutionReport(
                    plan_name=plan.get("plan_name", "<unnamed>"),
                    ok=False,
                    steps=results,
                    planning_source=planning_source,
                ).to_dict()
            results.append(ToolResult(
                index=index,
                tool=tool,
                args=args,
                ok=True,
                message=message,
            ))
            index += 1
        return ExecutionReport(
            plan_name=plan.get("plan_name", "<unnamed>"),
            ok=True,
            steps=results,
            planning_source=planning_source,
        ).to_dict()

    def _init_tool_state(self):
        self.tool_state = {
            "station": {"A": "pick", "A2": "clear", "B": "pick", "B2": "clear"},
            "parts": {"A": {}, "A2": {}, "B": {}, "B2": {}},
            "holding": {"A": None, "B": None, "mid": None},
        }

    def _execute_tool(self, tool: str, args: dict):
        spec = TOOL_REGISTRY.get(tool)
        if spec is None:
            raise ProductionStepError(f"Unknown tool: {tool}")
        executor = getattr(self, spec.executor_name, None)
        if executor is None:
            raise ProductionStepError(
                f"Tool {tool} has missing executor: {spec.executor_name}")
        executor(spec, args)
        return "completed"

    def _execute_transport_tool(self, spec, _args: dict):
        line, source, target = spec.route
        self._tool_transport(line, source, target)

    def _can_parallel_tool_pair(self, first: dict, second: dict) -> bool:
        first_spec = TOOL_REGISTRY.get(first.get("tool"))
        second_spec = TOOL_REGISTRY.get(second.get("tool"))
        if first_spec is None or second_spec is None:
            return False
        if "cross_line" in {first_spec.category, second_spec.category}:
            return False
        if first_spec.category != "transport" or second_spec.category != "transport":
            return False
        first_family = self._tool_family(first_spec)
        second_family = self._tool_family(second_spec)
        if first_family is None or second_family is None:
            return False
        if first_family == second_family:
            return self._can_parallel_same_family_pair(first_spec, second_spec)
        return True

    def _can_parallel_same_family_pair(self, first_spec, second_spec) -> bool:
        if first_spec.category != "transport" or second_spec.category != "transport":
            return False
        pair = frozenset({first_spec.name, second_spec.name})
        return pair in {
            frozenset({
                "Transport_A2_Assemble_Clear",
                "Transport_A_Forward_Assemble",
            }),
            frozenset({
                "Transport_B2_Assemble_Clear",
                "Transport_B_Forward_Assemble",
            }),
        }

    def _execute_parallel_pair(self, first: dict, second: dict):
        first_spec = TOOL_REGISTRY[first["tool"]]
        second_spec = TOOL_REGISTRY[second["tool"]]
        if first_spec.category == "transport" and second_spec.category == "transport":
            self._execute_transport_pair(first, second)
            return

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(self._execute_tool, first["tool"], first.get("args", {})),
                executor.submit(self._execute_tool, second["tool"], second.get("args", {})),
            ]
            for future in as_completed(futures):
                future.result()

    def _execute_transport_pair(self, first: dict, second: dict):
        specs = [TOOL_REGISTRY[first["tool"]], TOOL_REGISTRY[second["tool"]]]
        moves = []
        for spec in specs:
            line, source, target = spec.route
            self._require_station(line, source)
            owner = self._line_owner(line)
            x, y = self._station_position(line, target)
            moves.append((line, source, target, owner, self.shuttles[owner], x, y))
        self._move_shuttles_parallel(moves)
        for line, _source, target, _owner, _shuttle, _x, _y in moves:
            self.tool_state["station"][line] = target

    def _execute_load_tool(self, spec, args: dict):
        self._tool_load(spec.line, args)

    def _execute_hold_tool(self, spec, args: dict):
        self._tool_hold_from(spec.source_line, spec.arm_line, args)

    def _execute_place_tool(self, spec, args: dict):
        self._tool_place(spec.line, args)

    def _execute_inspect_tool(self, spec, _args: dict):
        self._tool_inspect(spec.line)

    def _execute_unload_tool(self, spec, args: dict):
        self._tool_unload(spec.line, args)

    def _execute_cross_line_tool(self, spec, args: dict):
        self._tool_cross_line(spec.source_line, spec.target_line, args)

    def _tool_transport(self, line: str, source: str, target: str):
        self._require_station(line, source)
        owner = self._line_owner(line)
        shuttle = self.shuttles[owner]
        x, y = self._station_position(line, target)
        self._move_shuttle(owner, shuttle, x, y, f"{line}_{source}_{target}")
        self.tool_state["station"][line] = target

    def _tool_load(self, line: str, args: dict):
        self._require_station(line, "pick")
        part = args["part"]
        if part in self.tool_state["parts"][line]:
            raise ProductionStepError(f"{part} is already on line {line}")
        part_handle = self._fresh_part(part)
        owner = self._line_owner(line)
        local_offset = tuple(args.get("local_offset", (0.0, 0.0)))
        self._load_part_to_shuttle(
            owner,
            self._put_arm(line),
            part_handle,
            self.shuttles[owner],
            self.shuttle_handles[owner],
            self._line_center_x(line),
            Y_PUT,
            SHUTTLE_PART_Z,
            local_offset=local_offset)
        self.tool_state["parts"][line][part] = part_handle

    def _tool_hold(self, line: str, args: dict):
        self._tool_hold_from(line, line, args)

    def _tool_hold_from(self, source_line: str, arm_line: str, args: dict):
        self._require_station(source_line, "assemble")
        part = args["part"]
        if self.tool_state["holding"][arm_line] is not None:
            raise ProductionStepError(f"Line {arm_line} assemble arm is already holding")
        if part not in self.tool_state["parts"][source_line]:
            raise ProductionStepError(f"{part} is not on line {source_line} shuttle")
        handle = self.tool_state["parts"][source_line].pop(part)
        self._pick_and_hold(self._assemble_arm(arm_line), handle)
        self.tool_state["holding"][arm_line] = {"part": part, "handle": handle}

    def _tool_place(self, line: str, args: dict):
        self._require_station(line, "assemble")
        holding = self.tool_state["holding"][line]
        part = args["part"]
        if holding is None or holding["part"] != part:
            raise ProductionStepError(f"Line {line} assemble arm is not holding {part}")

        attach_to = args.get("attach_to")
        attach_handle = None
        if attach_to is not None:
            if attach_to not in self.tool_state["parts"][line]:
                raise ProductionStepError(f"{attach_to} is not on line {line} shuttle")
            attach_handle = self.tool_state["parts"][line][attach_to]

        layer = args.get("layer")
        if layer is None:
            layer = self._default_layer(part, attach_to)
        local_offset = tuple(args.get("local_offset", (0.0, 0.0)))
        target_pos = [
            self._line_center_x(line) + local_offset[0],
            Y_ASSEM + local_offset[1],
            SEGMENT_HEIGHT + SHUTTLE_PART_Z,
        ]
        self._place_held_on_shuttle(
            self._assemble_arm(line),
            holding["handle"],
            target_pos,
            self.shuttle_handles[self._line_owner(line)],
            SHUTTLE_PART_Z + float(layer) * ASSEMBLY_LAYER_Z,
            attach_to=attach_handle,
            local_offset=local_offset)

        self.tool_state["holding"][line] = None
        if attach_to is None:
            self.tool_state["parts"][line][part] = holding["handle"]

    def _tool_inspect(self, line: str):
        self._require_station(line, "camera")
        owner = self._line_owner(line)
        self._inspect(self.shuttles[owner], owner, self._line_center_x(line), Y_CAMERA)

    def _tool_unload(self, line: str, args: dict):
        self._require_station(line, "output")
        part = args["part"]
        if part not in self.tool_state["parts"][line]:
            raise ProductionStepError(f"{part} is not on line {line} shuttle")
        handle = self.tool_state["parts"][line].pop(part)
        self._finish_product(
            self._pick_arm(line),
            handle,
            self.output_bins[self._line_owner(line)])

    def _tool_cross_line(self, source: str, target: str, args: dict):
        part = args["part"]
        if part not in self.tool_state["parts"][source]:
            raise ProductionStepError(f"{part} is not on line {source} shuttle")
        handle = self.tool_state["parts"][source].pop(part)
        self.transfer_part_between_lines(
            handle,
            self._line_owner(source),
            self._line_owner(target),
            y=Y_ASSEM,
            target_offset=tuple(args.get("target_offset", (0.0, 0.0))))
        self.tool_state["parts"][target][part] = handle
        self.tool_state["station"][source] = "assemble"
        self.tool_state["station"][target] = "assemble"

    def _require_station(self, line: str, station: str):
        current = self.tool_state["station"][line]
        if current != station:
            raise ProductionStepError(
                f"Line {line} is at {current}, expected {station}")

    def _line_owner(self, line: str) -> str:
        if line == "A":
            return "line_a"
        if line == "A2":
            return "line_a_aux"
        if line == "B":
            return "line_b"
        if line == "B2":
            return "line_b_aux"
        raise ProductionStepError(f"Unknown line: {line}")

    @staticmethod
    def _line_family(line: str) -> str:
        return "A" if line.startswith("A") else "B"

    def _tool_family(self, spec):
        if spec.category == "transport" and spec.route is not None:
            return self._line_family(spec.route[0])
        if spec.category in {"load", "place", "inspect", "unload"} and spec.line:
            return self._line_family(spec.line)
        if spec.category == "hold" and spec.arm_line:
            return self._line_family(spec.arm_line)
        return None

    def _line_center_x(self, line: str) -> float:
        return LINE_A_CENTER_X if line in {"A", "A2"} else LINE_B_CENTER_X

    def _station_position(self, line: str, station: str):
        if line == "A" and station == "forward":
            return LINE_A_CENTER_X, Y_ASSEM - ASSEMBLE_FOLLOW_GAP_Y
        if line == "A2" and station == "clear":
            pos = self.shuttle_initial_positions["line_a_aux"]
            return pos[0], pos[1]
        if line == "B" and station == "forward":
            return LINE_B_CENTER_X, Y_ASSEM - ASSEMBLE_FOLLOW_GAP_Y
        if line == "B" and station == "clear":
            return LINE_B_CENTER_X - B_CLEAR_OFFSET_X, Y_ASSEM
        if line == "B2" and station == "clear":
            pos = self.shuttle_initial_positions["line_b_aux"]
            return pos[0], pos[1]
        return self._line_center_x(line), STATION_Y[station]

    def _put_arm(self, line: str):
        return self.arms["put_a" if line in {"A", "A2"} else "put_b"]

    def _assemble_arm(self, line: str):
        return self.arms["assemble_car" if line == "A" else "assemble_phone"]

    def _pick_arm(self, line: str):
        return self.arms["pick_car" if line == "A" else "pick_phone"]

    def _default_layer(self, part: str, attach_to) -> int:
        if attach_to is None:
            return 0
        if part == "screen":
            return 2
        return 1

    def _produce_one_car(self):
        for _ in self._car_workflow():
            pass

    def _produce_one_phone(self):
        for _ in self._phone_workflow():
            pass

    @staticmethod
    def _build_line_queues(tasks: list[dict]) -> dict[str, deque[str]]:
        queues = {"line_a": deque(), "line_b": deque()}
        for task in tasks:
            product = task["product"]
            quantity = int(task.get("quantity", 1))
            target_queue = None
            if product == "car":
                target_queue = queues["line_a"]
            elif product == "phone":
                target_queue = queues["line_b"]
            else:
                raise ValueError(f"Unsupported product: {product}")
            for _ in range(quantity):
                target_queue.append(product)
        return queues

    def _run_line_scheduler(self, queues: dict[str, deque[str]]):
        active = {}
        total = sum(len(queue) for queue in queues.values())
        completed = 0
        if total == 0:
            return

        while completed < total:
            for line in ("line_a", "line_b"):
                if line not in active and queues[line]:
                    product = queues[line].popleft()
                    active[line] = self._workflow_for_product(product)
                    print(f"[Scheduler] {line} start {product}")

            for line in ("line_a", "line_b"):
                workflow = active.get(line)
                if workflow is None:
                    continue
                try:
                    step = next(workflow)
                    print(f"[Scheduler] {line}: {step}")
                except StopIteration:
                    completed += 1
                    del active[line]

    def _workflow_for_product(self, product: str):
        if product == "car":
            return self._car_workflow()
        if product == "phone":
            return self._phone_workflow()
        raise ValueError(f"Unsupported product: {product}")

    def _car_workflow(self):
        print("[Factory] start car")
        frame = self._fresh_part("car_frame")
        base = self._fresh_part("car_base")
        shuttle = self.shuttles["line_a"]
        shuttle_handle = self.shuttle_handles["line_a"]
        assemble_pos = [LINE_A_CENTER_X, Y_ASSEM, SEGMENT_HEIGHT + SHUTTLE_PART_Z]

        # 1. The car frame comes from Shelf_A, then Robot_Assemble_Car holds it.
        self._load_part_to_shuttle(
            "line_a", self.arms["put_a"], frame, shuttle,
            shuttle_handle, LINE_A_CENTER_X, Y_PUT, SHUTTLE_PART_Z)
        yield "car frame loaded"
        self._move_shuttle("line_a", shuttle, LINE_A_CENTER_X, Y_ASSEM, "car_frame_to_assembly")
        yield "car frame moved to assembly"
        self._pick_and_hold(self.arms["assemble_car"], frame)
        yield "car frame held by assembly arm"
        self._move_shuttle("line_a", shuttle, LINE_A_CENTER_X, Y_PUT, "car_frame_shuttle_return")
        yield "car shuttle returned for base"

        # 2. The car base also comes from Shelf_A and returns to the same assembly station.
        self._load_part_to_shuttle(
            "line_a", self.arms["put_a"], base, shuttle,
            shuttle_handle, LINE_A_CENTER_X, Y_PUT, SHUTTLE_PART_Z)
        yield "car base loaded"
        self._move_shuttle("line_a", shuttle, LINE_A_CENTER_X, Y_ASSEM, "car_base_to_assembly")
        yield "car base moved to assembly"
        self._place_held_on_shuttle(
            self.arms["assemble_car"], frame, assemble_pos, shuttle_handle,
            SHUTTLE_PART_Z + ASSEMBLY_LAYER_Z, attach_to=base)
        yield "car assembled"

        # 3. Assembled car continues to inspection and output bin.
        self._inspect(shuttle, "line_a", LINE_A_CENTER_X, Y_CAMERA)
        yield "car inspected"
        self._move_shuttle("line_a", shuttle, LINE_A_CENTER_X, Y_POLISH, "car_to_pick")
        yield "car moved to output"
        self._finish_product(self.arms["pick_car"], base, self.output_bins["line_a"])
        print("[Factory] car complete")
        yield "car complete"

    def _phone_workflow(self):
        print("[Factory] start phone")
        base = self._fresh_part("phone_base")
        screen = self._fresh_part("screen")
        camera = self._fresh_part("camera_module")
        shuttle = self.shuttles["line_b"]
        shuttle_handle = self.shuttle_handles["line_b"]
        assemble_pos = [LINE_B_CENTER_X, Y_ASSEM, SEGMENT_HEIGHT + SHUTTLE_PART_Z]
        screen_slot = (0.035, 0.0)

        # 1. Camera is delivered first and held by Robot_Assemble_Phone.
        self._load_part_to_shuttle(
            "line_b", self.arms["put_b"], camera, shuttle,
            shuttle_handle, LINE_B_CENTER_X, Y_PUT, SHUTTLE_PART_Z)
        yield "phone camera loaded"
        self._move_shuttle("line_b", shuttle, LINE_B_CENTER_X, Y_ASSEM, "camera_to_assembly")
        yield "phone camera moved to assembly"
        self._pick_and_hold(self.arms["assemble_phone"], camera)
        yield "phone camera held by assembly arm"
        self._move_shuttle("line_b", shuttle, LINE_B_CENTER_X, Y_PUT, "camera_shuttle_return")
        yield "phone shuttle returned for base"

        # 2. Phone base is delivered from Shelf_B, then the camera is mounted.
        self._load_part_to_shuttle(
            "line_b", self.arms["put_b"], base, shuttle,
            shuttle_handle, LINE_B_CENTER_X, Y_PUT, SHUTTLE_PART_Z)
        yield "phone base loaded"
        self._move_shuttle("line_b", shuttle, LINE_B_CENTER_X, Y_ASSEM, "phone_base_to_assembly")
        yield "phone base moved to assembly"
        self._place_held_on_shuttle(
            self.arms["assemble_phone"], camera, assemble_pos, shuttle_handle,
            SHUTTLE_PART_Z + ASSEMBLY_LAYER_Z, attach_to=base)
        yield "phone camera assembled"

        # 3. The same shuttle returns to Shelf_B with the partial phone.
        # Robot_Put_B places the screen in an offset slot so it does not overlap.
        self._move_shuttle("line_b", shuttle, LINE_B_CENTER_X, Y_PUT, "phone_base_return")
        yield "phone shuttle returned for screen"
        self._load_part_to_shuttle(
            "line_b", self.arms["put_b"], screen, shuttle,
            shuttle_handle, LINE_B_CENTER_X, Y_PUT, SHUTTLE_PART_Z,
            local_offset=screen_slot)
        yield "phone screen loaded"
        self._move_shuttle("line_b", shuttle, LINE_B_CENTER_X, Y_ASSEM, "screen_to_assembly")
        yield "phone screen moved to assembly"
        self._pick_and_hold(self.arms["assemble_phone"], screen)
        yield "phone screen held by assembly arm"
        self._place_held_on_shuttle(
            self.arms["assemble_phone"], screen, assemble_pos, shuttle_handle,
            SHUTTLE_PART_Z + 2 * ASSEMBLY_LAYER_Z, attach_to=base)
        yield "phone assembled"

        # 4. Assembled phone continues to inspection and output bin.
        self._inspect(shuttle, "line_b", LINE_B_CENTER_X, Y_CAMERA)
        yield "phone inspected"
        self._move_shuttle("line_b", shuttle, LINE_B_CENTER_X, Y_POLISH, "phone_to_pick")
        yield "phone moved to output"
        self._finish_product(self.arms["pick_phone"], base, self.output_bins["line_b"])
        print("[Factory] phone complete")
        yield "phone complete"

    def _fresh_part(self, name: str) -> int:
        template = self.part_templates[name]
        try:
            handle = self.sim.copyPasteObjects([template], 0)[0]
        except Exception:
            handle = template
        self.sim.setObjectParent(handle, -1, True)
        self.sim.setObjectPosition(handle, -1, self.part_stock_positions[name])
        return handle

    def transfer_part_between_lines(self, part_handle: int,
                                    source: str,
                                    target: str,
                                    y: float = Y_ASSEM,
                                    target_offset: tuple[float, float] = (0.0, 0.0)):
        """Use Robot_Put_Mid to move a part from one line shuttle to the other."""
        if source not in self.shuttles or target not in self.shuttles:
            raise ValueError(f"Unknown transfer line: {source} -> {target}")

        source_x = self._mid_handoff_x(source)
        target_x = self._mid_handoff_x(target)
        source_shuttle = self.shuttles[source]
        target_shuttle = self.shuttles[target]
        target_handle = self.shuttle_handles[target]

        self._move_shuttle(source, source_shuttle, source_x, y,
                           f"mid_transfer_{source}_source")
        self._move_shuttle(target, target_shuttle, target_x, y,
                           f"mid_transfer_{target}_target")

        arm = self.arms["put_mid"]
        self._pick_and_hold(arm, part_handle)
        target_pos = [
            target_x + target_offset[0],
            y + target_offset[1],
            SEGMENT_HEIGHT + SHUTTLE_PART_Z,
        ]
        self._place_held_on_shuttle(
            arm, part_handle, target_pos, target_handle,
            SHUTTLE_PART_Z, local_offset=target_offset)

    def _line_x(self, line: str) -> float:
        if line == "line_a":
            return LINE_A_CENTER_X
        if line == "line_b":
            return LINE_B_CENTER_X
        raise ValueError(f"Unknown line: {line}")

    def _mid_handoff_x(self, line: str) -> float:
        min_gap = SHUTTLE_SIZE_X + 2 * SHUTTLE_SAFE_MARGIN + 0.005
        inner_x = min_gap / 2
        if line == "line_a":
            return -inner_x
        if line == "line_b":
            return inner_x
        raise ValueError(f"Unknown line: {line}")

    def _move_shuttle(self, owner: str, shuttle, x: float, y: float, key: str):
        with self.collision.reserve(owner, key, x, y):
            for wx, wy in self._avoidance_path(owner, shuttle, x, y):
                shuttle.move_to(wx, wy, DEFAULT_SHUTTLE_SPEED)

    def _move_shuttles_parallel(self, moves: list[tuple]):
        plans = []
        for _line, source, target, owner, shuttle, target_x, target_y in moves:
            pose = shuttle.get_pose()
            start_x, start_y, z = pose["x"], pose["y"], pose["z"]
            dx = target_x - start_x
            dy = target_y - start_y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < 1e-6:
                continue
            speeds = shuttle._trapezoidal_profile(distance, DEFAULT_SHUTTLE_SPEED,
                                                  shuttle.max_accel)
            plans.append({
                "key": f"{owner}_{source}_{target}_parallel",
                "owner": owner,
                "shuttle": shuttle,
                "start_x": start_x,
                "start_y": start_y,
                "target_x": target_x,
                "target_y": target_y,
                "z": z,
                "ux": dx / distance,
                "uy": dy / distance,
                "distance": distance,
                "speeds": speeds,
                "traveled": 0.0,
            })
        if not plans:
            return

        max_steps = max(len(plan["speeds"]) for plan in plans)
        for step_index in range(max_steps):
            candidates = []
            for plan in plans:
                if step_index >= len(plan["speeds"]):
                    candidates.append((plan, plan["target_x"], plan["target_y"]))
                    continue
                ds = plan["speeds"][step_index] * self.dt
                plan["traveled"] = min(plan["distance"], plan["traveled"] + ds)
                x = plan["start_x"] + plan["ux"] * plan["traveled"]
                y = plan["start_y"] + plan["uy"] * plan["traveled"]
                candidates.append((plan, x, y))
            self._require_parallel_candidates_safe(candidates)
            for plan, x, y in candidates:
                plan["shuttle"]._set_pose(x, y, plan["z"])
            self.sim.step()
            if self.render_delay > 0:
                time.sleep(self.render_delay)

        final_candidates = [
            (plan, plan["target_x"], plan["target_y"])
            for plan in plans
        ]
        self._require_parallel_candidates_safe(final_candidates)
        for plan, x, y in final_candidates:
            plan["shuttle"]._set_pose(x, y, plan["z"])
        self.sim.step()
        if self.render_delay > 0:
            time.sleep(self.render_delay)

    def _require_parallel_candidates_safe(self, candidates: list[tuple]) -> None:
        half_x = SHUTTLE_SIZE_X / 2 + SHUTTLE_SAFE_MARGIN
        half_y = SHUTTLE_SIZE_Y / 2 + SHUTTLE_SAFE_MARGIN
        for left_index, (_left_plan, left_x, left_y) in enumerate(candidates):
            for _right_plan, right_x, right_y in candidates[left_index + 1:]:
                if abs(left_x - right_x) < 2 * half_x and abs(left_y - right_y) < 2 * half_y:
                    raise ProductionStepError(
                        "Parallel shuttle transport would violate safe spacing."
                    )

    def _avoidance_path(self, owner: str, shuttle, target_x: float, target_y: float):
        pose = shuttle.get_pose()
        start_x, start_y = pose["x"], pose["y"]
        if self._segment_is_safe(owner, start_x, start_y, target_x, target_y):
            return [(target_x, target_y)]

        if owner == "line_a":
            lanes = [self._side_lane_x(owner, LINE_A_CENTER_X, "static_a2", 1)]
        elif owner == "line_b":
            lanes = [self._side_lane_x(owner, LINE_B_CENTER_X, "static_b2", -1)]
        else:
            lanes = [target_x]
        candidates = []
        for lane_x in lanes:
            candidates.append([(lane_x, start_y), (lane_x, target_y), (target_x, target_y)])

        for path in candidates:
            x0, y0 = start_x, start_y
            safe = True
            for x1, y1 in path:
                if not self._segment_is_safe(owner, x0, y0, x1, y1):
                    safe = False
                    break
                x0, y0 = x1, y1
            if safe:
                print(f"[Factory] avoidance path for {owner}: {path}")
                return path

        return [(target_x, target_y)]

    def _side_lane_x(self, owner: str, line_x: float,
                     obstacle_name: str, direction: int) -> float:
        half_box = SHUTTLE_SIZE_X / 2 + SHUTTLE_SAFE_MARGIN
        min_center_gap = 2 * half_box + 0.005
        obstacle_handle = self.obstacle_handles.get(obstacle_name)
        if obstacle_handle is None:
            return line_x + direction * (min_center_gap / 2)
        obstacle_x = self.sim.getObjectPosition(obstacle_handle, -1)[0]
        lane_x = obstacle_x + direction * min_center_gap
        if direction > 0:
            return max(line_x, lane_x)
        return min(line_x, lane_x)

    def _segment_is_safe(self, owner: str, x0: float, y0: float,
                         x1: float, y1: float, samples: int = 24) -> bool:
        for i in range(samples + 1):
            t = i / samples
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            if not self.collision.is_safe(owner, x, y):
                return False
        return True

    def _load_part_to_shuttle(self, owner, arm, part_handle, shuttle,
                              shuttle_handle, x, y, z_offset,
                              local_offset: tuple[float, float] = (0.0, 0.0)):
        self._move_shuttle(owner, shuttle, x, y, f"{arm.name}_load")
        part_pos = self.sim.getObjectPosition(part_handle, -1)
        picked = arm.pick_from_position(part_pos, part_handle)
        self._require_action(picked, arm, "load pick")
        self._require_holding(arm, part_handle, "load pick")
        place_pos = [
            x + local_offset[0],
            y + local_offset[1],
            SEGMENT_HEIGHT + SHUTTLE_PART_Z,
        ]
        placed = arm.place_at_position(
            place_pos,
            parent_handle=shuttle_handle,
            z_offset=z_offset,
            local_offset=local_offset)
        self._require_action(placed, arm, "load place")
        arm.move_to_home()

    def _pick_and_hold(self, arm, part_handle):
        part_pos = self.sim.getObjectPosition(part_handle, -1)
        picked = arm.pick_from_position(part_pos, part_handle)
        self._require_action(picked, arm, "pick")
        self._require_holding(arm, part_handle, "pick")

    def _place_held_on_shuttle(self, arm, held_part, target_pos, shuttle_handle,
                               z_offset, attach_to=None,
                               local_offset: tuple[float, float] = (0.0, 0.0)):
        placed = arm.place_at_position(
            target_pos, parent_handle=shuttle_handle,
            z_offset=z_offset, local_offset=local_offset)
        self._require_action(placed, arm, "place")
        if attach_to is not None:
            self.sim.setObjectParent(held_part, attach_to, True)
        arm.move_to_home()

    def _finish_product(self, arm, product_handle, output_bin_handle):
        product_pos = self.sim.getObjectPosition(product_handle, -1)
        picked = arm.pick_from_position(product_pos, product_handle)
        self._require_action(picked, arm, "finish pick")
        self._require_holding(arm, product_handle, "finish pick")
        bin_pos = self.sim.getObjectPosition(output_bin_handle, -1)
        drop = [bin_pos[0], bin_pos[1], SEGMENT_HEIGHT + SHUTTLE_PART_Z]
        placed = arm.place_at_position(
            drop, parent_handle=output_bin_handle,
            z_offset=SHUTTLE_PART_Z)
        self._require_action(placed, arm, "finish place")
        arm.move_to_home()

    def _inspect(self, shuttle, owner: str, x: float, y: float):
        self._move_shuttle(owner, shuttle, x, y, "camera")
        print(f"[Factory] camera inspection: {owner}")
        for _ in range(DEFAULT_INSPECT_STEPS):
            self.sim.step()
            if self.render_delay > 0:
                time.sleep(self.render_delay)

    def _require_action(self, ok: bool, arm, action: str):
        if not ok:
            raise ProductionStepError(
                f"{arm.name} failed to complete required action: {action}")

    def _require_holding(self, arm, part_handle: int, action: str):
        if not arm.is_holding:
            raise ProductionStepError(
                f"{arm.name} did not hold part {part_handle} after {action}")

    def _arm(self, name: str) -> RobotArmController:
        self._object([name])
        return RobotArmController(self.sim, name, self.dt, self.render_delay)

    def _object(self, aliases: list[str]) -> int:
        errors = []
        for alias in aliases:
            path = alias if alias.startswith("/") else f"/{alias}"
            try:
                return self.sim.getObject(path)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        raise SceneObjectError(
            "Required scene object not found. Tried: " + ", ".join(aliases)
        )
