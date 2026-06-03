"""Deterministic CoppeliaSim factory workflow controller."""

from collections import deque
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
    Y_ASSEM,
    Y_CAMERA,
    Y_POLISH,
    Y_PUT,
)
from shuttle_controller import ShuttleController


class SceneObjectError(RuntimeError):
    """Raised when required scene objects are missing."""


class ProductionStepError(RuntimeError):
    """Raised when a required pick/place/move step cannot be completed."""


class FactoryController:
    def __init__(self, sim, dt: float = SIM_DT,
                 render_delay: float = SIM_RENDER_DELAY):
        self.sim = sim
        self.dt = dt
        self.render_delay = render_delay
        self.arms: dict[str, RobotArmController] = {}
        self.shuttles: dict[str, ShuttleController] = {}
        self.shuttle_handles: dict[str, int] = {}
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
            "line_b": ["Shuttle_B1", "Shuttle_B0", "Shuttle_B"],
        }.items():
            handle = self._object(aliases)
            self.shuttle_handles[name] = handle
            self.shuttles[name] = ShuttleController(
                self.sim, handle, name=name, dt=self.dt,
                render_delay=self.render_delay)

        for name, aliases in {
            "static_a2": ["Shuttle_A2"],
            "static_b2": ["Shuttle_B2"],
        }.items():
            try:
                self.obstacle_handles[name] = self._object(aliases)
            except SceneObjectError:
                pass

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
        min_gap = SHUTTLE_SIZE_X + 2 * SHUTTLE_SAFE_MARGIN + 0.02
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
        min_center_gap = 2 * half_box + 0.01
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
