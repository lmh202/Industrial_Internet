"""Deterministic CoppeliaSim factory workflow controller."""

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
        expanded = []
        for task in tasks:
            product = task["product"]
            quantity = int(task.get("quantity", 1))
            for _ in range(quantity):
                expanded.append(product)

        print(f"[Agent] plan: {expanded}")
        for product in expanded:
            if product == "car":
                self._produce_one_car()
            elif product == "phone":
                self._produce_one_phone()
            else:
                raise ValueError(f"Unsupported product: {product}")

    def produce_car(self, quantity: int = 1):
        self.produce_plan([{"product": "car", "quantity": quantity}])

    def produce_phone(self, quantity: int = 1):
        self.produce_plan([{"product": "phone", "quantity": quantity}])

    def _init_scene(self):
        self.arms = {
            "put_a": self._arm("Robot_Put_A"),
            "put_b": self._arm("Robot_Put_B"),
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
        self._move_shuttle("line_a", shuttle, LINE_A_CENTER_X, Y_ASSEM, "car_frame_to_assembly")
        self._pick_and_hold(self.arms["assemble_car"], frame)
        self._move_shuttle("line_a", shuttle, LINE_A_CENTER_X, Y_PUT, "car_frame_shuttle_return")

        # 2. The car base also comes from Shelf_A and returns to the same assembly station.
        self._load_part_to_shuttle(
            "line_a", self.arms["put_a"], base, shuttle,
            shuttle_handle, LINE_A_CENTER_X, Y_PUT, SHUTTLE_PART_Z)
        self._move_shuttle("line_a", shuttle, LINE_A_CENTER_X, Y_ASSEM, "car_base_to_assembly")
        self._place_held_on_shuttle(
            self.arms["assemble_car"], frame, assemble_pos, shuttle_handle,
            SHUTTLE_PART_Z + ASSEMBLY_LAYER_Z, attach_to=base)

        # 3. Assembled car continues to inspection and output bin.
        self._inspect(shuttle, "line_a", LINE_A_CENTER_X, Y_CAMERA)
        self._move_shuttle("line_a", shuttle, LINE_A_CENTER_X, Y_POLISH, "car_to_pick")
        self._finish_product(self.arms["pick_car"], base, self.output_bins["line_a"])
        print("[Factory] car complete")

    def _produce_one_phone(self):
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
        self._move_shuttle("line_b", shuttle, LINE_B_CENTER_X, Y_ASSEM, "camera_to_assembly")
        self._pick_and_hold(self.arms["assemble_phone"], camera)
        self._move_shuttle("line_b", shuttle, LINE_B_CENTER_X, Y_PUT, "camera_shuttle_return")

        # 2. Phone base is delivered from Shelf_B, then the camera is mounted.
        self._load_part_to_shuttle(
            "line_b", self.arms["put_b"], base, shuttle,
            shuttle_handle, LINE_B_CENTER_X, Y_PUT, SHUTTLE_PART_Z)
        self._move_shuttle("line_b", shuttle, LINE_B_CENTER_X, Y_ASSEM, "phone_base_to_assembly")
        self._place_held_on_shuttle(
            self.arms["assemble_phone"], camera, assemble_pos, shuttle_handle,
            SHUTTLE_PART_Z + ASSEMBLY_LAYER_Z, attach_to=base)

        # 3. The same shuttle returns to Shelf_B with the partial phone.
        # Robot_Put_B places the screen in an offset slot so it does not overlap.
        self._move_shuttle("line_b", shuttle, LINE_B_CENTER_X, Y_PUT, "phone_base_return")
        self._load_part_to_shuttle(
            "line_b", self.arms["put_b"], screen, shuttle,
            shuttle_handle, LINE_B_CENTER_X, Y_PUT, SHUTTLE_PART_Z,
            local_offset=screen_slot)
        self._move_shuttle("line_b", shuttle, LINE_B_CENTER_X, Y_ASSEM, "screen_to_assembly")
        self._pick_and_hold(self.arms["assemble_phone"], screen)
        self._place_held_on_shuttle(
            self.arms["assemble_phone"], screen, assemble_pos, shuttle_handle,
            SHUTTLE_PART_Z + 2 * ASSEMBLY_LAYER_Z, attach_to=base)

        # 4. Assembled phone continues to inspection and output bin.
        self._inspect(shuttle, "line_b", LINE_B_CENTER_X, Y_CAMERA)
        self._move_shuttle("line_b", shuttle, LINE_B_CENTER_X, Y_POLISH, "phone_to_pick")
        self._finish_product(self.arms["pick_phone"], base, self.output_bins["line_b"])
        print("[Factory] phone complete")

    def _fresh_part(self, name: str) -> int:
        template = self.part_templates[name]
        try:
            handle = self.sim.copyPasteObjects([template], 0)[0]
        except Exception:
            handle = template
        self.sim.setObjectParent(handle, -1, True)
        self.sim.setObjectPosition(handle, -1, self.part_stock_positions[name])
        return handle

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
        arm.pick_from_position(part_pos, part_handle)
        place_pos = [
            x + local_offset[0],
            y + local_offset[1],
            SEGMENT_HEIGHT + SHUTTLE_PART_Z,
        ]
        arm.place_at_position(
            place_pos,
            parent_handle=shuttle_handle,
            z_offset=z_offset,
            local_offset=local_offset)
        arm.move_to_home()

    def _pick_and_hold(self, arm, part_handle):
        part_pos = self.sim.getObjectPosition(part_handle, -1)
        arm.pick_from_position(part_pos, part_handle)

    def _place_held_on_shuttle(self, arm, held_part, target_pos, shuttle_handle,
                               z_offset, attach_to=None):
        arm.place_at_position(target_pos, parent_handle=shuttle_handle,
                              z_offset=z_offset)
        if attach_to is not None:
            self.sim.setObjectParent(held_part, attach_to, True)
        arm.move_to_home()

    def _finish_product(self, arm, product_handle, output_bin_handle):
        product_pos = self.sim.getObjectPosition(product_handle, -1)
        arm.pick_from_position(product_pos, product_handle)
        bin_pos = self.sim.getObjectPosition(output_bin_handle, -1)
        drop = [bin_pos[0], bin_pos[1], SEGMENT_HEIGHT + SHUTTLE_PART_Z]
        arm.place_at_position(drop, parent_handle=output_bin_handle,
                              z_offset=SHUTTLE_PART_Z)
        arm.move_to_home()

    def _inspect(self, shuttle, owner: str, x: float, y: float):
        self._move_shuttle(owner, shuttle, x, y, "camera")
        print(f"[Factory] camera inspection: {owner}")
        for _ in range(DEFAULT_INSPECT_STEPS):
            self.sim.step()
            if self.render_delay > 0:
                time.sleep(self.render_delay)

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
