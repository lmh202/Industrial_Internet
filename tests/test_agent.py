import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent import (
    PlanValidationError,
    ProductionAgent,
    validate_plan,
    validate_plan_sequence,
)
from collision_manager import CollisionManager
from factory_controller import FactoryController, ProductionStepError
from scene_config import (
    LINE_B_CENTER_X,
    SHUTTLE_SAFE_MARGIN,
    SHUTTLE_SIZE_X,
    SHUTTLE_SIZE_Y,
    Y_ASSEM,
)
from tool_registry import TOOL_REGISTRY, build_tool_prompt
import main
import planner


CAR_PLAN = {
    "plan_name": "one_car",
    "steps": [
        {"tool": "Load_A_Pick", "args": {"part": "car_base"}},
        {"tool": "Transport_A_Pick_Assemble", "args": {}},
        {"tool": "Transport_A_Assemble_Forward", "args": {}},
        {"tool": "Transport_A2_Clear_Pick", "args": {}},
        {"tool": "Load_A2_Pick", "args": {"part": "car_frame"}},
        {"tool": "Transport_A2_Pick_Assemble", "args": {}},
        {"tool": "Hold_A2_Assemble", "args": {"part": "car_frame"}},
        {"tool": "Transport_A2_Assemble_Clear", "args": {}},
        {"tool": "Transport_A_Forward_Assemble", "args": {}},
        {
            "tool": "Place_A_Assemble",
            "args": {"part": "car_frame", "attach_to": "car_base", "layer": 1},
        },
        {"tool": "Transport_A_Assemble_Camera", "args": {}},
        {"tool": "Inspect_A", "args": {}},
        {"tool": "Transport_A_Camera_Output", "args": {}},
        {"tool": "Unload_A_Output", "args": {"part": "car_base"}},
    ],
}

PHONE_PLAN = {
    "plan_name": "one_phone",
    "steps": [
        {"tool": "Load_B_Pick", "args": {"part": "phone_base"}},
        {"tool": "Transport_B_Pick_Assemble", "args": {}},
        {"tool": "Transport_B_Assemble_Forward", "args": {}},
        {"tool": "Transport_B2_Clear_Pick", "args": {}},
        {"tool": "Load_B2_Pick", "args": {"part": "screen"}},
        {"tool": "Transport_B2_Pick_Assemble", "args": {}},
        {"tool": "Hold_B2_Assemble", "args": {"part": "screen"}},
        {"tool": "Transport_B2_Assemble_Clear", "args": {}},
        {"tool": "Transport_B_Forward_Assemble", "args": {}},
        {
            "tool": "Place_B_Assemble",
            "args": {"part": "screen", "attach_to": "phone_base", "layer": 1},
        },
        {"tool": "Transport_B_Assemble_Forward", "args": {}},
        {"tool": "Transport_B2_Clear_Pick", "args": {}},
        {"tool": "Load_B2_Pick", "args": {"part": "camera_module"}},
        {"tool": "Transport_B2_Pick_Assemble", "args": {}},
        {"tool": "Hold_B2_Assemble", "args": {"part": "camera_module"}},
        {"tool": "Transport_B2_Assemble_Clear", "args": {}},
        {"tool": "Transport_B_Forward_Assemble", "args": {}},
        {
            "tool": "Place_B_Assemble",
            "args": {"part": "camera_module", "attach_to": "phone_base", "layer": 2},
        },
        {"tool": "Transport_B_Assemble_Camera", "args": {}},
        {"tool": "Inspect_B", "args": {}},
        {"tool": "Transport_B_Camera_Output", "args": {}},
        {"tool": "Unload_B_Output", "args": {"part": "phone_base"}},
    ],
}


class FakeLLMClient:
    def __init__(self, payload):
        self.payloads = payload if isinstance(payload, list) else [payload]
        self.calls = 0
        self.user_prompts = []

    @property
    def is_configured(self):
        return True

    def chat_json(self, _system_prompt, _user_prompt):
        self.user_prompts.append(_user_prompt)
        index = min(self.calls, len(self.payloads) - 1)
        self.calls += 1
        return self.payloads[index]


class DisabledLLMClient:
    @property
    def is_configured(self):
        return False


class PlannerTests(unittest.TestCase):
    def test_validates_car_plan(self):
        self.assertEqual(validate_plan(CAR_PLAN)["plan_name"], "one_car")

    def test_validates_phone_plan(self):
        self.assertEqual(validate_plan(PHONE_PLAN)["plan_name"], "one_phone")

    def test_agent_returns_validated_tool_plan(self):
        agent = ProductionAgent(llm_client=FakeLLMClient(CAR_PLAN))
        plan = agent.run("生产一辆车")
        self.assertEqual(plan["steps"][0]["tool"], "Load_A_Pick")

    def test_agent_uses_llm_before_matching_rule_plan(self):
        llm = FakeLLMClient({"approved": True, "reason": "candidate matches request"})
        agent = ProductionAgent(llm_client=llm)
        plan = agent.run("生产一辆车")

        self.assertEqual(llm.calls, 1)
        self.assertEqual(plan["plan_name"], "produce_1_car")
        self.assertEqual(plan["planning_source"], "llm_approved_candidate")
        self.assertIn("Candidate JSON", llm.user_prompts[0])

    def test_agent_generates_full_plan_when_candidate_is_rejected(self):
        llm = FakeLLMClient([
            {"approved": False, "reason": "needs changes"},
            CAR_PLAN,
        ])
        agent = ProductionAgent(llm_client=llm)

        plan = agent.run("生产一辆车")

        self.assertEqual(llm.calls, 2)
        self.assertEqual(plan["plan_name"], "one_car")
        self.assertEqual(plan["planning_source"], "llm")
        self.assertEqual(plan["planning_hint"], "rule_candidate")

    def test_agent_does_not_use_rule_fallback_by_default(self):
        bad_plan = {
            "plan_name": "bad",
            "steps": [{"tool": "Load_A_Pick", "args": {}}],
        }
        agent = ProductionAgent(llm_client=FakeLLMClient(bad_plan))

        with self.assertRaises(PlanValidationError):
            agent.run("生产一辆车")

    def test_agent_rule_fallback_requires_explicit_switch(self):
        original = planner.AGENT_RULE_FALLBACK
        planner.AGENT_RULE_FALLBACK = True
        try:
            bad_plan = {
                "plan_name": "bad",
                "steps": [{"tool": "Load_A_Pick", "args": {}}],
            }
            agent = ProductionAgent(llm_client=FakeLLMClient(bad_plan))

            plan = agent.run("生产一辆车")
        finally:
            planner.AGENT_RULE_FALLBACK = original

        self.assertEqual(plan["planning_source"], "rule_fallback")
        self.assertEqual(plan["plan_name"], "produce_1_car")

    def test_rule_plans_unspecified_phone_part_to_output(self):
        agent = ProductionAgent(llm_client=DisabledLLMClient())
        plan = agent.run("把手机产线中的某个零件移到output中")
        tools = [step["tool"] for step in plan["steps"]]

        self.assertEqual(plan["plan_name"], "move_b_phone_base_to_output")
        self.assertEqual(tools, [
            "Load_B_Pick",
            "Transport_B_Pick_Assemble",
            "Transport_B_Assemble_Camera",
            "Transport_B_Camera_Output",
            "Unload_B_Output",
        ])
        self.assertEqual(plan["steps"][0]["args"]["part"], "phone_base")

    def test_rule_plans_phone_screen_to_output_with_aux_shuttle(self):
        agent = ProductionAgent(llm_client=DisabledLLMClient())
        plan = agent.run("把手机产线中的屏幕移到output中")
        tools = [step["tool"] for step in plan["steps"]]

        self.assertEqual(plan["plan_name"], "move_b_screen_to_output")
        self.assertIn("Load_B2_Pick", tools)
        self.assertIn("Hold_B2_Assemble", tools)
        self.assertEqual(plan["steps"][-1],
                         {"tool": "Unload_B_Output", "args": {"part": "screen"}})

    def test_rule_plans_car_frame_to_output(self):
        agent = ProductionAgent(llm_client=DisabledLLMClient())
        plan = agent.run("把汽车产线中的车架送到输出区")

        self.assertEqual(plan["plan_name"], "move_a_car_frame_to_output")
        self.assertEqual(plan["steps"][0],
                         {"tool": "Load_A_Pick", "args": {"part": "car_frame"}})
        self.assertEqual(plan["steps"][-1],
                         {"tool": "Unload_A_Output", "args": {"part": "car_frame"}})

    def test_rule_plans_one_car_production(self):
        agent = ProductionAgent(llm_client=DisabledLLMClient())
        plan = agent.run("生产一辆车")

        self.assertEqual(plan["plan_name"], "produce_1_car")
        self.assertEqual(plan["steps"], CAR_PLAN["steps"])

    def test_rule_plans_one_phone_production(self):
        agent = ProductionAgent(llm_client=DisabledLLMClient())
        plan = agent.run("生产一部手机")

        self.assertEqual(plan["plan_name"], "produce_1_phone")
        self.assertEqual(plan["steps"], PHONE_PLAN["steps"])

    def test_rule_plans_mixed_product_quantities(self):
        agent = ProductionAgent(llm_client=DisabledLLMClient())
        plan = agent.run("生产车一辆手机两部")
        tools = [step["tool"] for step in plan["steps"]]

        self.assertEqual(plan["plan_name"], "produce_1_car_2_phone")
        self.assertEqual(tools.count("Unload_A_Output"), 1)
        self.assertEqual(tools.count("Unload_B_Output"), 2)
        first_phone_unload = tools.index("Unload_B_Output")
        self.assertEqual(tools[first_phone_unload + 1], "Transport_B_Output_Pick")

    def test_rule_interleaves_simultaneous_car_and_phone_start(self):
        agent = ProductionAgent(llm_client=DisabledLLMClient())
        plan = agent.run(
            "\u540c\u65f6\u542f\u52a8\u751f\u4ea7\u4e00\u90e8"
            "\u624b\u673a\u548c\u4e00\u8f86\u6c7d\u8f66"
        )
        tools = [step["tool"] for step in plan["steps"]]

        self.assertEqual(plan["plan_name"], "parallel_start_1_car_1_phone")
        self.assertIn("Load_A_Pick", tools[:4])
        self.assertIn("Load_B_Pick", tools[:4])
        self.assertLess(tools.index("Load_A_Pick"), tools.index("Unload_B_Output"))
        self.assertLess(tools.index("Load_B_Pick"), tools.index("Unload_A_Output"))

    def test_agent_retries_invalid_sequence_plan(self):
        bad_plan = {
            "plan_name": "bad_phone",
            "steps": [
                {"tool": "Load_B_Pick", "args": {"part": "camera_module"}},
                {"tool": "Transport_B_Pick_Assemble", "args": {}},
                {"tool": "Transport_B_Assemble_Clear", "args": {}},
                {"tool": "Load_B_Pick", "args": {"part": "screen"}},
            ],
        }
        llm = FakeLLMClient([bad_plan, PHONE_PLAN])
        agent = ProductionAgent(llm_client=llm)

        plan = agent.run("produce one phone")

        self.assertEqual(plan["plan_name"], "one_phone")
        self.assertEqual(llm.calls, 2)

    def test_rejects_unknown_tool_and_macro(self):
        with self.assertRaises(PlanValidationError):
            validate_plan({
                "plan_name": "bad",
                "steps": [{"tool": "make_car", "args": {}}],
            })

    def test_rejects_unknown_part(self):
        with self.assertRaises(PlanValidationError):
            validate_plan({
                "plan_name": "bad",
                "steps": [{"tool": "Load_A_Pick", "args": {"part": "cup"}}],
            })

    def test_rejects_screen_loaded_on_main_b_shuttle(self):
        with self.assertRaisesRegex(PlanValidationError, "Load_B2_Pick"):
            validate_plan({
                "plan_name": "bad",
                "steps": [{"tool": "Load_B_Pick", "args": {"part": "screen"}}],
            })

    def test_rejects_missing_args(self):
        with self.assertRaises(PlanValidationError):
            validate_plan({
                "plan_name": "bad",
                "steps": [{"tool": "Load_A_Pick"}],
            })

    def test_rejects_invalid_offset(self):
        with self.assertRaises(PlanValidationError):
            validate_plan({
                "plan_name": "bad",
                "steps": [
                    {"tool": "Transport_A_B",
                     "args": {"part": "car_frame", "target_offset": ["x", 0]}}
                ],
            })

    def test_accepts_cross_line_transfer(self):
        plan = validate_plan({
            "plan_name": "transfer",
            "steps": [
                {"tool": "Transport_A_B",
                 "args": {"part": "car_frame", "target_offset": [0.01, 0.0]}}
            ],
        })
        self.assertEqual(plan["steps"][0]["args"]["target_offset"], [0.01, 0.0])

    def test_accepts_output_to_pick_return_tool(self):
        plan = validate_plan({
            "plan_name": "return",
            "steps": [{"tool": "Transport_B_Output_Pick", "args": {}}],
        })
        self.assertEqual(plan["steps"][0]["tool"], "Transport_B_Output_Pick")

    def test_validates_two_phone_plan_with_output_return_between_runs(self):
        plan = validate_plan({
            "plan_name": "two_phones",
            "steps": (
                PHONE_PLAN["steps"]
                + [{"tool": "Transport_B_Output_Pick", "args": {}}]
                + PHONE_PLAN["steps"]
            ),
        })
        tools = [step["tool"] for step in plan["steps"]]
        first_unload = tools.index("Unload_B_Output")
        self.assertEqual(tools[first_unload + 1], "Transport_B_Output_Pick")
        self.assertEqual(tools[first_unload + 2], "Load_B_Pick")

    def test_sequence_rejects_loading_main_b_while_b_is_clear(self):
        plan = validate_plan({
            "plan_name": "bad_phone",
            "steps": [
                {"tool": "Load_B_Pick", "args": {"part": "camera_module"}},
                {"tool": "Transport_B_Pick_Assemble", "args": {}},
                {"tool": "Transport_B_Assemble_Clear", "args": {}},
                {"tool": "Load_B_Pick", "args": {"part": "phone_base"}},
            ],
        })
        with self.assertRaisesRegex(PlanValidationError, "line B at pick"):
            validate_plan_sequence(plan)

    def test_save_plan_overwrites_json_file(self):
        with TemporaryDirectory() as tmpdir:
            original = main.PLAN_OUTPUT_PATH
            main.PLAN_OUTPUT_PATH = Path(tmpdir) / "plan.json"
            try:
                main._save_plan({"plan_name": "first", "steps": []})
                main._save_plan({"plan_name": "second", "steps": []})
                text = main.PLAN_OUTPUT_PATH.read_text(encoding="utf-8")
            finally:
                main.PLAN_OUTPUT_PATH = original
        self.assertIn('"plan_name": "second"', text)
        self.assertNotIn('"plan_name": "first"', text)


class ToolRegistryTests(unittest.TestCase):
    def test_registry_describes_all_expected_tools(self):
        for tool in (
            "Load_A_Pick",
            "Transport_B2_Clear_Pick",
            "Hold_B2_Assemble",
            "Place_B_Assemble",
            "Unload_B_Output",
            "Transport_A_B",
        ):
            spec = TOOL_REGISTRY[tool]
            self.assertEqual(spec.name, tool)
            self.assertTrue(spec.executor_name)
            self.assertIsInstance(spec.args_schema, tuple)

    def test_generated_tool_prompt_uses_registry(self):
        prompt = build_tool_prompt()

        self.assertIn("Load_A_Pick", prompt)
        self.assertIn("Transport_B2_Clear_Pick", prompt)
        self.assertIn("Transport_A_B", prompt)


class FakeSim:
    def __init__(self, positions):
        self.positions = positions

    def getObjectPosition(self, handle, _relative_to):
        return self.positions[handle]


class CollisionManagerTests(unittest.TestCase):
    def test_detects_overlap(self):
        sim = FakeSim({1: [0.0, 0.0, 0.0], 2: [0.5, 0.0, 0.0]})
        mgr = CollisionManager(sim, {"a": 1, "b": 2})
        self.assertFalse(mgr.is_safe("b", 0.01, 0.01))

    def test_allows_free_area(self):
        sim = FakeSim({1: [-0.3305, 0.5, 0.0], 2: [0.3305, -0.5, 0.0]})
        mgr = CollisionManager(sim, {"a": 1, "b": 2})
        self.assertTrue(mgr.is_safe("a", -0.3305, 0.0))


class ToolExecutorStateTests(unittest.TestCase):
    def test_execute_tool_plan_dispatches_steps_in_order(self):
        controller = FactoryController.__new__(FactoryController)
        calls = []
        controller._execute_tool = lambda tool, args: calls.append((tool, args))
        report = controller.execute_tool_plan({
            "plan_name": "dispatch",
            "steps": [
                {"tool": "Load_A_Pick", "args": {"part": "car_frame"}},
                {"tool": "Transport_A_Pick_Assemble", "args": {}},
            ],
        })
        self.assertEqual(
            calls,
            [
                ("Load_A_Pick", {"part": "car_frame"}),
                ("Transport_A_Pick_Assemble", {}),
            ],
        )
        self.assertTrue(report["ok"])
        self.assertEqual(len(report["steps"]), 2)

    def test_execute_tool_plan_reports_failure_and_stops(self):
        controller = FactoryController.__new__(FactoryController)
        calls = []

        def fail_second(tool, args):
            calls.append((tool, args))
            if tool == "Transport_A_Pick_Assemble":
                raise ProductionStepError("blocked path")

        controller._execute_tool = fail_second
        report = controller.execute_tool_plan({
            "plan_name": "dispatch_failure",
            "steps": [
                {"tool": "Load_A_Pick", "args": {"part": "car_frame"}},
                {"tool": "Transport_A_Pick_Assemble", "args": {}},
                {"tool": "Transport_A_Assemble_Camera", "args": {}},
            ],
        })

        self.assertFalse(report["ok"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["steps"][-1]["message"], "blocked path")

    def test_execute_tool_plan_batches_parallel_transport_pair(self):
        controller = FactoryController.__new__(FactoryController)
        pairs = []
        singles = []
        controller._execute_transport_pair = lambda first, second: pairs.append(
            (first["tool"], second["tool"]))
        controller._execute_tool = lambda tool, args: singles.append((tool, args))
        report = controller.execute_tool_plan({
            "plan_name": "parallel_pair",
            "steps": [
                {"tool": "Transport_B2_Assemble_Clear", "args": {}},
                {"tool": "Transport_B_Forward_Assemble", "args": {}},
            ],
        })

        self.assertTrue(report["ok"])
        self.assertEqual(pairs, [(
            "Transport_B2_Assemble_Clear",
            "Transport_B_Forward_Assemble",
        )])
        self.assertEqual(singles, [])
        self.assertEqual(report["steps"][0]["message"], "completed in parallel")

    def test_execute_tool_plan_batches_independent_arm_tools(self):
        controller = FactoryController.__new__(FactoryController)
        calls = []
        controller._execute_tool = lambda tool, args: calls.append((tool, args))
        report = controller.execute_tool_plan({
            "plan_name": "parallel_arms",
            "steps": [
                {"tool": "Hold_A2_Assemble", "args": {"part": "car_frame"}},
                {"tool": "Hold_B2_Assemble", "args": {"part": "screen"}},
            ],
        })

        self.assertTrue(report["ok"])
        self.assertEqual(set(tool for tool, _args in calls), {
            "Hold_A2_Assemble",
            "Hold_B2_Assemble",
        })
        self.assertEqual(report["steps"][0]["message"], "completed in parallel")

    def test_parallel_transport_spacing_failure_falls_back_to_serial(self):
        controller = FactoryController.__new__(FactoryController)
        singles = []

        def fail_parallel(_first, _second):
            raise ProductionStepError(
                "Parallel shuttle transport would violate safe spacing.")

        controller._execute_transport_pair = fail_parallel
        controller._execute_tool = lambda tool, args: singles.append((tool, args))
        report = controller.execute_tool_plan({
            "plan_name": "parallel_fallback",
            "steps": [
                {"tool": "Transport_B2_Assemble_Clear", "args": {}},
                {"tool": "Transport_B_Forward_Assemble", "args": {}},
            ],
        })

        self.assertTrue(report["ok"])
        self.assertEqual(singles, [
            ("Transport_B2_Assemble_Clear", {}),
            ("Transport_B_Forward_Assemble", {}),
        ])
        self.assertIn("serial fallback", report["steps"][0]["message"])

    def test_hold_updates_assemble_state(self):
        controller = FactoryController.__new__(FactoryController)
        controller._init_tool_state()
        controller.tool_state["station"]["A"] = "assemble"
        controller.tool_state["parts"]["A"]["car_frame"] = 101
        controller.arms = {"assemble_car": object()}
        picked = []
        controller._pick_and_hold = lambda arm, handle: picked.append((arm, handle))

        controller._tool_hold("A", {"part": "car_frame"})

        self.assertEqual(picked[0][1], 101)
        self.assertNotIn("car_frame", controller.tool_state["parts"]["A"])
        self.assertEqual(controller.tool_state["holding"]["A"]["part"], "car_frame")

    def test_place_clears_holding_and_keeps_assembly_parent(self):
        controller = FactoryController.__new__(FactoryController)
        controller._init_tool_state()
        controller.tool_state["station"]["A"] = "assemble"
        controller.tool_state["parts"]["A"]["car_base"] = 201
        controller.tool_state["holding"]["A"] = {"part": "car_frame", "handle": 101}
        controller.arms = {"assemble_car": object()}
        controller.shuttle_handles = {"line_a": 301}
        placements = []
        controller._place_held_on_shuttle = (
            lambda *args, **kwargs: placements.append((args, kwargs)))

        controller._tool_place(
            "A",
            {"part": "car_frame", "attach_to": "car_base", "layer": 1},
        )

        self.assertIsNone(controller.tool_state["holding"]["A"])
        self.assertNotIn("car_frame", controller.tool_state["parts"]["A"])
        self.assertEqual(placements[0][1]["attach_to"], 201)

    def test_wrong_station_rejects_hold(self):
        controller = FactoryController.__new__(FactoryController)
        controller._init_tool_state()
        controller.tool_state["station"]["A"] = "pick"
        with self.assertRaises(ProductionStepError):
            controller._tool_hold("A", {"part": "car_frame"})

    def test_b2_hold_uses_b_assemble_arm(self):
        controller = FactoryController.__new__(FactoryController)
        controller._init_tool_state()
        controller.tool_state["station"]["B2"] = "assemble"
        controller.tool_state["parts"]["B2"]["screen"] = 401
        controller.arms = {"assemble_phone": object()}
        picked = []
        controller._pick_and_hold = lambda arm, handle: picked.append((arm, handle))

        controller._execute_tool("Hold_B2_Assemble", {"part": "screen"})

        self.assertEqual(picked[0][1], 401)
        self.assertNotIn("screen", controller.tool_state["parts"]["B2"])
        self.assertEqual(controller.tool_state["holding"]["B"]["part"], "screen")

    def test_b2_transport_to_clear_updates_station(self):
        controller = FactoryController.__new__(FactoryController)
        controller._init_tool_state()
        controller.tool_state["station"]["B2"] = "assemble"
        controller.shuttles = {"line_b_aux": object()}
        controller.shuttle_initial_positions = {"line_b_aux": [0.42, -0.7, 0.1]}
        moves = []
        controller._move_shuttle = lambda owner, shuttle, x, y, key: moves.append(
            (owner, x, y, key))

        controller._execute_tool("Transport_B2_Assemble_Clear", {})

        self.assertEqual(controller.tool_state["station"]["B2"], "clear")
        self.assertEqual(moves[0][0], "line_b_aux")
        self.assertEqual(moves[0][1], 0.42)
        self.assertEqual(moves[0][2], -0.7)

    def test_b_clear_position_uses_minimum_non_overlapping_center_gap(self):
        controller = FactoryController.__new__(FactoryController)
        x, y = controller._station_position("B", "clear")
        expected_x = LINE_B_CENTER_X - (SHUTTLE_SIZE_X + 2 * SHUTTLE_SAFE_MARGIN)
        self.assertAlmostEqual(x, expected_x)
        self.assertAlmostEqual(y, Y_ASSEM)
        self.assertAlmostEqual(LINE_B_CENTER_X - x,
                               SHUTTLE_SIZE_X + 2 * SHUTTLE_SAFE_MARGIN)

    def test_forward_and_aux_assemble_positions_follow_on_same_line(self):
        controller = FactoryController.__new__(FactoryController)
        main_x, main_y = controller._station_position("B", "forward")
        aux_x, aux_y = controller._station_position("B2", "assemble")

        self.assertAlmostEqual(main_x, LINE_B_CENTER_X)
        self.assertAlmostEqual(aux_x, LINE_B_CENTER_X)
        self.assertLess(main_y, Y_ASSEM)
        self.assertAlmostEqual(aux_y, Y_ASSEM)
        self.assertAlmostEqual(
            aux_y - main_y,
            SHUTTLE_SIZE_Y + 2 * SHUTTLE_SAFE_MARGIN,
        )

    def test_output_to_pick_transport_updates_station(self):
        controller = FactoryController.__new__(FactoryController)
        controller._init_tool_state()
        controller.tool_state["station"]["B"] = "output"
        controller.shuttles = {"line_b": object()}
        moves = []
        controller._move_shuttle = lambda owner, shuttle, x, y, key: moves.append(
            (owner, x, y, key))

        controller._execute_tool("Transport_B_Output_Pick", {})

        self.assertEqual(controller.tool_state["station"]["B"], "pick")
        self.assertEqual(moves[0][0], "line_b")


class LineSchedulerTests(unittest.TestCase):
    def test_builds_separate_line_queues(self):
        queues = FactoryController._build_line_queues([
            {"product": "car", "quantity": 1},
            {"product": "phone", "quantity": 2},
            {"product": "car", "quantity": 1},
        ])
        self.assertEqual(list(queues["line_a"]), ["car", "car"])
        self.assertEqual(list(queues["line_b"]), ["phone", "phone"])

    def test_rejects_unsupported_scheduler_product(self):
        with self.assertRaises(ValueError):
            FactoryController._build_line_queues([
                {"product": "cup", "quantity": 1},
            ])


if __name__ == "__main__":
    unittest.main()
