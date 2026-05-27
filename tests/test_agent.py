import unittest

from agent import ProductionAgent, TaskParseError
from collision_manager import CollisionManager


class AgentParserTests(unittest.TestCase):
    def setUp(self):
        self.agent = ProductionAgent(llm_client=None, allow_rule_fallback=True)
        self.agent.llm_client.api_key = ""

    def test_parse_car_and_phone_order(self):
        tasks = self.agent.run("连续生产一辆车和两部手机")
        self.assertEqual(
            tasks,
            [{"product": "car", "quantity": 1},
             {"product": "phone", "quantity": 2}],
        )

    def test_parse_digits(self):
        tasks = self.agent.run("produce 2 phones and 1 car")
        self.assertEqual(
            tasks,
            [{"product": "phone", "quantity": 2},
             {"product": "car", "quantity": 1}],
        )

    def test_validate_rejects_unknown_product(self):
        with self.assertRaises(TaskParseError):
            self.agent._validate_payload(
                {"tasks": [{"product": "cup", "quantity": 1}]})


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


if __name__ == "__main__":
    unittest.main()
