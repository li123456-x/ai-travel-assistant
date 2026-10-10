"""交通修改与失败拦截的离线测试；所有工具、模型和执行日志均使用替身。"""

import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_functions(relative_path, namespace):
    path = PROJECT_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for node in functions:
        node.decorator_list = []
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class TransportNodeTests(unittest.TestCase):
    def setUp(self):
        self.intercity = Mock(return_value="交通参考信息，实际信息以官方渠道为准")
        self.routes = Mock(side_effect=lambda arguments: json.dumps({
            "status": "1",
            "route": {
                "transits" if arguments["mode"] == "transit" else "paths": [
                    {"distance": "3000", "duration": "900"}
                ],
            },
        }))
        self.geocoding = Mock(return_value=({
            "真实酒店": {"location": "108.9,34.3", "citycode": "029"},
            "西安城墙": {"location": "108.95,34.25", "citycode": "029"},
        }, {}))
        self.model = Mock(return_value=SimpleNamespace(content="交通方案"))
        self.steps = Mock()
        self.namespace = load_functions("agents/transport.py", {
            "json": json,
            "time": time,
            "ThreadPoolExecutor": ThreadPoolExecutor,
            "search_intercity_transport": SimpleNamespace(invoke=self.intercity),
            "search_transport_route_by_coords": SimpleNamespace(invoke=self.routes),
            "resolve_locations": self.geocoding,
            "model": SimpleNamespace(invoke=self.model),
            "record_execution_step_safe": self.steps,
        })
        self.state = {
            "origin": "临沂", "destination": "西安",
            "departure_date": "2026-10-09", "return_date": "2026-10-10",
            "hotel_name": "真实酒店", "attraction_names": ["西安城墙"],
            "people": 1, "budget": 2000, "constraints": [],
        }

    def run_node(self, **updates):
        return self.namespace["transport_node"]({**self.state, **updates})

    def test_selected_modes_reach_both_real_tool_calls(self):
        result = self.run_node(
            city_transport_mode="driving", intercity_transport_mode="flight",
            transport_revision_request="使用自驾和飞机",
        )
        self.assertEqual(result["transport_result"], "交通方案")
        intercity_arguments = [call.args[0] for call in self.intercity.call_args_list]
        self.assertEqual(len(intercity_arguments), 2)
        self.assertTrue(all(item["transport_mode"] == "flight" for item in intercity_arguments))
        self.assertEqual({item["origin"] for item in intercity_arguments}, {"临沂", "西安"})
        self.assertTrue(all(call.args[0]["mode"] == "driving" for call in self.routes.call_args_list))
        prompt = self.model.call_args.args[0]
        self.assertIn("使用自驾和飞机", prompt)
        self.assertIn("市内方式：driving", prompt)
        self.assertIn("去程及返程方式：flight", prompt)

    def test_default_modes_and_walking_are_supported(self):
        self.run_node()
        self.assertTrue(all(call.args[0]["mode"] == "transit" for call in self.routes.call_args_list))
        self.assertTrue(all(call.args[0]["transport_mode"] == "train" for call in self.intercity.call_args_list))
        self.routes.reset_mock()
        self.run_node(city_transport_mode="walking")
        self.assertTrue(all(call.args[0]["mode"] == "walking" for call in self.routes.call_args_list))

    def test_invalid_modes_stop_before_external_calls(self):
        for updates in [dict(city_transport_mode="taxi"), dict(intercity_transport_mode="bus")]:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.run_node(**updates)
        self.intercity.assert_not_called()

    def test_missing_business_inputs_stop_before_external_calls(self):
        for updates in [dict(hotel_name=" "), dict(attraction_names=[]), dict(attraction_names=[" "])]:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.run_node(**updates)
        self.intercity.assert_not_called()

    def test_empty_or_error_intercity_result_is_logged_and_blocks_model(self):
        for result in ["", "暂未查询到城际交通", "城际交通查询失败：timeout"]:
            self.intercity.return_value = result
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.run_node()
            self.assertFalse(self.steps.call_args.kwargs["success"])
        self.model.assert_not_called()
        self.routes.assert_not_called()

    def test_missing_coordinate_is_logged_and_blocks_model(self):
        self.geocoding.return_value = ({}, {"真实酒店": "找不到地点"})
        with self.assertRaisesRegex(ValueError, "无法规划"):
            self.run_node()
        self.assertFalse(self.steps.call_args.kwargs["success"])
        self.assertEqual(self.steps.call_args.kwargs["step_name"], "amap_geocoding")
        self.model.assert_not_called()
        self.routes.assert_not_called()

    def test_geocoding_exception_is_logged(self):
        self.geocoding.side_effect = RuntimeError("timeout")
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            self.run_node()
        self.assertFalse(self.steps.call_args.kwargs["success"])
        self.model.assert_not_called()

    def test_empty_paths_or_api_error_block_model(self):
        self.routes.side_effect = None
        for result in [
            "路线规划失败：无路线", "not json",
            '{"status":"0"}',
            '{"status":"1","route":{"transits":[]}}',
            '{"status":"1","route":{"transits":[{}]}}',
        ]:
            self.routes.return_value = result
            with self.subTest(result=result), self.assertRaisesRegex(ValueError, "路线查询未完成"):
                self.run_node()
            self.assertFalse(self.steps.call_args.kwargs["success"])
            self.assertEqual(self.steps.call_args.kwargs["step_name"], "amap_city_route")
        self.model.assert_not_called()

    def test_blank_model_output_is_logged_as_failure(self):
        self.model.return_value = SimpleNamespace(content="  ")
        with self.assertRaisesRegex(ValueError, "空说明"):
            self.run_node()
        self.assertFalse(self.steps.call_args.kwargs["success"])


class IntercityToolTests(unittest.TestCase):
    def setUp(self):
        self.search = Mock(return_value={"results": [{
            "title": "参考交通", "content": "实际信息以官方为准", "url": "https://example.com"
        }]})
        self.namespace = load_functions("tools/intercity_transport.py", {})
        self.namespace["_get_tavily_client"] = lambda: SimpleNamespace(search=self.search)

    def invoke(self, mode="train"):
        return self.namespace["search_intercity_transport"]("临沂", "西安", "2026-10-09", mode)

    def test_flight_uses_airline_keywords(self):
        self.invoke("flight")
        query = self.search.call_args.kwargs["query"]
        self.assertIn("航班", query)
        self.assertNotIn("高铁", query)

    def test_train_uses_rail_keywords(self):
        self.invoke()
        self.assertIn("高铁", self.search.call_args.kwargs["query"])

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.invoke("bus")
        self.search.assert_not_called()

    def test_empty_search_does_not_become_a_success_string(self):
        for results in [[], {}, [None], [{"title": "", "content": " "}]]:
            self.search.return_value = {"results": results}
            with self.subTest(results=results), self.assertRaises(ValueError):
                self.invoke()


if __name__ == "__main__":
    unittest.main()
