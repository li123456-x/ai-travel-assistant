"""Exercise the real bounded agent graph with a scripted model and offline tools."""

import ast
import json
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unicodedata
import unittest
from unittest.mock import Mock

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    wrap_model_call,
)
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from config.state import TravelState
from config.trip_policy import requires_hotel, skipped_hotel_update


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOTEL_NAME = "全季酒店大雁塔店"


class ScriptedHotelModel(BaseChatModel):
    responses: list[AIMessage] = Field(default_factory=list)
    seen_messages: list = Field(default_factory=list)
    seen_tools: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-hotel"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.seen_tools.append([getattr(item, "name", None) for item in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_messages.append(list(messages))
        if not self.responses:
            raise AssertionError("Unexpected extra model call")
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def search_call(query="全季", call_id="search-1"):
    return AIMessage(content="", tool_calls=[{
        "name": "search_hotel_candidates", "args": {"search_keyword": query}, "id": call_id,
    }])


def walking_call(hotel_name=HOTEL_NAME, attraction="大雁塔", call_id="walking-1"):
    return AIMessage(content="", tool_calls=[{
        "name": "check_hotel_walking_route",
        "args": {"hotel_name": hotel_name, "attraction_name": attraction}, "id": call_id,
    }])


def decision_call(name=HOTEL_NAME, summary="根据真实候选推荐全季酒店"):
    return AIMessage(content="", tool_calls=[{
        "name": "HotelDecision", "args": {"hotel_name": name, "summary": summary}, "id": "final",
    }])


class HotelReActTests(unittest.TestCase):
    def setUp(self):
        self.model = ScriptedHotelModel()
        self.search = Mock(return_value=json.dumps([{
            "name": HOTEL_NAME, "location": "108.96,34.21", "address": "真实候选地址",
        }], ensure_ascii=False))
        self.geocoding = Mock(return_value={"location": "108.97,34.22", "citycode": "029"})
        self.routes = Mock(return_value=json.dumps({
            "status": "1", "route": {"paths": [{"distance": "1200", "cost": {"duration": "900"}}]},
        }))
        self.steps = Mock()
        self.namespace = {
            "__name__": "_offline_hotel_node", "json": json, "math": math, "threading": threading,
            "time": time, "unicodedata": unicodedata, "BaseModel": BaseModel, "Field": Field,
            "create_agent": create_agent, "ModelCallLimitMiddleware": ModelCallLimitMiddleware,
            "ToolCallLimitMiddleware": ToolCallLimitMiddleware, "wrap_model_call": wrap_model_call,
            "ToolStrategy": ToolStrategy, "tool": tool, "TravelState": dict, "model": self.model,
            "requires_hotel": requires_hotel, "skipped_hotel_update": skipped_hotel_update,
            "record_execution_step_safe": self.steps,
            "search_hotels": SimpleNamespace(invoke=self.search), "get_location_info": self.geocoding,
            "search_transport_route_by_coords": SimpleNamespace(invoke=self.routes),
        }
        # Preserve the actual node and agent factory, isolating only import-time integrations.
        path = PROJECT_ROOT / "agents" / "hotel.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), self.namespace)
        self.state = {
            "destination": "西安", "days": 2, "people": 2, "budget": 3000,
            "preferences": ["历史文化"], "constraints": ["不要太赶"],
            "attractions_result": "安排大雁塔", "attraction_names": ["大雁塔"],
            "hotel_search_keyword": "全季", "hotel_excluded_names": [],
            "hotel_revision_request": "全季，靠近大雁塔", "agent_run_id": 42,
        }

    def run_node(self, responses, **updates):
        self.model.responses = responses
        return self.namespace["hotel_node"]({**self.state, **updates})

    def observation(self, turn):
        messages = self.model.seen_messages[turn]
        return json.loads(next(message.content for message in reversed(messages)
                               if isinstance(message, ToolMessage)))

    def test_search_then_structured_decision_preserves_outer_contract(self):
        result = self.run_node([search_call(), decision_call()])
        self.assertEqual(result, {"hotel_name": HOTEL_NAME, "hotel_result": "根据真实候选推荐全季酒店"})
        self.search.assert_called_once_with({"city": "西安", "keyword": "全季"})
        self.assertEqual(self.observation(1)["status"], "ok")
        model_steps = [call.kwargs for call in self.steps.call_args_list if call.kwargs["step_type"] == "llm"]
        self.assertEqual(len(model_steps), 2)
        self.assertTrue(all(step["agent_run_id"] == 42 and step["success"] for step in model_steps))

    def test_empty_candidates_allow_model_to_change_search(self):
        self.search.side_effect = ["[]", self.search.return_value]
        self.run_node([search_call(), search_call("大雁塔", "search-2"), decision_call()])
        self.assertEqual(self.observation(1)["status"], "empty")
        self.assertEqual(self.search.call_args_list[1].args[0], {"city": "西安", "keyword": "全季 大雁塔"})

    def test_brand_and_exclusions_cannot_be_relaxed_by_model(self):
        self.search.return_value = json.dumps([
            {"name": "全季酒店旧店"}, {"name": "如家酒店"}, {"name": HOTEL_NAME},
        ], ensure_ascii=False)
        self.run_node([search_call("如家"), decision_call()], hotel_excluded_names=["全季 酒店旧店"])
        self.assertEqual(self.search.call_args.args[0]["keyword"], "全季 如家")
        self.assertEqual([item["name"] for item in self.observation(1)["candidates"]], [HOTEL_NAME])

    def test_route_uses_candidate_coordinates_without_geocoding_hotel_again(self):
        self.run_node([search_call(), walking_call(), decision_call()])
        self.geocoding.assert_called_once_with("大雁塔", "西安")
        self.assertEqual(self.routes.call_args.args[0]["origin_location"], "108.96,34.21")
        self.assertEqual(self.routes.call_args.args[0]["mode"], "walking")
        evidence = self.observation(2)
        self.assertEqual(evidence["distance_m"], 1200)
        self.assertEqual(evidence["duration_s"], 900)

    def test_duplicate_search_uses_cached_result(self):
        self.run_node([search_call(), search_call(" 全季 ", "search-2"), decision_call()])
        self.search.assert_called_once()

    def test_duplicate_walking_query_uses_cached_result_and_can_still_finish(self):
        self.run_node([search_call(), walking_call(), walking_call(call_id="walking-2"), decision_call()])
        self.routes.assert_called_once()
        self.geocoding.assert_called_once()
        self.assertEqual(len(self.model.seen_messages), 4)

    def test_missing_coordinate_or_empty_route_is_not_fabricated(self):
        for candidate, response in [
            ({"name": HOTEL_NAME}, self.routes.return_value),
            ({"name": HOTEL_NAME, "location": "108.96,34.21"}, '{"status":"1","route":{"paths":[]}}'),
        ]:
            with self.subTest(candidate=candidate):
                self.model.seen_messages.clear()
                self.search.return_value = json.dumps([candidate], ensure_ascii=False)
                self.routes.return_value = response
                self.run_node([search_call(), walking_call(), decision_call()])
                self.assertEqual(self.observation(2)["status"], "unavailable")
                self.assertNotIn("distance_m", self.observation(2))

    def test_duration_not_returned_by_api_stays_unknown(self):
        self.routes.return_value = '{"status":"1","route":{"paths":[{"distance":"1000"}]}}'
        self.run_node([search_call(), walking_call(), decision_call()])
        self.assertIsNone(self.observation(2)["duration_s"])

    def test_unknown_hotel_or_attraction_stops_before_route_request(self):
        for hotel_name, attraction in [("编造酒店", "大雁塔"), (HOTEL_NAME, "不在行程中的景点")]:
            with self.subTest(hotel=hotel_name, attraction=attraction), self.assertRaisesRegex(ValueError, "只能核查"):
                self.run_node([search_call(), walking_call(hotel_name, attraction), decision_call()])
        self.routes.assert_not_called()
        self.geocoding.assert_not_called()

    def test_unqueried_or_fabricated_final_hotel_is_rejected(self):
        for responses in [[decision_call()], [search_call(), decision_call("编造酒店")]]:
            with self.subTest(responses=responses), self.assertRaisesRegex(ValueError, "真实查询候选"):
                self.run_node(responses)

    def test_tool_failure_propagates_with_original_reason_and_failure_log(self):
        self.search.return_value = "酒店查询请求失败：connection timeout"
        with self.assertRaisesRegex(ValueError, "connection timeout"):
            self.run_node([search_call(), decision_call()])
        self.assertEqual(len(self.model.seen_messages), 1)
        self.assertFalse(self.steps.call_args.kwargs["success"])

    def test_malformed_route_and_invalid_numbers_stop_generation(self):
        for response in [
            "交通查询请求失败：timeout", '{"status":"0"}',
            '{"status":"1","route":{"paths":[{}]}}',
            '{"status":"1","route":{"paths":[{"distance":"NaN"}]}}',
        ]:
            with self.subTest(response=response), self.assertRaises(ValueError):
                self.routes.return_value = response
                self.run_node([search_call(), walking_call(), decision_call()])
            self.assertFalse(self.steps.call_args.kwargs["success"])

    def test_business_tool_budget_blocks_fourth_query(self):
        responses = [search_call(f"全季区域{index}", f"search-{index}") for index in range(4)]
        result = self.run_node(responses + [decision_call()])
        self.assertEqual(result["hotel_name"], HOTEL_NAME)
        self.assertEqual(self.search.call_count, 3)
        self.assertEqual(self.observation(4)["status"], "budget_exhausted")
        self.assertEqual(self.model.seen_tools[-1], ["HotelDecision"])

    def test_oversized_tool_batch_is_blocked_before_requests(self):
        calls = [search_call(f"全季区域{index}", f"search-{index}").tool_calls[0] for index in range(5)]
        result = self.run_node([AIMessage(content="", tool_calls=calls), decision_call()])
        self.assertEqual(result["hotel_name"], HOTEL_NAME)
        self.assertEqual(self.search.call_count, 3)
        self.assertEqual(self.model.seen_tools[-1], ["HotelDecision"])

    def test_one_day_hotel_node_makes_no_model_or_tool_calls(self):
        result = self.run_node([], days=1)
        self.assertEqual(result["hotel_status"], "skipped")
        self.assertEqual(result["hotel_name"], "")
        self.search.assert_not_called()
        self.assertEqual(self.model.seen_messages, [])

    def test_empty_candidates_at_budget_end_cannot_be_fabricated(self):
        self.search.return_value = "[]"
        responses = [search_call(f"全季区域{index}", f"search-{index}") for index in range(3)]
        with self.assertRaisesRegex(ValueError, "没有找到真实酒店候选"):
            self.run_node(responses + [decision_call()])
        self.assertEqual(self.search.call_count, 3)

    def test_six_requested_routes_after_search_finish_with_only_two_real_checks(self):
        calls = [walking_call(call_id=f"walking-{index}").tool_calls[0] for index in range(6)]
        result = self.run_node([search_call(), AIMessage(content="", tool_calls=calls), decision_call()])
        self.assertEqual(result["hotel_name"], HOTEL_NAME)
        self.search.assert_called_once()
        self.routes.assert_called_once()
        tool_steps = [call.kwargs for call in self.steps.call_args_list if call.kwargs["step_type"] == "tool"]
        self.assertEqual(len(tool_steps), 3)

    def test_model_budget_stops_before_next_model_call(self):
        self.namespace["HOTEL_MAX_MODEL_CALLS"] = 2
        with self.assertRaises(ModelCallLimitExceededError):
            self.run_node([search_call(), walking_call(), decision_call()])
        self.assertEqual(len(self.model.seen_messages), 2)

    def test_plain_text_without_structured_decision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "结构化决策"):
            self.run_node([AIMessage(content="直接推荐一个酒店")])
        self.assertEqual(len(self.model.seen_messages), 1)
        self.search.assert_not_called()

    def test_candidates_and_cache_are_not_shared_between_attempts(self):
        self.run_node([search_call(), decision_call()])
        self.run_node([search_call(), decision_call()])
        self.assertEqual(self.search.call_count, 2)

    def test_outer_interrupt_resume_uses_fresh_hotel_context(self):
        def human_review(state):
            decision = interrupt({"hotel_name": state["hotel_name"]})
            if decision["approved"]:
                return Command(goto=END)
            return Command(update={
                "hotel_search_keyword": decision["keyword"],
                "hotel_excluded_names": [state["hotel_name"]],
            }, goto="hotel")

        builder = StateGraph(TravelState)
        builder.add_node("hotel", self.namespace["hotel_node"])
        builder.add_node("human_review", human_review)
        builder.add_edge(START, "hotel")
        builder.add_edge("hotel", "human_review")
        graph = builder.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "offline-hotel-outer"}}
        self.model.responses = [search_call(), decision_call()]
        first = graph.invoke(self.state, config)
        self.assertIn("__interrupt__", first)
        self.assertEqual(first["hotel_name"], HOTEL_NAME)

        new_name = "如家酒店大雁塔店"
        self.search.return_value = json.dumps([{"name": new_name}], ensure_ascii=False)
        self.model.responses = [search_call("如家"), decision_call(new_name)]
        second = graph.invoke(Command(resume={"approved": False, "keyword": "如家"}), config)
        self.assertEqual(second["hotel_name"], new_name)
        self.assertEqual(second["attraction_names"], ["大雁塔"])
        self.assertEqual(self.search.call_count, 2)
        finished = graph.invoke(Command(resume={"approved": True}), config)
        self.assertNotIn("__interrupt__", finished)

    def test_model_error_is_logged_and_propagated(self):
        with self.assertRaisesRegex(AssertionError, "Unexpected extra model call"):
            self.run_node([])
        self.assertEqual(self.steps.call_args.kwargs["step_name"], "hotel_react_llm")
        self.assertFalse(self.steps.call_args.kwargs["success"])


class HotelToolEmptyResultTests(unittest.TestCase):
    def test_normal_empty_poi_response_is_json_list(self):
        path = PROJECT_ROOT / "tools" / "hotel.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        for node in functions:
            node.decorator_list = []
        response = Mock()
        response.json.return_value = {"status": "1", "pois": []}
        requests = Mock()
        requests.get.return_value = response
        namespace = {"json": json, "AMAP_KEY": "offline-key", "requests": requests}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        self.assertEqual(json.loads(namespace["search_hotels"]("西安")), [])


if __name__ == "__main__":
    unittest.main()
