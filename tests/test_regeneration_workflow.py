"""局部重新生成的离线回归测试。

使用真实 LangGraph 编排、路由、共享状态和人工审核中断；只有业务
Agent 和 Harness 被替换为确定性的模拟实现，不访问模型、API 或数据库。
"""

import ast
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from datetime import date
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agents.supervisor import supervisor_node
from config.revision import build_revision_context, build_revision_update, validate_node_update
from config.trip_policy import requires_hotel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _stub_modules(stubs):
    """只还原替换的模块，不卸载期间正常导入的 LangGraph 依赖。

    patch.dict(sys.modules) 会还原整个映射，新导入的框架模块也会被删除，
    随后的再次导入可能产生不同身份的 GraphInterrupt 等异常类型。
    """
    absent = object()
    originals = {name: sys.modules.get(name, absent) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, original in originals.items():
            if original is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def initial_state():
    """完整的假数据；不依赖用户实际配置或数据库中的任务。"""
    return {
        "session_id": 42,
        "thread_id": "offline-travel-42",
        "trigger_reason": "initial",
        "agent_run_id": None,
        "user_request": "两人到西安游玩两天，总预算 2000 元",
        "origin": "临沂",
        "destination": "西安",
        "departure_date": "2026-10-09",
        "return_date": "2026-10-10",
        "days": 2,
        "people": 2,
        "budget": 2000.0,
        "preferences": ["历史文化"],
        "constraints": ["不要太累"],
        "attractions_result": "",
        "attraction_names": [],
        "hotel_result": "",
        "hotel_name": "",
        "transport_result": "",
        "planner_result": "",
        "review_result": "",
        "review_passed": False,
        "review_retry_count": 0,
        "max_review_retries": 2,
        "human_review_result": "",
        "human_review_passed": False,
        "human_revision_count": 0,
        "max_human_revisions": 10,
        "next_agent": "",
        "attraction_status": "pending",
        "hotel_status": "pending",
        "transport_status": "pending",
        "attraction_revision_request": "",
        "hotel_revision_request": "",
        "transport_revision_request": "",
        "planner_revision_request": "",
        "attraction_excluded_names": [],
        "hotel_search_keyword": "酒店",
        "hotel_excluded_names": [],
        "city_transport_mode": "transit",
        "intercity_transport_mode": "train",
    }


def completed_state():
    return {
        **initial_state(),
        "attraction_status": "completed",
        "hotel_status": "completed",
        "transport_status": "completed",
        "attraction_names": ["西安城墙", "陕西历史博物馆"],
        "attractions_result": "历史文化景点说明",
        "hotel_name": "原酒店",
        "hotel_result": "原酒店推荐说明",
        "transport_result": "原酒店到景点的路线",
        "planner_result": "旧版本完整行程",
        "review_result": "上一轮审核通过",
        "review_passed": True,
        "review_retry_count": 2,
    }


class RevisionUpdateTests(unittest.TestCase):
    def test_hotel_change_invalidates_only_hotel_and_transport(self):
        state = completed_state()
        original = deepcopy(state)
        update = build_revision_update(
            state,
            {
                "approved": False,
                "comment": " 换成全季，靠近地铁 ",
                "revision_scope": "hotel",
                "hotel_keyword": " 全季 ",
            },
        )
        merged = {**state, **update}

        self.assertEqual(state, original, "构造更新不能直接修改原 State")
        self.assertEqual(merged["attraction_status"], "completed")
        self.assertEqual(merged["attraction_names"], original["attraction_names"])
        self.assertEqual(merged["attractions_result"], original["attractions_result"])
        self.assertEqual(merged["hotel_status"], "dirty")
        self.assertEqual(merged["transport_status"], "dirty")
        self.assertEqual(merged["hotel_name"], "")
        self.assertEqual(merged["hotel_result"], "")
        self.assertEqual(merged["transport_result"], "")
        self.assertEqual(merged["hotel_search_keyword"], "全季")
        self.assertIn("原酒店", merged["hotel_excluded_names"])
        self.assertEqual(merged["hotel_revision_request"], "换成全季，靠近地铁")
        self.assertEqual(merged["planner_result"], original["planner_result"])
        self.assertEqual(merged["review_retry_count"], 0)
        self.assertFalse(merged["review_passed"])
        self.assertEqual(merged["review_result"], "")
        self.assertEqual(merged["session_id"], 42)
        self.assertEqual(merged["thread_id"], state["thread_id"])

    def test_attraction_change_invalidates_all_specialists(self):
        state = completed_state()
        merged = {
            **state,
            **build_revision_update(
                state,
                {
                    "approved": False,
                    "comment": "不要西安城墙",
                    "revision_scope": "attraction",
                    "attraction_excluded_names": ["西安城墙"],
                },
            ),
        }
        for name in ("attraction", "hotel", "transport"):
            self.assertEqual(merged[f"{name}_status"], "dirty")
        self.assertEqual(merged["attraction_names"], [])
        for name in ("attractions_result", "hotel_result", "transport_result", "hotel_name"):
            self.assertEqual(merged[name], "")
        self.assertEqual(merged["attraction_excluded_names"], ["西安城墙"])
        self.assertEqual(merged["attraction_revision_request"], "不要西安城墙")

    def test_transport_change_preserves_hotel_and_omitted_intercity_mode(self):
        state = completed_state()
        state["intercity_transport_mode"] = "flight"
        state["hotel_revision_request"] = "必须入住全季"
        merged = {
            **state,
            **build_revision_update(
                state,
                {
                    "approved": False,
                    "comment": "市内改为自驾",
                    "revision_scope": "transport",
                    "city_transport_mode": "driving",
                },
            ),
        }
        self.assertEqual(merged["attraction_status"], "completed")
        self.assertEqual(merged["hotel_status"], "completed")
        self.assertEqual(merged["hotel_name"], "原酒店")
        self.assertEqual(merged["transport_status"], "dirty")
        self.assertEqual(merged["city_transport_mode"], "driving")
        self.assertEqual(merged["intercity_transport_mode"], "flight")
        self.assertEqual(merged["hotel_revision_request"], "必须入住全季")

    def test_itinerary_change_preserves_all_specialist_results(self):
        state = completed_state()
        merged = {
            **state,
            **build_revision_update(state, {"approved": False, "comment": "每天少排一个景点"}),
        }
        for key in (
            "attraction_status", "hotel_status", "transport_status",
            "attractions_result", "attraction_names", "hotel_result",
            "hotel_name", "transport_result", "planner_result",
        ):
            self.assertEqual(merged[key], state[key])
        self.assertEqual(merged["planner_revision_request"], "每天少排一个景点")
        self.assertEqual(merged["human_revision_count"], 1)

    def test_approval_does_not_invalidate_or_reset_review(self):
        state = completed_state()
        update = build_revision_update(state, {"approved": True, "comment": "同意"})
        merged = {**state, **update}
        self.assertTrue(merged["human_review_passed"])
        self.assertEqual(merged["trigger_reason"], "human_review_pass")
        self.assertEqual(merged["review_retry_count"], 2)
        for key in ("attraction_status", "hotel_status", "transport_status", "hotel_name"):
            self.assertEqual(merged[key], state[key])

    def test_existing_exclusions_and_other_requests_survive_later_edit(self):
        state = completed_state()
        state.update({
            "attraction_excluded_names": ["西安城墙"],
            "hotel_search_keyword": "全季",
            "hotel_revision_request": "必须入住全季",
            "transport_revision_request": "市内自驾",
            "city_transport_mode": "driving",
        })
        merged = {
            **state,
            **build_revision_update(
                state,
                {"approved": False, "comment": "再推荐一个博物馆", "revision_scope": "attraction"},
            ),
        }
        for key in (
            "attraction_excluded_names", "hotel_search_keyword",
            "hotel_revision_request", "transport_revision_request", "city_transport_mode",
        ):
            self.assertEqual(merged[key], state[key])

    def test_explicit_empty_exclusions_clear_previous_exclusions(self):
        state = completed_state()
        state["attraction_excluded_names"] = ["西安城墙"]
        update = build_revision_update(
            state,
            {
                "approved": False,
                "comment": "重新选择，不再排除之前的景点",
                "revision_scope": "attraction",
                "attraction_excluded_names": [],
            },
        )
        self.assertEqual(update["attraction_excluded_names"], [])

    def test_blank_hotel_keyword_uses_generic_search(self):
        update = build_revision_update(
            completed_state(),
            {
                "approved": False,
                "comment": "换一家，不限制品牌",
                "revision_scope": "hotel",
                "hotel_keyword": "   ",
            },
        )
        self.assertEqual(update["hotel_search_keyword"], "酒店")

    def test_invalid_payloads_are_rejected_before_updating_state(self):
        payloads = [
            None,
            [],
            {},
            {"approved": "false", "comment": "更换"},
            {"approved": 0, "comment": "更换"},
            {"approved": False, "comment": "  "},
            {"approved": False, "comment": 123},
            {"approved": False, "comment": "更换", "revision_scope": "unknown"},
            {"approved": False, "comment": "更换", "revision_scope": "hotel", "hotel_keyword": []},
            {"approved": False, "comment": "更换", "revision_scope": "attraction", "attraction_excluded_names": "西安城墙"},
            {"approved": False, "comment": "更换", "revision_scope": "attraction", "attraction_excluded_names": [None]},
            {"approved": False, "comment": "更换", "revision_scope": "transport", "city_transport_mode": "teleport"},
            {"approved": False, "comment": "更换", "revision_scope": "transport", "intercity_transport_mode": "teleport"},
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                state = completed_state()
                original = deepcopy(state)
                with self.assertRaises((TypeError, ValueError)):
                    build_revision_update(state, payload)
                self.assertEqual(state, original)


class SupervisorStatusTests(unittest.TestCase):
    def test_dirty_status_wins_over_nonempty_stale_result(self):
        state = completed_state()
        state["hotel_status"] = "dirty"
        # 故意保留旧文字，证明路由依据状态而不是字符串是否存在。
        with redirect_stdout(io.StringIO()):
            update = supervisor_node(state)
        self.assertEqual(update["next_agent"], "hotel")

    def test_only_dirty_transport_starts_from_transport(self):
        state = completed_state()
        state["transport_status"] = "dirty"
        with redirect_stdout(io.StringIO()):
            update = supervisor_node(state)
        self.assertEqual(update["next_agent"], "transport")

    def test_invalid_completed_attractions_invalidate_all_dependents(self):
        state = completed_state()
        state["attraction_names"] = []
        with redirect_stdout(io.StringIO()):
            update = supervisor_node(state)
        self.assertEqual(update["next_agent"], "attraction")
        for name in ("attraction", "hotel", "transport"):
            self.assertEqual(update[f"{name}_status"], "dirty")
        self.assertEqual(update["attraction_names"], [])
        for field in ("attractions_result", "hotel_result", "hotel_name", "transport_result"):
            self.assertEqual(update[field], "")

    def test_invalid_completed_hotel_keeps_attractions_but_invalidates_transport(self):
        state = completed_state()
        state["hotel_name"] = " "
        with redirect_stdout(io.StringIO()):
            update = supervisor_node(state)
        self.assertEqual(update["next_agent"], "hotel")
        self.assertEqual(update["hotel_status"], "dirty")
        self.assertEqual(update["transport_status"], "dirty")
        self.assertEqual(update["hotel_result"], "")
        self.assertEqual(update["transport_result"], "")
        self.assertNotIn("attraction_status", update)
        self.assertNotIn("attraction_names", update)


class NodeUpdateValidationTests(unittest.TestCase):
    def test_valid_results_mark_specialists_completed(self):
        cases = {
            "attraction": {"attractions_result": "景点说明", "attraction_names": ["陕西历史博物馆"]},
            "hotel": {"hotel_result": "酒店说明", "hotel_name": "全季酒店"},
            "transport": {"transport_result": "有效交通说明"},
        }
        for name, update in cases.items():
            with self.subTest(name=name):
                checked = validate_node_update(name, update)
                self.assertEqual(checked[f"{name}_status"], "completed")

    def test_invalid_or_empty_outputs_are_rejected(self):
        cases = [
            ("attraction", {"attractions_result": "说明", "attraction_names": []}),
            ("attraction", {"attractions_result": "说明", "attraction_names": [" "]}),
            ("attraction", {"attractions_result": " ", "attraction_names": ["景点"]}),
            ("hotel", {"hotel_result": "说明", "hotel_name": " "}),
            ("hotel", {"hotel_result": " ", "hotel_name": "酒店"}),
            ("transport", {"transport_result": " "}),
            ("planner", {"planner_result": " "}),
            ("reviewer", {"review_passed": "true", "review_result": "通过"}),
            ("reviewer", {"review_passed": True, "review_result": " "}),
            ("hotel", None),
        ]
        for name, update in cases:
            with self.subTest(name=name, update=update), self.assertRaises((TypeError, ValueError)):
                validate_node_update(name, update)


class PlannerRevisionInstructionTests(unittest.TestCase):
    def test_ai_rejection_keeps_all_user_requests_and_adds_ai_feedback(self):
        # 只编译实际纯函数，不执行 planner.py 中的模型和数据库导入。
        path = PROJECT_ROOT / "agents" / "planner.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_build_revision_instruction"
        ]
        self.assertEqual(len(functions), 1)
        namespace = {
            "TravelState": dict, "build_revision_context": build_revision_context,
            "requires_hotel": requires_hotel,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)

        state = completed_state()
        requests = {
            "attraction_revision_request": "不要西安城墙，增加博物馆",
            "hotel_revision_request": "必须入住全季并靠近地铁",
            "transport_revision_request": "市内自驾，去程和返程坐飞机",
            "planner_revision_request": "每天保留两个小时休息",
        }
        state.update(requests)
        state.update({
            "trigger_reason": "ai_review_reject",
            "review_result": "返程当天需要提前到机场",
            "attraction_excluded_names": ["西安城墙"],
            "hotel_search_keyword": "全季",
            "city_transport_mode": "driving",
            "intercity_transport_mode": "flight",
        })
        instruction = namespace["_build_revision_instruction"](state)
        for request in requests.values():
            self.assertIn(request, instruction)
        self.assertIn(state["review_result"], instruction)
        self.assertIn("driving", instruction)
        self.assertIn("flight", instruction)
        self.assertIn("排除景点", instruction)


class WorkflowFixture:
    """读取实际图定义，仅将有外部副作用的节点和 Harness 替换掉。"""

    def __init__(self):
        self.calls = []
        self.inputs = []
        self.hotel_count = 0
        self.remaining_review_rejections = 0
        self.fail_hotel_when_dirty = False

        def attraction(state):
            excluded = set(state.get("attraction_excluded_names", []))
            names = [name for name in ["西安城墙", "陕西历史博物馆", "大雁塔"] if name not in excluded]
            return {"attraction_names": names, "attractions_result": "景点：" + "、".join(names)}

        def hotel(state):
            if self.fail_hotel_when_dirty and state.get("hotel_status") == "dirty":
                self.fail_hotel_when_dirty = False
                raise ValueError("模拟新酒店查询无候选")
            self.hotel_count += 1
            keyword = state.get("hotel_search_keyword", "酒店")
            name = f"{keyword}候选酒店{self.hotel_count}"
            return {"hotel_name": name, "hotel_result": "酒店推荐：" + name}

        def transport(state):
            return {"transport_result": (
                f"从{state['hotel_name']}出发，市内{state['city_transport_mode']}，"
                f"城际{state['intercity_transport_mode']}"
            )}

        def planner(state):
            return {"planner_result": (
                f"行程：{state['attractions_result']}；{state['hotel_result']}；"
                f"{state['transport_result']}；修改：{state.get('planner_revision_request', '')}"
            )}

        def reviewer(state):
            if self.remaining_review_rejections:
                self.remaining_review_rejections -= 1
                return {
                    "review_passed": False,
                    "review_result": "模拟审核发现问题",
                    "review_retry_count": state["review_retry_count"] + 1,
                    "trigger_reason": "ai_review_reject",
                }
            return {
                "review_passed": True,
                "review_result": "模拟审核通过",
                "trigger_reason": "ai_review_pass",
            }

        def harness(*, agent_name, agent_func, state, max_retries):
            self.calls.append(agent_name)
            self.inputs.append((agent_name, deepcopy(state)))
            return agent_func(dict(state))

        stubs = {}
        for name, func in {
            "attraction": attraction, "hotel": hotel, "transport": transport,
            "planner": planner, "reviewer": reviewer,
        }.items():
            module = ModuleType(f"agents.{name}")
            setattr(module, f"{name}_node", func)
            stubs[module.__name__] = module
        module = ModuleType("harness.runtime")
        module.run_with_harness = harness
        stubs[module.__name__] = module

        spec = importlib.util.spec_from_file_location(
            "_offline_regeneration_graph", PROJECT_ROOT / "config" / "graph.py"
        )
        graph_module = importlib.util.module_from_spec(spec)
        # 仅还原这几个模拟模块，不会将假的 Agent 留给其他测试。
        with _stub_modules(stubs):
            spec.loader.exec_module(graph_module)
        self.graph = graph_module.builder.compile(checkpointer=InMemorySaver())
        self.config = {"configurable": {"thread_id": initial_state()["thread_id"]}}

    def invoke(self, value):
        with redirect_stdout(io.StringIO()):
            return self.graph.invoke(value, config=self.config)

    def generate(self):
        return self.invoke(initial_state())

    def resume(self, **decision):
        return self.invoke(Command(resume=decision))

    def clear_calls(self):
        self.calls.clear()
        self.inputs.clear()


class RegenerationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture = WorkflowFixture()

    def assert_paused_for_human(self, result):
        self.assertTrue(result.get("__interrupt__"))
        self.assertEqual(self.fixture.graph.get_state(self.fixture.config).next, ("human_review",))

    def test_first_generation_runs_specialists_in_dependency_order(self):
        result = self.fixture.generate()
        self.assertEqual(self.fixture.calls, ["attraction", "hotel", "transport", "planner", "reviewer"])
        self.assert_paused_for_human(result)
        for name in ("attraction", "hotel", "transport"):
            self.assertEqual(result[f"{name}_status"], "completed")
        inputs = dict(self.fixture.inputs)
        self.assertEqual(inputs["hotel"]["attraction_names"], result["attraction_names"])
        self.assertEqual(inputs["transport"]["hotel_name"], result["hotel_name"])

    def test_one_day_workflow_skips_hotel_and_still_reaches_human_review(self):
        state = initial_state()
        state.update({"days": 1, "return_date": state["departure_date"]})
        result = self.fixture.invoke(state)
        self.assertEqual(self.fixture.calls, ["attraction", "transport", "planner", "reviewer"])
        self.assertEqual(result["hotel_status"], "skipped")
        self.assertEqual(result["hotel_name"], "")
        self.assert_paused_for_human(result)
        self.fixture.clear_calls()
        changed = self.fixture.resume(
            approved=False, comment="不要西安城墙", revision_scope="attraction",
            attraction_excluded_names=["西安城墙"],
        )
        self.assertEqual(self.fixture.calls, ["attraction", "transport", "planner", "reviewer"])
        self.assertEqual(changed["hotel_status"], "skipped")

    def test_hotel_change_reuses_attractions_and_updates_transport(self):
        original = self.fixture.generate()
        self.fixture.clear_calls()
        result = self.fixture.resume(
            approved=False, comment="换成全季", revision_scope="hotel", hotel_keyword="全季"
        )
        self.assertEqual(self.fixture.calls, ["hotel", "transport", "planner", "reviewer"])
        self.assertEqual(result["attraction_names"], original["attraction_names"])
        self.assertNotEqual(result["hotel_name"], original["hotel_name"])
        self.assertIn(original["hotel_name"], result["hotel_excluded_names"])
        self.assertIn(result["hotel_name"], result["transport_result"])
        self.assertEqual(result["session_id"], original["session_id"])
        self.assertEqual(result["thread_id"], original["thread_id"])
        self.assert_paused_for_human(result)

    def test_attraction_change_reruns_downstream_nodes(self):
        self.fixture.generate()
        self.fixture.clear_calls()
        result = self.fixture.resume(
            approved=False,
            comment="不要西安城墙",
            revision_scope="attraction",
            attraction_excluded_names=["西安城墙"],
        )
        self.assertEqual(self.fixture.calls, ["attraction", "hotel", "transport", "planner", "reviewer"])
        self.assertNotIn("西安城墙", result["attraction_names"])
        self.assert_paused_for_human(result)

    def test_transport_change_does_not_rerun_hotel_or_attractions(self):
        original = self.fixture.generate()
        self.fixture.clear_calls()
        result = self.fixture.resume(
            approved=False,
            comment="市内自驾，城际飞机",
            revision_scope="transport",
            city_transport_mode="driving",
            intercity_transport_mode="flight",
        )
        self.assertEqual(self.fixture.calls, ["transport", "planner", "reviewer"])
        self.assertEqual(result["hotel_name"], original["hotel_name"])
        self.assertEqual(result["attraction_names"], original["attraction_names"])
        self.assertIn("driving", result["transport_result"])
        self.assertIn("flight", result["transport_result"])
        self.assert_paused_for_human(result)

    def test_itinerary_change_reruns_only_planner_and_reviewer(self):
        original = self.fixture.generate()
        self.fixture.clear_calls()
        result = self.fixture.resume(approved=False, comment="每天少安排一些", revision_scope="planner")
        self.assertEqual(self.fixture.calls, ["planner", "reviewer"])
        for key in ("attractions_result", "hotel_result", "transport_result"):
            self.assertEqual(result[key], original[key])
        self.assertEqual(result["planner_revision_request"], "每天少安排一些")
        self.assert_paused_for_human(result)

    def test_approval_finishes_without_rerunning_any_business_node(self):
        original = self.fixture.generate()
        self.fixture.clear_calls()
        result = self.fixture.resume(approved=True, comment="确认通过")
        self.assertEqual(self.fixture.calls, [])
        self.assertTrue(result["human_review_passed"])
        self.assertEqual(result["planner_result"], original["planner_result"])
        self.assertEqual(self.fixture.graph.get_state(self.fixture.config).next, ())

    def test_successive_edits_keep_earlier_constraints(self):
        self.fixture.generate()
        self.fixture.resume(
            approved=False, comment="必须入住全季", revision_scope="hotel", hotel_keyword="全季"
        )
        self.fixture.resume(
            approved=False, comment="市内必须自驾", revision_scope="transport", city_transport_mode="driving"
        )
        self.fixture.clear_calls()
        result = self.fixture.resume(
            approved=False,
            comment="不要西安城墙",
            revision_scope="attraction",
            attraction_excluded_names=["西安城墙"],
        )
        self.assertEqual(result["hotel_search_keyword"], "全季")
        self.assertEqual(result["hotel_revision_request"], "必须入住全季")
        self.assertEqual(result["transport_revision_request"], "市内必须自驾")
        self.assertEqual(result["city_transport_mode"], "driving")
        inputs = dict(self.fixture.inputs)
        self.assertEqual(inputs["hotel"]["hotel_revision_request"], "必须入住全季")
        self.assertEqual(inputs["transport"]["city_transport_mode"], "driving")
        self.assertEqual(result["human_revision_count"], 3)

    def test_ai_rejection_loops_only_planner_and_new_human_edit_resets_count(self):
        self.fixture.remaining_review_rejections = 2
        first = self.fixture.generate()
        self.assertFalse(first["review_passed"])
        self.assertEqual(first["review_retry_count"], 2)
        self.assertEqual(self.fixture.calls, [
            "attraction", "hotel", "transport", "planner", "reviewer", "planner", "reviewer",
        ])
        self.assert_paused_for_human(first)

        self.fixture.clear_calls()
        self.fixture.remaining_review_rejections = 1
        second = self.fixture.resume(approved=False, comment="换一家酒店", revision_scope="hotel")
        self.assertEqual(self.fixture.calls, ["hotel", "transport", "planner", "reviewer", "planner", "reviewer"])
        self.assertTrue(second["review_passed"])
        self.assertEqual(second["review_retry_count"], 1)
        reviewer_inputs = [state for name, state in self.fixture.inputs if name == "reviewer"]
        self.assertEqual(reviewer_inputs[0]["review_retry_count"], 0)
        planner_inputs = [state for name, state in self.fixture.inputs if name == "planner"]
        self.assertEqual(planner_inputs[-1]["hotel_revision_request"], "换一家酒店")

    def test_cancel_failed_regeneration_restores_real_human_interrupt_checkpoint(self):
        old = deepcopy(self.fixture.generate())
        self.fixture.fail_hotel_when_dirty = True
        with self.assertRaisesRegex(ValueError, "无候选"):
            self.fixture.resume(
                approved=False,
                comment="换成不存在的品牌",
                revision_scope="hotel",
                hotel_keyword="不存在的品牌",
            )
        failed = self.fixture.graph.get_state(self.fixture.config)
        self.assertEqual(failed.next, ("hotel",))
        self.assertEqual(failed.values["hotel_status"], "dirty")

        # 与页面取消操作完全一致：以上次已审核到人工节点的结果替换当前状态。
        self.fixture.graph.update_state(self.fixture.config, old, as_node="reviewer")
        self.fixture.clear_calls()
        restored = self.fixture.invoke(None)
        self.assertEqual(self.fixture.calls, [], "取消不得重跑业务节点或自动批准")
        self.assert_paused_for_human(restored)
        for key in (
            "planner_result", "hotel_name", "hotel_result", "transport_result",
            "hotel_status", "transport_status", "session_id", "thread_id",
            "hotel_search_keyword", "human_revision_count",
        ):
            self.assertEqual(restored[key], old[key])
        self.assertFalse(restored["human_review_passed"])

        regenerated = self.fixture.resume(
            approved=False, comment="改为全季", revision_scope="hotel", hotel_keyword="全季"
        )
        self.assertEqual(regenerated["hotel_search_keyword"], "全季")
        self.assert_paused_for_human(regenerated)


def _load_app_helpers(graph, thread_id="existing-travel-task"):
    """不导入 app.py，避免执行页面和数据库模块的顶层代码。"""
    from config.memory import capture_initial_request
    path = PROJECT_ROOT / "app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"_resume_graph", "_graph_config", "_split_text", "_build_initial_state"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "Command": Command,
        "graph": graph,
        "capture_initial_request": capture_initial_request,
        "st": SimpleNamespace(session_state=SimpleNamespace(thread_id=thread_id)),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class AppWorkflowHelperTests(unittest.TestCase):
    def test_resume_passes_only_provided_parameters_and_keeps_thread_id(self):
        graph = Mock()
        graph.invoke.return_value = {"ok": True}
        namespace = _load_app_helpers(graph)
        result = namespace["_resume_graph"](
            approved=False,
            comment="市内改自驾",
            revision_scope="transport",
            city_transport_mode="driving",
        )
        self.assertEqual(result, {"ok": True})
        args, kwargs = graph.invoke.call_args
        self.assertIsInstance(args[0], Command)
        self.assertEqual(args[0].resume["revision_scope"], "transport")
        self.assertEqual(args[0].resume["city_transport_mode"], "driving")
        self.assertNotIn("intercity_transport_mode", args[0].resume)
        self.assertNotIn("hotel_keyword", args[0].resume)
        self.assertNotIn("attraction_excluded_names", args[0].resume)
        self.assertEqual(kwargs["config"]["configurable"]["thread_id"], "existing-travel-task")

    def test_original_approval_call_remains_compatible(self):
        graph = Mock()
        namespace = _load_app_helpers(graph)
        namespace["_resume_graph"](approved=True, comment="同意")
        args, _ = graph.invoke.call_args
        self.assertTrue(args[0].resume["approved"])

    def test_new_task_initializes_all_revision_fields(self):
        namespace = _load_app_helpers(Mock())
        state = namespace["_build_initial_state"](
            "临沂", "西安", date(2026, 10, 9), date(2026, 10, 10),
            2, 2, 2000, "历史文化", "不要太累",
        )
        for name in ("attraction", "hotel", "transport"):
            self.assertEqual(state[f"{name}_status"], "pending")
            self.assertEqual(state[f"{name}_revision_request"], "")
        self.assertEqual(state["planner_revision_request"], "")
        self.assertEqual(state["attraction_excluded_names"], [])
        self.assertEqual(state["hotel_excluded_names"], [])
        self.assertEqual(state["hotel_search_keyword"], "酒店")
        self.assertEqual(state["city_transport_mode"], "transit")
        self.assertEqual(state["intercity_transport_mode"], "train")


if __name__ == "__main__":
    unittest.main()
