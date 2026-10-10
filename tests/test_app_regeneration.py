"""使用真实 Streamlit 页面组件的离线回归测试，不读取密钥或连接外部服务。"""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

from langgraph.types import Command
from streamlit.testing.v1 import AppTest

from config.revision import build_revision_update


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@contextmanager
def _stub_modules(stubs):
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


class OfflineGraph:
    def __init__(self):
        self.state = None
        self.calls = []
        self.update_calls = []
        self.fail_next = False
        self.hotel_number = 0

    def update_state(self, config, values, *, as_node):
        self.update_calls.append((deepcopy(config), deepcopy(values), as_node))
        self.state = {key: deepcopy(value) for key, value in values.items() if key != "__interrupt__"}
        return config

    def invoke(self, value, *, config):
        self.calls.append((deepcopy(value), deepcopy(config)))
        if isinstance(value, dict):
            self.state = deepcopy(value)
        elif isinstance(value, Command):
            self.state.update(build_revision_update(self.state, value.resume))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("离线模拟节点失败")

        if not self.state["attraction_names"]:
            excluded = self.state.get("attraction_excluded_names", [])
            self.state["attraction_names"] = [
                name for name in ["西安城墙", "陕西历史博物馆", "大雁塔"] if name not in excluded
            ]
            self.state["attractions_result"] = "景点：" + "、".join(self.state["attraction_names"])
            self.state["attraction_status"] = "completed"
        if not self.state["hotel_name"]:
            self.hotel_number += 1
            self.state["hotel_name"] = f"{self.state['hotel_search_keyword']}酒店{self.hotel_number}"
            self.state["hotel_result"] = "住宿：" + self.state["hotel_name"]
            self.state["hotel_status"] = "completed"
        self.state["transport_result"] = (
            f"{self.state['hotel_name']}："
            f"{self.state['city_transport_mode']} / {self.state['intercity_transport_mode']}"
        )
        self.state["transport_status"] = "completed"
        self.state["planner_result"] = "离线完整行程：" + self.state["transport_result"]
        self.state["review_passed"] = True
        self.state["review_result"] = "离线审核通过"
        result = deepcopy(self.state)
        if not (isinstance(value, Command) and value.resume["approved"]):
            result["__interrupt__"] = [True]
        return result


class OfflineRepository:
    def __init__(self):
        self.sessions = []
        self.plans = []
        self.reviews = []
        self.finalized = []
        self.completed = []
        self.deleted = []
        self.fail_next_plan_save = False

    def create_travel_session(self, *, thread_id, state):
        self.sessions.append({"thread_id": thread_id, "state": deepcopy(state)})
        return 100 + len(self.sessions)

    def create_plan_with_review(self, *, session_id, state, version):
        if self.fail_next_plan_save:
            self.fail_next_plan_save = False
            raise RuntimeError("离线模拟方案保存失败")
        plan_id = 200 + len(self.plans)
        self.plans.append({
            "session_id": session_id, "version": version, "plan_id": plan_id,
            "state": deepcopy(state), "is_final": False,
        })
        return plan_id

    def create_human_review(self, **kwargs):
        self.reviews.append(deepcopy(kwargs))

    def get_plan_versions(self, session_id):
        return [
            {
                "version": item["version"], "plan_id": item["plan_id"],
                "is_final": item["is_final"],
                "planner_result": item["state"]["planner_result"],
                "review_result": item["state"]["review_result"],
                "review_passed": item["state"]["review_passed"],
                "human_comment": "",
            }
            for item in self.plans if item["session_id"] == session_id
        ]

    def get_travel_sessions(self, search=""):
        result = []
        for index, item in reversed(list(enumerate(self.sessions, start=101))):
            if not any(plan["session_id"] == index for plan in self.plans):
                continue
            state = item["state"]
            if search.strip() and not any(search.strip() in state[field] for field in (
                "origin", "destination", "user_request"
            )):
                continue
            result.append({
                "session_id": index,
                "origin": state["origin"], "destination": state["destination"],
                "departure_date": state["departure_date"], "return_date": state["return_date"],
                "status": "completed" if index in self.completed else "planning",
            })
        return result

    def mark_plan_final(self, plan_id):
        self.finalized.append(plan_id)
        selected = next(item for item in self.plans if item["plan_id"] == plan_id)
        for item in self.plans:
            if item["session_id"] == selected["session_id"]:
                item["is_final"] = item["plan_id"] == plan_id
        return True

    def complete_travel_session(self, session_id):
        self.completed.append(session_id)

    def delete_travel_plan(self, plan_id):
        self.deleted.append(plan_id)
        self.plans = [item for item in self.plans if item["plan_id"] != plan_id]
        return True


class StreamlitRegenerationTests(unittest.TestCase):
    def setUp(self):
        self.graph = OfflineGraph()
        self.repository = OfflineRepository()
        graph_module = ModuleType("config.graph")
        graph_module.builder = SimpleNamespace(compile=lambda **kwargs: self.graph)
        repository_module = ModuleType("database.repository")
        for name in (
            "complete_travel_session", "create_human_review", "create_plan_with_review",
            "create_travel_session", "delete_travel_plan", "get_plan_versions", "get_travel_sessions", "mark_plan_final",
        ):
            setattr(repository_module, name, getattr(self.repository, name))
        context = _stub_modules({
            "config.graph": graph_module,
            "database.repository": repository_module,
        })
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.app = AppTest.from_file(str(APP_PATH), default_timeout=5).run()
        self.assert_no_exception()

    def assert_no_exception(self):
        self.assertEqual(len(self.app.exception), 0, str(self.app.exception))

    def by_label(self, element_type, label):
        return next(element for element in getattr(self.app, element_type) if element.label == label)

    def generate(self):
        self.by_label("button", "🚀 生成旅游方案").click().run()
        self.assert_no_exception()

    def open_feedback(self, scope_label):
        self.app.button(key="reject_plan").click().run()
        self.by_label("selectbox", "修改类型").select(scope_label).run()
        self.assert_no_exception()

    def submit(self, comment):
        self.by_label("text_area", "请输入修改意见").set_value(comment).run()
        self.app.button(key="submit_feedback").click().run()
        self.assert_no_exception()

    def assert_no_final_confirmation(self):
        self.assertFalse(any("当前旅游方案已最终确认" in element.value for element in self.app.success))

    def assert_failed_state(self):
        self.assertTrue(self.app.session_state["generation_failed"])
        self.assertTrue(self.app.session_state["last_generation_error"])
        self.assertTrue(any("失败" in element.value for element in self.app.error))
        self.assert_no_final_confirmation()
        self.assertFalse(any(element.key == "approve_plan" for element in self.app.button))
        self.assertFalse(any(element.key == "reject_plan" for element in self.app.button))
        self.assertTrue(self.app.button(key="retry_failed_generation"))

    def test_initial_page_and_first_generation(self):
        self.assertEqual(len(self.repository.sessions), 0)
        self.assertEqual(len(self.graph.calls), 0)
        self.generate()
        self.assertEqual(len(self.repository.sessions), 1)
        self.assertEqual(self.app.session_state["version"], 1)
        self.assertEqual(self.repository.plans[0]["version"], 1)
        self.assertTrue(self.app.session_state["running"])
        self.assertFalse(self.app.session_state["generation_failed"])
        self.assert_no_final_confirmation()
        self.assertTrue(self.app.button(key="approve_plan"))
        # 活跃任务不允许从侧栏删除或切换最终版。
        self.assertTrue(self.app.button(key="set_final_200").disabled)
        self.assertTrue(self.app.button(key="delete_plan_200").disabled)

    def test_fresh_page_reads_saved_history_without_resuming_graph(self):
        self.generate()
        call_count = len(self.graph.calls)
        self.app = AppTest.from_file(str(APP_PATH), default_timeout=5).run()
        self.assert_no_exception()
        self.assertIsNone(self.app.session_state["session_id"])
        self.assertIsNone(self.app.session_state["result"])
        self.assertEqual(self.app.selectbox(key="history_session_id").value, 101)
        self.assertTrue(any("离线完整行程" in item.value for item in self.app.markdown))
        self.assertEqual(len(self.graph.calls), call_count)
        self.assertFalse(any(item.key == "approve_plan" for item in self.app.button))

    def test_history_search_and_clear_work_in_fresh_session(self):
        self.generate()
        self.app = AppTest.from_file(str(APP_PATH), default_timeout=5).run()
        self.by_label("text_input", "搜索历史任务").set_value("不存在的目的地").run()
        self.assert_no_exception()
        self.assertFalse(any(item.key == "history_session_id" for item in self.app.selectbox))
        self.by_label("text_input", "搜索历史任务").set_value("西安").run()
        self.assert_no_exception()
        self.assertEqual(self.app.selectbox(key="history_session_id").value, 101)

    def test_fresh_page_can_delete_saved_version(self):
        self.generate()
        self.app = AppTest.from_file(str(APP_PATH), default_timeout=5).run()
        call_count = len(self.graph.calls)
        self.assertFalse(self.app.button(key="delete_plan_200").disabled)
        self.app.button(key="delete_plan_200").click().run()
        self.assert_no_exception()
        self.assertEqual(self.repository.deleted, [200])
        self.assertEqual(self.repository.plans, [])
        self.assertTrue(any("暂无已保存的历史方案" in item.value for item in self.app.caption))
        self.assertEqual(len(self.graph.calls), call_count)

    def test_managing_other_history_does_not_switch_active_task(self):
        self.generate()
        self.generate()
        current_session = self.app.session_state["session_id"]
        current_thread = self.app.session_state["thread_id"]
        current_plan = self.app.session_state["current_plan_id"]
        call_count = len(self.graph.calls)
        self.app.selectbox(key="history_session_id").set_value(101).run()
        self.assert_no_exception()
        self.assertFalse(self.app.button(key="set_final_200").disabled)
        self.app.button(key="set_final_200").click().run()
        self.assert_no_exception()
        self.assertEqual(self.app.session_state["current_plan_id"], current_plan)
        self.app.button(key="delete_plan_200").click().run()
        self.assert_no_exception()
        self.assertEqual(self.app.session_state["session_id"], current_session)
        self.assertEqual(self.app.session_state["thread_id"], current_thread)
        self.assertEqual(self.app.session_state["current_plan_id"], current_plan)
        self.assertEqual(len(self.graph.calls), call_count)
        self.assertTrue(self.app.button(key=f"delete_plan_{current_plan}").disabled)

    def test_hotel_widgets_submit_keyword_keep_task_and_add_version(self):
        self.generate()
        original_thread = self.app.session_state["thread_id"]
        original_session = self.app.session_state["session_id"]
        old_hotel = self.app.session_state["result"]["hotel_name"]
        self.open_feedback("更换酒店")
        self.by_label("text_input", "酒店品牌或名称关键词，可留空").set_value("全季").run()
        self.submit("换成全季，靠近地铁")

        command, config = self.graph.calls[-1]
        self.assertIsInstance(command, Command)
        self.assertEqual(command.resume["revision_scope"], "hotel")
        self.assertEqual(command.resume["hotel_keyword"], "全季")
        self.assertNotIn("city_transport_mode", command.resume)
        result = self.app.session_state["result"]
        self.assertIn(old_hotel, result["hotel_excluded_names"])
        self.assertNotEqual(result["hotel_name"], old_hotel)
        self.assertEqual(self.app.session_state["thread_id"], original_thread)
        self.assertEqual(config["configurable"]["thread_id"], original_thread)
        self.assertEqual(self.app.session_state["session_id"], original_session)
        self.assertEqual(len(self.repository.sessions), 1)
        self.assertEqual([item["version"] for item in self.repository.plans], [1, 2])
        self.assertEqual(self.app.session_state["version"], 2)
        self.assertEqual(len(self.repository.reviews), 1)

    def test_attraction_multiselect_passes_exclusions(self):
        self.generate()
        self.open_feedback("更换景点")
        self.by_label(
            "multiselect", "不再安排的景点，可不选并在修改意见中描述新偏好"
        ).set_value(["西安城墙"]).run()
        self.submit("不要西安城墙")
        command, _ = self.graph.calls[-1]
        self.assertEqual(command.resume["revision_scope"], "attraction")
        self.assertEqual(command.resume["attraction_excluded_names"], ["西安城墙"])
        self.assertNotIn("西安城墙", self.app.session_state["result"]["attraction_names"])

    def test_transport_selectboxes_pass_actual_mode_values(self):
        self.generate()
        self.open_feedback("更换交通方式")
        self.by_label("selectbox", "市内交通方式").select("自驾").run()
        self.by_label("selectbox", "城际交通方式（去程和返程）").select("飞机").run()
        self.submit("市内自驾，来回飞机")
        command, _ = self.graph.calls[-1]
        self.assertEqual(command.resume["revision_scope"], "transport")
        self.assertEqual(command.resume["city_transport_mode"], "driving")
        self.assertEqual(command.resume["intercity_transport_mode"], "flight")
        self.assertNotIn("hotel_keyword", command.resume)

    def test_itinerary_default_does_not_pass_specialist_parameters(self):
        self.generate()
        self.open_feedback("仅调整行程")
        self.submit("每天少安排一些")
        command, _ = self.graph.calls[-1]
        self.assertEqual(command.resume["revision_scope"], "planner")
        for key in ("hotel_keyword", "attraction_excluded_names", "city_transport_mode", "intercity_transport_mode"):
            self.assertNotIn(key, command.resume)

    def test_blank_feedback_does_not_resume_or_write_audit(self):
        self.generate()
        self.open_feedback("更换酒店")
        original_calls = len(self.graph.calls)
        self.submit("   ")
        self.assertEqual(len(self.graph.calls), original_calls)
        self.assertEqual(len(self.repository.reviews), 0)
        self.assertEqual(self.app.session_state["version"], 1)
        self.assertTrue(any("请输入修改意见" in element.value for element in self.app.warning))

    def test_approval_finalizes_existing_version_without_new_task(self):
        self.generate()
        plan_id = self.app.session_state["current_plan_id"]
        session_id = self.app.session_state["session_id"]
        self.app.button(key="approve_plan").click().run()
        self.assert_no_exception()
        self.assertEqual(self.repository.finalized, [plan_id])
        self.assertEqual(self.repository.completed, [session_id])
        self.assertEqual(len(self.repository.plans), 1)
        self.assertEqual(len(self.repository.sessions), 1)
        self.assertFalse(self.app.session_state["running"])
        self.assertTrue(any("当前旅游方案已最终确认" in element.value for element in self.app.success))

    def test_first_generation_failure_retry_reuses_session_and_checkpoint(self):
        self.graph.fail_next = True
        self.generate()
        original_thread = self.app.session_state["thread_id"]
        original_session = self.app.session_state["session_id"]
        self.assert_failed_state()
        self.assertEqual(self.app.session_state["version"], 0)
        self.assertIsNone(self.app.session_state["current_plan_id"])
        self.assertEqual(len(self.repository.plans), 0)

        self.app.button(key="retry_failed_generation").click().run()
        self.assert_no_exception()
        value, config = self.graph.calls[-1]
        self.assertIsNone(value)
        self.assertEqual(config["configurable"]["thread_id"], original_thread)
        self.assertEqual(self.app.session_state["session_id"], original_session)
        self.assertEqual(len(self.repository.sessions), 1)
        self.assertEqual(self.app.session_state["version"], 1)
        self.assertFalse(self.app.session_state["generation_failed"])

    def test_feedback_failure_disables_old_review_and_retries_same_task(self):
        self.generate()
        original_thread = self.app.session_state["thread_id"]
        original_session = self.app.session_state["session_id"]
        original_result = deepcopy(self.app.session_state["result"])
        self.open_feedback("更换酒店")
        self.graph.fail_next = True
        self.submit("换一家酒店")
        self.assert_failed_state()
        self.assertEqual(self.app.session_state["version"], 1)
        self.assertEqual(self.app.session_state["result"], original_result)
        self.assertTrue(self.app.button(key="delete_plan_200").disabled)
        self.assertTrue(self.app.button(key="set_final_200").disabled)

        self.app.button(key="retry_failed_generation").click().run()
        self.assert_no_exception()
        value, config = self.graph.calls[-1]
        self.assertIsNone(value)
        self.assertEqual(config["configurable"]["thread_id"], original_thread)
        self.assertEqual(self.app.session_state["session_id"], original_session)
        self.assertEqual(len(self.repository.sessions), 1)
        self.assertEqual(len(self.repository.reviews), 1, "失败重试不应重复提交审核意见")
        self.assertEqual(self.app.session_state["version"], 2)
        self.assertFalse(self.app.session_state["generation_failed"])
        self.assertNotEqual(self.app.session_state["result"]["hotel_name"], original_result["hotel_name"])

    def test_plan_save_failure_does_not_consume_version_and_retry_saves_once(self):
        self.repository.fail_next_plan_save = True
        self.generate()
        self.assert_failed_state()
        self.assertEqual(self.app.session_state["version"], 0)
        self.assertEqual(len(self.repository.plans), 0)
        self.app.button(key="retry_failed_generation").click().run()
        self.assert_no_exception()
        self.assertEqual(self.app.session_state["version"], 1)
        self.assertEqual(len(self.repository.plans), 1)
        self.assertEqual(len(self.repository.sessions), 1)

    def test_cancel_failed_hotel_change_restores_old_review_and_allows_new_request(self):
        self.generate()
        old_result = deepcopy(self.app.session_state["result"])
        old_thread = self.app.session_state["thread_id"]
        old_session = self.app.session_state["session_id"]
        old_plan_id = self.app.session_state["current_plan_id"]
        self.open_feedback("更换酒店")
        self.by_label("text_input", "酒店品牌或名称关键词，可留空").set_value("不存在的品牌").run()
        self.graph.fail_next = True
        self.submit("换成这个品牌")
        self.assert_failed_state()
        # 失败消息应能跨普通页面重运行保留。
        self.app.run()
        self.assert_failed_state()

        self.app.button(key="cancel_failed_regeneration").click().run()
        self.assert_no_exception()
        self.assertEqual(len(self.graph.update_calls), 1)
        config, restored_values, as_node = self.graph.update_calls[0]
        self.assertEqual(config["configurable"]["thread_id"], old_thread)
        self.assertEqual(restored_values, old_result)
        self.assertEqual(as_node, "reviewer")
        self.assertIsNone(self.graph.calls[-1][0])
        self.assertEqual(self.app.session_state["result"], old_result)
        self.assertEqual(self.app.session_state["thread_id"], old_thread)
        self.assertEqual(self.app.session_state["session_id"], old_session)
        self.assertEqual(self.app.session_state["current_plan_id"], old_plan_id)
        self.assertEqual(self.app.session_state["version"], 1)
        self.assertEqual(len(self.repository.sessions), 1)
        self.assertEqual(len(self.repository.plans), 1)
        self.assertFalse(self.app.session_state["generation_failed"])
        self.assertFalse(self.app.session_state["pending_plan_save"])
        self.assertEqual(self.app.session_state["last_generation_error"], "")
        self.assertTrue(self.app.session_state["running"])
        self.assert_no_final_confirmation()
        self.assertEqual(self.repository.finalized, [])
        self.assertEqual(self.repository.completed, [])
        self.assertTrue(self.app.button(key="approve_plan"))

        # 重新调整条件，而不是继续使用失败请求中的关键词。
        self.open_feedback("更换酒店")
        self.by_label("text_input", "酒店品牌或名称关键词，可留空").set_value("全季").run()
        self.submit("改为全季")
        command, _ = self.graph.calls[-1]
        self.assertEqual(command.resume["hotel_keyword"], "全季")
        self.assertEqual(self.app.session_state["version"], 2)
        self.assertEqual(self.app.session_state["session_id"], old_session)
        self.assertEqual(len(self.repository.sessions), 1)


if __name__ == "__main__":
    unittest.main()
