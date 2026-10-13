"""Offline policy, ORM persistence, cache eligibility and deletion regressions."""

import ast
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from sqlalchemy import create_engine, func, or_
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.schema import CreateTable

from config.memory import (
    build_memory_payload, build_memory_record, capture_initial_request,
    confirmation_signature, memory_signature, request_state_from_payload,
)
from config.revision import build_revision_update, validate_node_update
from config.trip_policy import requires_hotel, walking_policy


ROOT = Path(__file__).resolve().parents[1]


def request_state():
    return {
        "origin": "临沂", "destination": "西安",
        "departure_date": "2026-10-20", "return_date": "2026-10-21", "days": 2,
        "people": 2, "budget": 2000.0, "preferences": ["历史文化", "美食"],
        "constraints": ["不要太累"], "user_request": "离线长期记忆测试",
        "attraction_names": ["大雁塔"], "hotel_name": "真实候选酒店",
        "attractions_result": "景点证据", "hotel_result": "酒店证据",
        "transport_result": "交通证据", "planner_result": "已保存的完整方案",
        "review_passed": True, "review_result": "测试审核通过",
        "transport_segments": [],
    }


class TripPolicyTests(unittest.TestCase):
    def test_one_day_never_requires_hotel_and_two_days_do(self):
        self.assertFalse(requires_hotel({"days": 1, "needs_hotel": True}))
        self.assertTrue(requires_hotel({"days": 2}))
        self.assertFalse(requires_hotel({"departure_date": "2026-10-20", "return_date": "2026-10-20"}))

    def test_hotel_validation_uses_original_state_for_skipped_output(self):
        checked = validate_node_update("hotel", {}, state={"days": 1})
        self.assertEqual(checked["hotel_status"], "skipped")
        self.assertEqual(checked["hotel_name"], "")
        with self.assertRaises(ValueError):
            validate_node_update("hotel", {}, state={"days": 2})

    def test_one_day_cannot_be_revised_into_a_hotel_query(self):
        with self.assertRaisesRegex(ValueError, "一日游"):
            build_revision_update({"days": 1}, {
                "approved": False, "comment": "换酒店", "revision_scope": "hotel",
            })

    def test_one_day_attraction_revision_keeps_hotel_skipped(self):
        update = build_revision_update({"days": 1}, {
            "approved": False, "comment": "换景点", "revision_scope": "attraction",
        })
        self.assertEqual(update["hotel_status"], "skipped")
        self.assertEqual(update["transport_segments"], [])

    def test_walking_policy_rejects_nonfinite_negative_and_boolean_limits(self):
        for value in (float("nan"), float("inf"), -1, 0, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                walking_policy({"max_walking_distance_m": value})
        with self.assertRaises(ValueError):
            walking_policy({"allow_walking_fallback": "false"})

    def test_signature_normalizes_budget_and_unranked_preferences(self):
        original = request_state()
        changed = {**original, "budget": 2000, "preferences": ["美食", "历史文化", "美食"]}
        self.assertEqual(build_memory_record(original), build_memory_record(changed))

    def test_all_effective_conditions_change_the_signature(self):
        original = request_state()
        key = build_memory_record(original)["request_signature"]
        changes = [
            {"origin": "北京"}, {"destination": "深圳"}, {"budget": 2001}, {"people": 3},
            {"preferences": ["海滨"]}, {"constraints": ["不能步行"]},
            {"city_transport_mode": "driving"}, {"intercity_transport_mode": "flight"},
            {"allow_walking_fallback": False}, {"max_walking_distance_m": 1000},
            {"max_walking_duration_s": 1200}, {"max_total_walking_distance_m": 2000},
            {"max_total_walking_duration_s": 1800},
            {"departure_date": "2026-10-22", "return_date": "2026-10-23"},
        ]
        for change in changes:
            with self.subTest(change=change):
                self.assertNotEqual(build_memory_record({**original, **change})["request_signature"], key)

    def test_partial_legacy_records_do_not_guess_request_parameters(self):
        self.assertIsNone(build_memory_record({"planner_result": "旧方案"}))

    def test_revision_filters_do_not_change_the_original_query_key(self):
        original = request_state()
        changed = {**original, "attraction_excluded_names": ["大雁塔"],
                   "hotel_search_keyword": "全季", "hotel_excluded_names": ["旧酒店"],
                   "planner_revision_request": "每天增加休息"}
        self.assertEqual(build_memory_record(original), build_memory_record(changed))

    def test_frozen_request_survives_changes_to_transport_settings(self):
        original = request_state()
        changed = {**original, "initial_request_state": capture_initial_request(original),
                   "city_transport_mode": "driving", "max_walking_distance_m": 500}
        self.assertEqual(build_memory_record(original), build_memory_record(changed))
        changed["initial_request_state"]["preferences"].append("摄影")
        self.assertNotIn("摄影", original["preferences"])

    def test_one_day_memory_ignores_unused_hotel_parameters(self):
        original = {**request_state(), "days": 1, "return_date": "2026-10-20"}
        changed = {
            **original, "hotel_search_keyword": "全季",
            "hotel_excluded_names": ["某酒店"], "hotel_revision_request": "旧住宿意见",
        }
        self.assertEqual(build_memory_record(original), build_memory_record(changed))


class MemoryRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.url = "sqlite:///" + str(Path(self.directory.name) / "memory.db")
        self.engine = create_engine(self.url)
        self.addCleanup(self.engine.dispose)
        self.base = declarative_base()
        namespace = {"Base": self.base}
        path = ROOT / "database" / "models.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module == "config.database"
        )]
        exec(compile(tree, str(path), "exec"), namespace)
        self.base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine)
        namespace.update({
            "get_db": self.factory, "func": func, "or_": or_,
            "build_memory_record": build_memory_record,
            "build_memory_payload": build_memory_payload, "memory_signature": memory_signature,
            "confirmation_signature": confirmation_signature,
            "request_state_from_payload": request_state_from_payload, "defaultdict": defaultdict,
        })
        path = ROOT / "database" / "repository.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        path = ROOT / "database" / "memory_migration.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        self.ns = namespace

    def confirmed_state(self):
        state = {**request_state(), "thread_id": uuid4().hex, "human_review_passed": True}
        state["initial_request_state"] = capture_initial_request(state)
        return state

    def test_only_confirmed_result_is_saved_with_reviews_memory_and_completion(self):
        state = self.confirmed_state()
        sid = self.ns["create_travel_session"](state["thread_id"], state)
        state.update({"hotel_name": "修改后的酒店", "city_transport_mode": "driving",
                      "review_passed": False, "human_feedback_history": [{"comment": "换便宜酒店"}]})
        saved = self.ns["save_confirmed_plan"](sid, state, "人工确认")
        self.assertEqual(saved["version"], 1)
        cached = self.ns["find_reusable_plan"](request_state())
        self.assertEqual(cached["result"]["hotel_name"], "修改后的酒店")
        self.assertFalse(cached["result"]["review_passed"])
        self.assertEqual(cached["result"]["effective_request_payload"]["city_transport_mode"], "driving")
        with self.factory() as db:
            for name in ("TravelPlan", "AIReview", "HumanReview", "TravelPlanMemory"):
                self.assertEqual(db.query(self.ns[name]).count(), 1)
            self.assertEqual(db.get(self.ns["TravelSession"], sid).status, "completed")

    def test_unconfirmed_result_cannot_be_saved_as_final(self):
        state = self.confirmed_state()
        sid = self.ns["create_travel_session"](state["thread_id"], state)
        state["human_review_passed"] = False
        with self.assertRaisesRegex(ValueError, "人工确认"):
            self.ns["save_confirmed_plan"](sid, state, "尚未确认")
        with self.factory() as db:
            self.assertEqual(db.query(self.ns["TravelPlan"]).count(), 0)

    def test_final_transaction_failure_rolls_back_everything_and_retries_version_one(self):
        state = self.confirmed_state()
        sid = self.ns["create_travel_session"](state["thread_id"], state)
        original = self.ns["_add_plan_memory"]
        def fail(*args):
            raise RuntimeError("memory failure")
        self.ns["_add_plan_memory"] = fail
        with self.assertRaisesRegex(RuntimeError, "memory failure"):
            self.ns["save_confirmed_plan"](sid, state, "确认")
        with self.factory() as db:
            for name in ("TravelPlan", "AIReview", "HumanReview", "TravelPlanMemory"):
                self.assertEqual(db.query(self.ns[name]).count(), 0)
            self.assertEqual(db.get(self.ns["TravelSession"], sid).status, "planning")
        self.ns["_add_plan_memory"] = original
        self.assertEqual(self.ns["save_confirmed_plan"](sid, state, "确认")["version"], 1)

    def test_repeated_confirmation_and_lost_commit_ack_are_idempotent(self):
        state = self.confirmed_state()
        sid = self.ns["create_travel_session"](state["thread_id"], state)
        failures = []
        class LostAcknowledgement(Session):
            def commit(inner_self):
                super().commit()
                if not failures:
                    failures.append(True)
                    raise RuntimeError("commit acknowledged late")
        self.ns["get_db"] = sessionmaker(bind=self.engine, class_=LostAcknowledgement)
        with self.assertRaisesRegex(RuntimeError, "commit acknowledged late"):
            self.ns["save_confirmed_plan"](sid, state, "确认")
        self.ns["get_db"] = self.factory
        first = self.ns["save_confirmed_plan"](sid, state, "确认")
        second = self.ns["save_confirmed_plan"](sid, state, "确认")
        self.assertEqual(first, second)
        with self.factory() as db:
            for name in ("TravelPlan", "AIReview", "HumanReview", "TravelPlanMemory"):
                self.assertEqual(db.query(self.ns[name]).count(), 1)

    def test_completed_task_cannot_be_reconfirmed_with_different_content(self):
        state = self.confirmed_state()
        sid = self.ns["create_travel_session"](state["thread_id"], state)
        self.ns["save_confirmed_plan"](sid, state, "确认")
        with self.assertRaisesRegex(ValueError, "已完成"):
            self.ns["save_confirmed_plan"](sid, {**state, "planner_result": "另一份内容"}, "确认")

    def legacy_payload(self, state, **changes):
        return {
            **build_memory_payload(state), "schema_version": 1,
            "attraction_excluded_names": [], "hotel_search_keyword": "酒店",
            "hotel_excluded_names": [],
            "revision_requests": {scope: "" for scope in ("attraction", "hotel", "transport", "planner")},
            **changes,
        }

    def set_legacy_memory(self, plan_id, payload):
        with self.factory() as db:
            memory = db.query(self.ns["TravelPlanMemory"]).filter_by(plan_id=plan_id).one()
            memory.request_payload = payload
            memory.request_signature = memory_signature(payload)
            db.commit()

    def test_legacy_final_uses_first_unmodified_request_after_idempotent_migration(self):
        initial = request_state()
        sid, draft_id = self.seed(final=False, approved=False)
        final_state = {**initial, "city_transport_mode": "driving", "planner_result": "改过的最终方案"}
        final_id = self.ns["create_plan_with_review"](sid, final_state, 2)
        self.ns["create_human_review"](final_id, True, "同意修改结果")
        self.ns["mark_plan_final"](final_id)
        self.set_legacy_memory(draft_id, self.legacy_payload(initial))
        self.set_legacy_memory(final_id, self.legacy_payload(final_state,
            hotel_excluded_names=["旧酒店"],
            revision_requests={"hotel": "换便宜的", "planner": "", "attraction": "", "transport": "改自驾"},
        ))
        preview = self.ns["migrate_plan_memory"]()
        self.assertEqual(preview["updated_plan_ids"], [draft_id, final_id])
        self.assertIsNone(self.ns["find_reusable_plan"](initial))
        report = self.ns["migrate_plan_memory"](apply=True)
        self.assertEqual(report["updated_plan_ids"], [draft_id, final_id])
        cached = self.ns["find_reusable_plan"](initial)
        self.assertEqual(cached["plan_id"], final_id)
        self.assertEqual(cached["result"]["planner_result"], "改过的最终方案")
        self.assertEqual(cached["result"]["hotel_excluded_names"], ["旧酒店"])
        self.assertEqual(self.ns["migrate_plan_memory"](apply=True)["updated_plan_ids"], [])
        with self.factory() as db:
            self.assertEqual(db.query(self.ns["TravelPlan"]).count(), 2)

    def test_migration_does_not_guess_original_transport_from_a_modified_snapshot(self):
        _, pid = self.seed()
        self.set_legacy_memory(pid, self.legacy_payload(request_state(),
            attraction_excluded_names=["大雁塔"],
            revision_requests={"attraction": "不要这个景点"},
        ))
        report = self.ns["migrate_plan_memory"](apply=True)
        self.assertEqual(report["updated_plan_ids"], [])
        self.assertEqual(len(report["skipped_session_ids"]), 1)
        self.assertIsNone(self.ns["find_reusable_plan"](request_state()))

    def test_migration_rejects_an_initial_snapshot_that_disagrees_with_task_inputs(self):
        _, pid = self.seed()
        self.set_legacy_memory(pid, self.legacy_payload({**request_state(), "budget": 999}))
        report = self.ns["migrate_plan_memory"](apply=True)
        self.assertEqual(report["updated_plan_ids"], [])

    def seed(self, *, state=None, final=True, completed=True, approved=True):
        state = state or request_state()
        sid = self.ns["create_travel_session"](uuid4().hex, state)
        pid = self.ns["create_plan_with_review"](sid, state, 1)
        self.ns["create_human_review"](pid, approved, "离线人工审核")
        if final:
            self.ns["mark_plan_final"](pid)
        if completed:
            self.ns["complete_travel_session"](sid)
        return sid, pid

    def test_exact_request_reads_confirmed_plan_without_generating_a_new_task(self):
        sid, pid = self.seed()
        cached = self.ns["find_reusable_plan"](request_state())
        self.assertEqual(cached["plan_id"], pid)
        self.assertEqual(cached["session_id"], sid)
        self.assertEqual(cached["result"]["planner_result"], request_state()["planner_result"])
        with self.factory() as db:
            self.assertEqual(db.query(self.ns["TravelSession"]).count(), 1)

    def test_new_memory_table_compiles_for_mysql_without_old_table_changes(self):
        table = self.ns["TravelPlanMemory"].__table__
        ddl = str(CreateTable(table).compile(dialect=mysql.dialect()))
        self.assertIn("request_payload JSON NOT NULL", ddl)
        self.assertIn("result_snapshot JSON NOT NULL", ddl)
        self.assertIn("UNIQUE (plan_id)", ddl)
        self.assertIn("REFERENCES travel_plans (id) ON DELETE CASCADE", ddl)
        self.assertNotIn("request_payload", self.ns["TravelPlan"].__table__.c)

    def test_draft_unfinished_and_rejected_plans_are_not_reused(self):
        self.seed(final=False)
        self.seed(completed=False)
        self.seed(approved=False)
        self.assertIsNone(self.ns["find_reusable_plan"](request_state()))

    def test_latest_human_rejection_invalidates_previous_approval(self):
        _, pid = self.seed()
        self.ns["create_human_review"](pid, False, "重新修改")
        self.assertIsNone(self.ns["find_reusable_plan"](request_state()))

    def test_same_key_returns_latest_confirmed_version_and_final_flag_is_required(self):
        sid, old_pid = self.seed()
        state = {**request_state(), "planner_result": "新的已确认版本"}
        new_pid = self.ns["create_plan_with_review"](sid, state, 2)
        self.ns["create_human_review"](new_pid, True, "确认新版本")
        self.ns["mark_plan_final"](new_pid)
        cached = self.ns["find_reusable_plan"](request_state())
        self.assertEqual(cached["plan_id"], new_pid)
        self.assertEqual(cached["version"], 2)
        self.assertNotEqual(new_pid, old_pid)

    def test_request_payload_is_verified_even_when_a_signature_was_tampered(self):
        _, pid = self.seed()
        with self.factory() as db:
            memory = db.query(self.ns["TravelPlanMemory"]).filter_by(plan_id=pid).one()
            memory.request_payload = {"tampered": True}
            db.commit()
        self.assertIsNone(self.ns["find_reusable_plan"](request_state()))

    def test_deletion_removes_memory_and_prevents_future_cache_hits(self):
        _, pid = self.seed()
        self.assertTrue(self.ns["delete_travel_plan"](pid))
        self.assertIsNone(self.ns["find_reusable_plan"](request_state()))
        with self.factory() as db:
            for name in ("TravelPlanMemory", "TravelPlan", "AIReview", "HumanReview"):
                self.assertEqual(db.query(self.ns[name]).count(), 0)

    def test_persisted_memory_survives_reopening_engine(self):
        _, pid = self.seed()
        self.engine.dispose()
        reopened = create_engine(self.url)
        self.addCleanup(reopened.dispose)
        self.ns["get_db"] = sessionmaker(bind=reopened)
        self.assertEqual(self.ns["find_reusable_plan"](request_state())["plan_id"], pid)

    def test_changed_request_does_not_reuse_existing_plan(self):
        self.seed()
        self.assertIsNone(self.ns["find_reusable_plan"]({**request_state(), "budget": 2001}))

    def test_memory_add_failure_rolls_back_plan_and_review_together(self):
        state = request_state()
        sid = self.ns["create_travel_session"](uuid4().hex, state)
        original = self.ns["_add_plan_memory"]
        def fail(*args):
            raise RuntimeError("memory insert failure")
        self.ns["_add_plan_memory"] = fail
        with self.assertRaisesRegex(RuntimeError, "memory insert failure"):
            self.ns["create_plan_with_review"](sid, state, 1)
        self.ns["_add_plan_memory"] = original
        with self.factory() as db:
            self.assertEqual(db.query(self.ns["TravelPlan"]).count(), 0)
            self.assertEqual(db.query(self.ns["AIReview"]).count(), 0)


if __name__ == "__main__":
    unittest.main()
