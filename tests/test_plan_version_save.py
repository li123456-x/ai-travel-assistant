"""方案和审核事务保存的离线测试，无数据库连接。"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from config.memory import build_memory_record


class FakePlan(SimpleNamespace):
    pass


class FakeReview(SimpleNamespace):
    pass


class FakeSession:
    def __init__(self, failure=None):
        self.failure = failure
        self.added = []
        self.committed = []
        self.commit_calls = 0
        self.rollback_calls = 0
        self.close_calls = 0
        self.flush_calls = 0

    def add(self, item):
        if self.failure == "review_add" and isinstance(item, FakeReview):
            raise RuntimeError("审核写入失败")
        self.added.append(item)

    def flush(self):
        self.flush_calls += 1
        if self.failure == "flush":
            raise RuntimeError("方案写入失败")
        self.added[0].id = 101

    def commit(self):
        self.commit_calls += 1
        if self.failure == "commit":
            raise RuntimeError("提交失败")
        self.committed = list(self.added)

    def rollback(self):
        self.rollback_calls += 1
        self.added.clear()

    def close(self):
        self.close_calls += 1


def _load_save_function(db):
    # 仓库模块导入时读取数据库配置，只加载被测函数隔离这些外部副作用。
    path = Path(__file__).resolve().parents[1] / "database" / "repository.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_plan_with_review"
    )
    namespace = {"get_db": lambda: db, "TravelPlan": FakePlan, "AIReview": FakeReview}
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_add_plan_memory")
    namespace.update({"build_memory_record": build_memory_record, "TravelPlanMemory": FakePlan})
    exec(compile(ast.Module(body=[helper, function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["create_plan_with_review"]


class PlanVersionSaveTests(unittest.TestCase):
    def test_plan_and_review_are_saved_with_one_commit(self):
        db = FakeSession()
        save = _load_save_function(db)
        state = {
            "attractions_result": "完整景点结果",
            "hotel_result": "完整酒店结果",
            "transport_result": "完整交通结果",
            "planner_result": "完整行程",
            "review_passed": True,
            "review_result": "完整审核",
        }

        plan_id = save(session_id=7, state=state, version=3)

        self.assertEqual(plan_id, 101)
        self.assertEqual(db.flush_calls, 1)
        self.assertEqual(db.commit_calls, 1)
        self.assertEqual(db.rollback_calls, 0)
        self.assertEqual(db.close_calls, 1)
        self.assertEqual(len(db.committed), 2)
        plan, review = db.committed
        self.assertEqual(plan.session_id, 7)
        self.assertEqual(plan.version, 3)
        self.assertFalse(plan.is_final)
        for key in ["attractions_result", "hotel_result", "transport_result", "planner_result"]:
            self.assertEqual(getattr(plan, key), state[key])
        self.assertEqual(review.plan_id, plan_id)
        self.assertTrue(review.review_passed)
        self.assertEqual(review.review_result, state["review_result"])

    def test_flush_failure_rolls_back_and_closes(self):
        db = FakeSession("flush")
        with self.assertRaisesRegex(RuntimeError, "方案写入失败"):
            _load_save_function(db)(7, {}, 3)
        self.assertEqual(db.commit_calls, 0)
        self.assertEqual(db.rollback_calls, 1)
        self.assertEqual(db.close_calls, 1)
        self.assertEqual(db.committed, [])

    def test_review_failure_does_not_commit_partial_plan(self):
        db = FakeSession("review_add")
        with self.assertRaisesRegex(RuntimeError, "审核写入失败"):
            _load_save_function(db)(7, {}, 3)
        self.assertEqual(db.commit_calls, 0)
        self.assertEqual(db.rollback_calls, 1)
        self.assertEqual(db.close_calls, 1)
        self.assertEqual(db.committed, [])
        self.assertEqual(db.added, [])

    def test_commit_failure_rolls_back_and_closes(self):
        db = FakeSession("commit")
        with self.assertRaisesRegex(RuntimeError, "提交失败"):
            _load_save_function(db)(7, {}, 3)
        self.assertEqual(db.commit_calls, 1)
        self.assertEqual(db.rollback_calls, 1)
        self.assertEqual(db.close_calls, 1)
        self.assertEqual(db.committed, [])

    def test_missing_optional_results_keep_existing_defaults(self):
        db = FakeSession()
        _load_save_function(db)(7, {}, 3)
        plan, review = db.committed
        self.assertEqual(plan.planner_result, "")
        self.assertEqual(plan.attractions_result, "")
        self.assertFalse(review.review_passed)
        self.assertEqual(review.review_result, "")


if __name__ == "__main__":
    unittest.main()
