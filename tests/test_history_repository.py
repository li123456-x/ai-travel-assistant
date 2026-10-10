"""用真实 ORM 和临时数据库验证历史查询、重开及关联删除。"""

import ast
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sqlalchemy import create_engine, or_
from sqlalchemy.orm import declarative_base, sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HistoryRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.url = "sqlite:///" + str(Path(self.directory.name) / "history.db")
        self.engine = create_engine(self.url)
        self.addCleanup(self.engine.dispose)
        self.base = declarative_base()
        namespace = {"Base": self.base}
        path = PROJECT_ROOT / "database" / "models.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module == "config.database"
        )]
        exec(compile(tree, str(path), "exec"), namespace)
        self.base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine)
        namespace.update({"get_db": self.factory, "or_": or_})
        path = PROJECT_ROOT / "database" / "repository.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"get_travel_sessions", "get_plan_versions", "delete_travel_plan"}]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        self.namespace = namespace

    def seed(self, destination="西安", request="历史文化旅游", with_plan=True):
        with self.factory() as db:
            session = self.namespace["TravelSession"](
                thread_id=f"task-{destination}-{request}", user_request=request,
                origin="临沂", destination=destination,
                departure_date="2026-10-09", return_date="2026-10-10",
                days=2, people=2, budget=2000,
                created_at=datetime(2026, 10, 9),
            )
            db.add(session)
            db.flush()
            session_id = session.id
            if with_plan:
                plan = self.namespace["TravelPlan"](
                    session_id=session_id, version=1,
                    planner_result="已保存的完整旅游方案", is_final=True,
                )
                db.add(plan)
                db.flush()
                plan_id = plan.id
                db.add(self.namespace["AIReview"](
                    plan_id=plan_id, review_passed=True, review_result="审核通过"
                ))
                db.add(self.namespace["HumanReview"](
                    plan_id=plan_id, approved=True, comment="同意"
                ))
            db.commit()
            return session_id

    def test_search_lists_saved_tasks_newest_first_and_treats_wildcards_literally(self):
        first = self.seed()
        second = self.seed("北京", "预算100%满意_行程")
        self.seed("上海", with_plan=False)
        get_sessions = self.namespace["get_travel_sessions"]
        self.assertEqual([item["session_id"] for item in get_sessions()], [second, first])
        self.assertEqual(get_sessions(" 西安 ")[0]["session_id"], first)
        self.assertEqual(len(get_sessions("临沂")), 2)
        self.assertEqual([item["session_id"] for item in get_sessions("%")], [second])
        self.assertEqual([item["session_id"] for item in get_sessions("_")], [second])
        self.assertEqual(get_sessions("不存在"), [])

    def test_history_survives_new_engine_without_workflow_state(self):
        session_id = self.seed()
        self.engine.dispose()
        reopened = create_engine(self.url)
        self.addCleanup(reopened.dispose)
        self.namespace["get_db"] = sessionmaker(bind=reopened)
        self.assertEqual(self.namespace["get_travel_sessions"]()[0]["session_id"], session_id)
        versions = self.namespace["get_plan_versions"](session_id)
        self.assertEqual(versions[0]["planner_result"], "已保存的完整旅游方案")
        self.assertEqual(versions[0]["review_result"], "审核通过")
        self.assertEqual(versions[0]["human_comment"], "同意")
        self.assertTrue(versions[0]["is_final"])

    def test_delete_removes_selected_version_and_reviews_but_preserves_other_task(self):
        first = self.seed()
        second = self.seed("北京")
        versions = self.namespace["get_plan_versions"](first)
        self.assertTrue(self.namespace["delete_travel_plan"](versions[0]["plan_id"]))
        self.assertFalse(self.namespace["delete_travel_plan"](versions[0]["plan_id"]))
        self.assertEqual(self.namespace["get_plan_versions"](first), [])
        self.assertEqual(self.namespace["get_travel_sessions"]()[0]["session_id"], second)
        with self.factory() as db:
            for model in ("TravelPlan", "AIReview", "HumanReview"):
                self.assertEqual(db.query(self.namespace[model]).count(), 1)
            self.assertEqual(db.query(self.namespace["TravelSession"]).count(), 2)


if __name__ == "__main__":
    unittest.main()
