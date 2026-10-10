"""选择结果校验的离线测试，不导入模型配置、不访问 API 或数据库。"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unicodedata
import unittest


def _load_helpers(filename, names):
    # 项目节点在导入时配置模型；只加载这些纯函数以保持测试完全离线。
    path = Path(__file__).resolve().parents[1] / "agents" / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"json": json, "unicodedata": unicodedata}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


_attraction = _load_helpers(
    "attraction.py",
    {"_normalize_attraction_name", "_validate_attraction_decision"},
)
_hotel = _load_helpers(
    "hotel.py",
    {"_normalize_hotel_name", "_filter_hotel_candidates", "_validate_hotel_decision"},
)


class AttractionSelectionTests(unittest.TestCase):
    def test_strips_and_deduplicates_names(self):
        decision = SimpleNamespace(
            attraction_names=[" 西安城墙 ", "西安 城墙", "陕西历史博物馆"],
            summary=" 推荐说明 ",
        )
        _attraction["_validate_attraction_decision"](decision, [])
        self.assertEqual(decision.attraction_names, ["西安城墙", "陕西历史博物馆"])
        self.assertEqual(decision.summary, "推荐说明")

    def test_excluded_name_is_rejected_after_normalization(self):
        decision = SimpleNamespace(attraction_names=["西安 城墙"], summary="说明")
        with self.assertRaisesRegex(ValueError, "排除"):
            _attraction["_validate_attraction_decision"](decision, [" 西安城墙 "])

    def test_empty_list_is_rejected(self):
        with self.assertRaises(ValueError):
            _attraction["_validate_attraction_decision"](
                SimpleNamespace(attraction_names=[], summary="说明"), []
            )

    def test_blank_name_is_rejected(self):
        with self.assertRaises(ValueError):
            _attraction["_validate_attraction_decision"](
                SimpleNamespace(attraction_names=["  "], summary="说明"), []
            )

    def test_blank_summary_is_rejected(self):
        with self.assertRaises(ValueError):
            _attraction["_validate_attraction_decision"](
                SimpleNamespace(attraction_names=["西安城墙"], summary="  "), []
            )


class HotelSelectionTests(unittest.TestCase):
    def test_keyword_and_exclusion_filter_preserves_poi(self):
        candidates = [
            {"name": "全季酒店旧店", "location": "108,34"},
            {"name": " 全季酒店新店 ", "location": "109,35"},
            {"name": "如家酒店"},
        ]
        filtered = _hotel["_filter_hotel_candidates"](
            json.dumps(candidates, ensure_ascii=False), "全季", ["全季 酒店旧店"]
        )
        self.assertEqual(filtered, [{"name": "全季酒店新店", "location": "109,35"}])

    def test_generic_keyword_does_not_require_literal_hotel_suffix(self):
        filtered = _hotel["_filter_hotel_candidates"](
            json.dumps([{"name": "西安某某宾馆"}]), "酒店", []
        )
        self.assertEqual(filtered[0]["name"], "西安某某宾馆")

    def test_tool_error_string_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "有效候选"):
            _hotel["_filter_hotel_candidates"]("酒店查询失败", "酒店", [])

    def test_empty_and_invalid_candidates_are_rejected(self):
        for result in ["[]", "{}", '[{"name": " "}]', '[null, 3]']:
            with self.subTest(result=result), self.assertRaises(ValueError):
                _hotel["_filter_hotel_candidates"](result, "酒店", [])

    def test_no_matching_keyword_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "关键词"):
            _hotel["_filter_hotel_candidates"]('[{"name": "如家酒店"}]', "全季", [])

    def test_same_old_hotel_cannot_be_selected(self):
        with self.assertRaises(ValueError):
            _hotel["_filter_hotel_candidates"](
                '[{"name": "全季酒店旧店"}]', "酒店", ["全季酒店旧店"]
            )

    def test_returned_name_must_belong_to_real_candidates(self):
        decision = SimpleNamespace(hotel_name="编造酒店", summary="说明")
        with self.assertRaisesRegex(ValueError, "真实查询候选"):
            _hotel["_validate_hotel_decision"](decision, [{"name": "真实酒店"}])

    def test_selected_name_is_restored_to_canonical_poi_name(self):
        decision = SimpleNamespace(hotel_name="全季 酒店新店", summary=" 说明 ")
        _hotel["_validate_hotel_decision"](decision, [{"name": "全季酒店新店"}])
        self.assertEqual(decision.hotel_name, "全季酒店新店")
        self.assertEqual(decision.summary, "说明")

    def test_blank_hotel_name_or_summary_is_rejected(self):
        for name, summary in [(" ", "说明"), ("真实酒店", " ")]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                _hotel["_validate_hotel_decision"](
                    SimpleNamespace(hotel_name=name, summary=summary),
                    [{"name": "真实酒店"}],
                )


if __name__ == "__main__":
    unittest.main()
