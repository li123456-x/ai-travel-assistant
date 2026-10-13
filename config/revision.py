"""Pure validation and dependency rules for task-local regeneration.

No model, database, or external API is called from this module.
"""

import unicodedata
from copy import deepcopy

from config.trip_policy import requires_hotel, skipped_hotel_update, walking_policy


REVISION_SCOPES = frozenset({"planner", "attraction", "hotel", "transport"})
CITY_TRANSPORT_MODES = frozenset({"transit", "driving", "walking"})
INTERCITY_TRANSPORT_MODES = frozenset({"train", "flight"})
INVALIDATION_RULES = {
    "planner": (),
    "attraction": ("attraction", "hotel", "transport"),
    "hotel": ("hotel", "transport"),
    "transport": ("transport",),
}
RESULT_FIELDS = {
    "attraction": ("attractions_result", "attraction_names"),
    "hotel": ("hotel_result", "hotel_name"),
    "transport": ("transport_result", "transport_segments"),
}


def normalize_name(name: str) -> str:
    return "".join(unicodedata.normalize("NFKC", name).split()).casefold()


def _text(value, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label}必须是文本")
    value = value.strip()
    if not value and not allow_empty:
        raise ValueError(f"{label}不能为空")
    return value


def _names(value, label: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{label}必须是{'可为空的' if allow_empty else '非空的'}列表")
    result = []
    seen = set()
    for item in value:
        name = _text(item, label)
        key = normalize_name(name)
        if key not in seen:
            result.append(name)
            seen.add(key)
    return result


def build_revision_update(state: dict, decision: dict) -> dict:
    """Validate feedback, persist its parameters, and invalidate dependencies."""
    if not isinstance(decision, dict) or type(decision.get("approved")) is not bool:
        raise ValueError("人工审核结果必须包含布尔类型的 approved")
    approved = decision["approved"]
    comment = _text(decision.get("comment", ""), "修改意见", allow_empty=approved)
    update = {
        "human_review_passed": approved,
        "human_review_result": comment,
    }
    if approved:
        update["trigger_reason"] = "human_review_pass"
        return update

    scope = decision.get("revision_scope", "planner")
    if not isinstance(scope, str) or scope not in REVISION_SCOPES:
        raise ValueError("不支持的修改类型")
    if scope == "hotel" and not requires_hotel(state):
        raise ValueError("一日游不查询酒店；请修改景点、交通或行程")

    update.update({
        "trigger_reason": "human_review_reject",
        "human_revision_count": state.get("human_revision_count", 0) + 1,
        "review_retry_count": 0,
        "review_passed": False,
        "review_result": "",
        f"{scope}_revision_request": comment,
    })

    if scope == "attraction":
        if "attraction_excluded_names" in decision:
            update["attraction_excluded_names"] = _names(
                decision["attraction_excluded_names"], "排除景点", allow_empty=True
            )
    elif scope == "hotel":
        keyword = decision.get("hotel_keyword", "酒店")
        update["hotel_search_keyword"] = _text(
            keyword, "酒店关键词", allow_empty=True
        ) or "酒店"
        old_hotel = state.get("hotel_name", "")
        update["hotel_excluded_names"] = (
            [old_hotel.strip()] if isinstance(old_hotel, str) and old_hotel.strip() else []
        )
    elif scope == "transport":
        for key, choices, default, label in (
            ("city_transport_mode", CITY_TRANSPORT_MODES, "transit", "市内交通方式"),
            ("intercity_transport_mode", INTERCITY_TRANSPORT_MODES, "train", "城际交通方式"),
        ):
            value = decision.get(key, state.get(key, default))
            if not isinstance(value, str) or value not in choices:
                raise ValueError(f"不支持的{label}")
            update[key] = value
        policy_state = {**state, **{
            key: decision[key]
            for key in (
                "allow_walking_fallback", "max_walking_distance_m", "max_walking_duration_s",
                "max_total_walking_distance_m", "max_total_walking_duration_s",
            ) if key in decision
        }}
        update.update(walking_policy(policy_state))

    for node_name in INVALIDATION_RULES[scope]:
        update[f"{node_name}_status"] = "dirty"
        for field in RESULT_FIELDS[node_name]:
            update[field] = [] if field in {"attraction_names", "transport_segments"} else ""

    if not requires_hotel(state):
        update.update(skipped_hotel_update())

    update["human_feedback_history"] = [
        *deepcopy(state.get("human_feedback_history", [])),
        {"revision_scope": scope, "comment": comment},
    ]

    # Keep session/thread identifiers and the previous planner_result unchanged.
    # Existing database versions remain independent of these state updates.
    return update


def validate_node_update(agent_name: str, update: dict, *, state: dict | None = None) -> dict:
    """Check structural usability before recording a node as completed."""
    if not isinstance(update, dict):
        raise ValueError(f"{agent_name} 节点必须返回状态字典")
    checked = dict(update)
    if agent_name == "hotel" and not requires_hotel(state if state is not None else checked):
        checked.update(skipped_hotel_update())
        return checked
    result_keys = {
        "attraction": "attractions_result",
        "hotel": "hotel_result",
        "transport": "transport_result",
        "planner": "planner_result",
        "reviewer": "review_result",
    }
    if agent_name not in result_keys:
        raise ValueError(f"不支持的节点：{agent_name}")
    field = result_keys[agent_name]
    checked[field] = _text(update.get(field), f"{agent_name} 输出")
    if agent_name == "attraction":
        checked["attraction_names"] = _names(update.get("attraction_names"), "景点名称")
    elif agent_name == "hotel":
        checked["hotel_name"] = _text(update.get("hotel_name"), "酒店名称")
    elif agent_name == "reviewer" and type(update.get("review_passed")) is not bool:
        raise ValueError("AI 审核结果必须包含布尔类型的 review_passed")
    if agent_name in RESULT_FIELDS:
        checked[f"{agent_name}_status"] = "completed"
    return checked


def build_revision_context(state: dict) -> str:
    """Keep the latest request for each scope across subsequent review rounds."""
    lines = []
    for scope, label in (
        ("attraction", "景点修改要求"),
        ("hotel", "酒店修改要求"),
        ("transport", "交通修改要求"),
        ("planner", "行程修改要求"),
    ):
        request = state.get(f"{scope}_revision_request", "")
        if request:
            lines.append(f"{label}：{request}")
    lines.extend([
        f"住宿规则：{'需要酒店' if requires_hotel(state) else '一日游，不查询或安排酒店'}",
        f"排除景点：{state.get('attraction_excluded_names', [])}",
        f"酒店查询关键词：{state.get('hotel_search_keyword', '酒店')}",
        f"排除酒店：{state.get('hotel_excluded_names', [])}",
        f"市内交通方式：{state.get('city_transport_mode', 'transit')}",
        f"城际交通方式：{state.get('intercity_transport_mode', 'train')}",
        f"公交为空时的步行兜底规则：{walking_policy(state)}",
    ])
    return "\n".join(lines)
