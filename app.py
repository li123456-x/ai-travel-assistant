import uuid
from datetime import date, timedelta

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from config.graph import builder
from config.memory import capture_initial_request
from config.revision import build_revision_update
from config.trip_policy import requires_hotel
from database.repository import (
    create_travel_session,
    delete_travel_plan,
    find_reusable_plan,
    get_plan_versions,
    get_travel_sessions,
    mark_plan_final,
    save_confirmed_plan,
)

st.set_page_config(
    page_title="AI 多智能体旅游助手",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ AI 多智能体旅游助手")
st.caption("基于 LangGraph + 多智能体 + Tool Calling + Human-in-the-loop + MySQL")


GRAPH_SCHEMA_VERSION = 4

if (
    "graph" not in st.session_state
    or st.session_state.get("graph_schema_version") != GRAPH_SCHEMA_VERSION
):
    if "graph" in st.session_state:
        # Old in-memory checkpoints do not contain the regeneration fields.
        # Only reset this browser session; database history is not deleted.
        for key in (
            "thread_id", "result", "running", "show_feedback", "session_id",
            "current_plan_id", "version", "generation_failed", "pending_plan_save",
            "last_generation_error",
            "cached_plan",
            "pending_generation", "confirmation_error",
        ):
            st.session_state.pop(key, None)
        st.info("工作流已更新，请创建新任务。数据库中的历史方案仍然保留。")
    memory = InMemorySaver()
    st.session_state.graph = builder.compile(checkpointer=memory)
    st.session_state.graph_schema_version = GRAPH_SCHEMA_VERSION


DEFAULT_SESSION_STATE = {
    "thread_id": lambda: f"travel_{uuid.uuid4().hex[:8]}",
    "result": None,
    "running": False,
    "show_feedback": False,
    "session_id": None,
    "current_plan_id": None,
    "version": 0,
    "generation_failed": False,
    "pending_generation": False,
    "confirmation_error": "",
    "last_generation_error": "",
    "cached_plan": None,
}

for key, default_value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = (
            default_value() if callable(default_value) else default_value
        )


graph = st.session_state.graph


def _graph_config():
    return {
        "configurable": {
            "thread_id": st.session_state.thread_id,
        }
    }


def _split_text(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _build_initial_state(
    origin,
    destination,
    departure_date,
    return_date,
    days,
    people,
    budget,
    preferences_text,
    constraints_text,
    *,
    city_transport_mode="transit",
    intercity_transport_mode="train",
    allow_walking_fallback=True,
    max_walking_distance_m=2000.0,
    max_walking_duration_s=1800.0,
    max_total_walking_distance_m=4000.0,
    max_total_walking_duration_s=3600.0,
):
    preferences = _split_text(preferences_text)
    constraints = _split_text(constraints_text)

    user_request = (
        f"我从{origin}出发，"
        f"前往{destination}旅游，"
        f"出发日期为{departure_date}，"
        f"返程日期为{return_date}，"
        f"共旅游{days}天，"
        f"{people}个人，"
        f"总预算{budget}元，"
        f"偏好{preferences_text}，"
        f"要求{constraints_text}"
    )

    state = {
        "session_id": None,
        "thread_id": st.session_state.thread_id,
        "trigger_reason": "initial",
        "agent_run_id": None,
        "user_request": user_request,
        "origin": origin.strip(),
        "destination": destination.strip(),
        "departure_date": departure_date.strftime("%Y-%m-%d"),
        "return_date": return_date.strftime("%Y-%m-%d"),
        "days": int(days),
        "needs_hotel": int(days) > 1,
        "people": int(people),
        "budget": float(budget),
        "preferences": preferences,
        "constraints": constraints,
        "attractions_result": "",
        "attraction_names": [],
        "hotel_result": "",
        "hotel_name": "",
        "transport_result": "",
        "transport_segments": [],
        "planner_result": "",
        "review_result": "",
        "review_passed": False,
        "review_retry_count": 0,
        "max_review_retries": 2,
        "human_review_result": "",
        "human_review_passed": False,
        "human_feedback_history": [],
        "human_revision_count": 0,
        "max_human_revisions": 3,
        "next_agent": "",

        "attraction_status": "pending",
        "hotel_status": "pending" if int(days) > 1 else "skipped",
        "transport_status": "pending",

        "attraction_revision_request": "",
        "hotel_revision_request": "",
        "transport_revision_request": "",
        "planner_revision_request": "",

        "attraction_excluded_names": [],
        "hotel_search_keyword": "酒店",
        "hotel_excluded_names": [],
        "city_transport_mode": city_transport_mode,
        "intercity_transport_mode": intercity_transport_mode,
        "allow_walking_fallback": allow_walking_fallback,
        "max_walking_distance_m": float(max_walking_distance_m),
        "max_walking_duration_s": float(max_walking_duration_s),
        "max_total_walking_distance_m": float(max_total_walking_distance_m),
        "max_total_walking_duration_s": float(max_total_walking_duration_s),

    }
    state["initial_request_state"] = capture_initial_request(state)
    return state


def _stage_candidate(result):
    if not isinstance(result, dict) or not result.get("__interrupt__"):
        raise ValueError("任务未到达人工审核，不能展示为待确认方案")
    st.session_state.result = result
    st.session_state.running = True
    st.session_state.show_feedback = False
    st.session_state.pending_generation = False
    st.session_state.generation_failed = False
    st.session_state.last_generation_error = ""
    st.session_state.confirmation_error = ""


def _resume_graph(
    approved: bool,
    comment: str,
    revision_scope: str = "planner",
    hotel_keyword: str | None = None,
    attraction_excluded_names: list[str] | None = None,
    city_transport_mode: str | None = None,
    intercity_transport_mode: str | None = None,
    allow_walking_fallback: bool | None = None,
    max_walking_distance_m: float | None = None,
    max_walking_duration_s: float | None = None,
    max_total_walking_distance_m: float | None = None,
    max_total_walking_duration_s: float | None = None,
):
    resume_data = {
        "approved": approved,
        "comment": comment.strip(),
        "revision_scope": revision_scope,
    }
    if hotel_keyword is not None:
        resume_data["hotel_keyword"] = hotel_keyword.strip() or "酒店"
    if attraction_excluded_names is not None:
        resume_data["attraction_excluded_names"] = list(attraction_excluded_names)
    if city_transport_mode is not None:
        resume_data["city_transport_mode"] = city_transport_mode
    if intercity_transport_mode is not None:
        resume_data["intercity_transport_mode"] = intercity_transport_mode
    for key, value in (
        ("allow_walking_fallback", allow_walking_fallback),
        ("max_walking_distance_m", max_walking_distance_m),
        ("max_walking_duration_s", max_walking_duration_s),
        ("max_total_walking_distance_m", max_total_walking_distance_m),
        ("max_total_walking_duration_s", max_total_walking_duration_s),
    ):
        if value is not None:
            resume_data[key] = value

    return graph.invoke(
        Command(resume=resume_data),
        config=_graph_config(),
    )


with st.sidebar:
    st.header("🧳 旅游需求")

    origin = st.text_input("出发地", value="临沂")
    destination = st.text_input("目的地", value="西安")
    departure_date = st.date_input("出发日期", value=date.today())
    return_date = st.date_input(
        "返程日期",
        value=date.today() + timedelta(days=1),
    )

    if return_date >= departure_date:
        calculated_days = (return_date - departure_date).days + 1
        st.info(f"本次行程共 {calculated_days} 天")

    people = st.number_input(
        "出行人数",
        min_value=1,
        max_value=20,
        value=2,
    )
    budget = st.number_input(
        "总预算（元）",
        min_value=0,
        value=2000,
        step=100,
    )
    preferences_text = st.text_input("旅游偏好", value="历史文化")
    constraints_text = st.text_input("其他要求", value="不要太累")
    with st.expander("交通设置", expanded=False):
        initial_city_options = {"公共交通优先": "transit", "自驾": "driving", "步行": "walking"}
        initial_city_label = st.selectbox(
            "初始市内交通方式", list(initial_city_options), key="initial_city_mode",
        )
        initial_intercity_options = {"高铁 / 火车": "train", "飞机": "flight"}
        initial_intercity_label = st.selectbox(
            "初始城际交通方式", list(initial_intercity_options), key="initial_intercity_mode",
        )
        allow_walking_fallback = st.checkbox(
            "公交无方案时允许短距离步行", value=True, key="initial_allow_walking",
        )
        max_walking_km = st.number_input(
            "单段步行兜底上限（公里）", min_value=0.1, max_value=20.0,
            value=2.0, step=0.1, key="initial_walking_km",
        )
        max_walking_minutes = st.number_input(
            "单段步行兜底上限（分钟）", min_value=1, max_value=240,
            value=30, key="initial_walking_minutes",
        )
        max_total_walking_km = st.number_input(
            "累计步行兜底上限（公里）", min_value=0.1, max_value=40.0,
            value=4.0, step=0.1, key="initial_total_walking_km",
        )
        max_total_walking_minutes = st.number_input(
            "累计步行兜底上限（分钟）", min_value=1, max_value=480,
            value=60, key="initial_total_walking_minutes",
        )
    force_refresh = st.checkbox("重新查询，不复用历史方案", value=False, key="force_refresh")
    generate_button = st.button("🚀 生成旅游方案", use_container_width=True)

    st.divider()
    st.header("📚 历史方案")
    history_search = st.text_input("搜索历史任务", placeholder="出发地、目的地或需求")
    try:
        history_sessions = get_travel_sessions(history_search)
    except Exception as e:
        st.error(f"读取历史任务失败：{e}")
        history_sessions = []

    if not history_sessions:
        st.caption("没有匹配的历史方案" if history_search.strip() else "暂无已保存的历史方案")
    else:
        history_by_id = {item["session_id"]: item for item in history_sessions}
        history_ids = list(history_by_id)
        if st.session_state.get("history_session_id") not in history_by_id:
            current_id = st.session_state.session_id
            st.session_state.history_session_id = (
                current_id if current_id in history_by_id else history_ids[0]
            )

        history_session_id = st.selectbox(
            "选择历史任务",
            history_ids,
            format_func=lambda session_id: (
                f"{history_by_id[session_id]['origin']} → "
                f"{history_by_id[session_id]['destination']}｜"
                f"{history_by_id[session_id]['departure_date']}｜任务 {session_id}"
            ),
            key="history_session_id",
        )
        history_session = history_by_id[history_session_id]
        st.caption(
            f"出行日期：{history_session['departure_date']} 至 "
            f"{history_session['return_date']}"
        )
        is_current_history = history_session_id == st.session_state.session_id
        history_locked = is_current_history and (
            st.session_state.running or st.session_state.generation_failed
        )

        try:
            versions = get_plan_versions(history_session_id)
        except Exception as e:
            st.error(f"读取历史版本失败：{e}")
            versions = []

        for item in versions:
            title = f"Version {item['version']}"
            if item["is_final"]:
                title += "｜最终版"

            with st.expander(title, expanded=False):
                st.caption(f"方案编号：{item['plan_id']}")
                st.markdown(item["planner_result"] or "暂无方案内容")

                if item["review_result"]:
                    st.markdown("**AI 审核**")
                    if item["review_passed"] is True:
                        st.success("AI 审核通过")
                    elif item["review_passed"] is False:
                        st.warning("AI 审核未通过")
                    st.markdown(item["review_result"])

                if item["human_comment"]:
                    st.markdown("**人工意见**")
                    st.markdown(item["human_comment"])

                set_final = st.button(
                    "设为最终版",
                    key=f"set_final_{item['plan_id']}",
                    use_container_width=True,
                    disabled=(
                        item["is_final"]
                        or history_locked
                    ),
                )

                delete_version = st.button(
                    "删除此版本",
                    key=f"delete_plan_{item['plan_id']}",
                    use_container_width=True,
                    disabled=history_locked,
                )

                if set_final:
                    try:
                        if not mark_plan_final(item["plan_id"]):
                            raise ValueError("该方案已不存在，请刷新历史列表")
                        if is_current_history:
                            st.session_state.current_plan_id = item["plan_id"]
                        cached = st.session_state.get("cached_plan")
                        if cached and cached["session_id"] == history_session_id:
                            st.session_state.cached_plan = None
                        st.success("已设为最终版")
                        st.rerun()
                    except Exception as e:
                        st.error(f"设置最终版失败：{e}")

                if delete_version:
                    try:
                        if not delete_travel_plan(item["plan_id"]):
                            raise ValueError("该方案已不存在，请刷新历史列表")
                        if is_current_history and st.session_state.current_plan_id == item["plan_id"]:
                            st.session_state.current_plan_id = None
                        cached = st.session_state.get("cached_plan")
                        if cached and cached["plan_id"] == item["plan_id"]:
                            st.session_state.cached_plan = None
                        st.success("已删除该版本")
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除版本失败：{e}")


if generate_button:
    if not origin.strip() or not destination.strip():
        st.error("出发地和目的地不能为空")
        st.stop()
    if return_date < departure_date:
        st.error("返程日期不能早于出发日期")
        st.stop()

    st.session_state.thread_id = f"travel_{uuid.uuid4().hex[:8]}"
    st.session_state.result = None
    st.session_state.running = False
    st.session_state.show_feedback = False
    st.session_state.session_id = None
    st.session_state.current_plan_id = None
    st.session_state.version = 0
    st.session_state.generation_failed = False
    st.session_state.pending_generation = False
    st.session_state.confirmation_error = ""
    st.session_state.last_generation_error = ""
    st.session_state.cached_plan = None

    days = (return_date - departure_date).days + 1
    initial_state = _build_initial_state(
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        return_date=return_date,
        days=days,
        people=people,
        budget=budget,
        preferences_text=preferences_text,
        constraints_text=constraints_text,
        city_transport_mode=initial_city_options[initial_city_label],
        intercity_transport_mode=initial_intercity_options[initial_intercity_label],
        allow_walking_fallback=allow_walking_fallback,
        max_walking_distance_m=max_walking_km * 1000,
        max_walking_duration_s=max_walking_minutes * 60,
        max_total_walking_distance_m=max_total_walking_km * 1000,
        max_total_walking_duration_s=max_total_walking_minutes * 60,
    )

    try:
        cached_plan = None if force_refresh else find_reusable_plan(initial_state)
    except Exception as e:
        st.error(f"读取长期记忆失败：{e}")
        st.stop()
    if cached_plan is not None:
        st.session_state.cached_plan = cached_plan
        st.rerun()

    try:
        session_id = create_travel_session(
            thread_id=st.session_state.thread_id,
            state=initial_state,
        )
        st.session_state.session_id = session_id
        initial_state["session_id"] = session_id
    except Exception as e:
        st.error(f"创建数据库旅行任务失败：{e}")
        st.stop()

    try:
        st.session_state.pending_generation = True
        with st.spinner("多智能体正在规划旅游方案..."):
            result = graph.invoke(initial_state, config=_graph_config())
        _stage_candidate(result)
    except Exception as e:
        st.session_state.generation_failed = True
        st.session_state.last_generation_error = f"LangGraph 执行失败：{e}"

    if not st.session_state.generation_failed:
        st.rerun()


if st.session_state.generation_failed:
    if st.session_state.last_generation_error:
        st.error(st.session_state.last_generation_error)
    st.warning("本次生成未完成，不能审核旧结果作为新方案。重试不会新建旅行任务。")
    if st.button("🔁 从失败位置重试", key="retry_failed_generation"):
        try:
            with st.spinner("正在继续当前任务..."):
                recovered = graph.invoke(None, config=_graph_config())
            _stage_candidate(recovered)
            st.rerun()
        except Exception as e:
            st.session_state.last_generation_error = f"重试失败：{e}"
            st.error(f"重试失败：{e}")

    if st.session_state.result and st.button(
        "↩️ 取消本次修改，回到上一版再调整条件",
        key="cancel_failed_regeneration",
    ):
        try:
            # Restore the previous in-memory candidate, not a persisted draft.
            # Restore that checkpoint, not the failed/dirty specialist outputs.
            graph.update_state(
                _graph_config(),
                st.session_state.result,
                as_node="reviewer",
            )
            restored = graph.invoke(None, config=_graph_config())
            if not restored.get("__interrupt__"):
                raise ValueError("未能恢复到上一版人工审核位置")
            st.session_state.result = restored
            st.session_state.generation_failed = False
            st.session_state.pending_generation = False
            st.session_state.last_generation_error = ""
            st.session_state.running = True
            st.session_state.show_feedback = False
            st.rerun()
        except Exception as e:
            st.error(f"返回上一版失败：{e}")


cached_plan = st.session_state.cached_plan
if cached_plan:
    saved = cached_plan["result"]
    st.subheader("历史方案复用")
    st.caption(
        f"来源任务 {cached_plan['session_id']}，Version {cached_plan['version']}，"
        f"保存时间：{cached_plan['created_at']}"
    )
    st.warning("这是已确认的历史方案快照，本次没有重新查询实时路线、班次或票价。")
    with st.expander("景点", expanded=False):
        st.markdown(saved["attractions_result"])
    if saved.get("needs_hotel", True):
        with st.expander("酒店", expanded=False):
            st.markdown(saved["hotel_result"])
    with st.expander("交通", expanded=False):
        st.markdown(saved["transport_result"])
        if saved.get("transport_segments"):
            st.dataframe(saved["transport_segments"], hide_index=True, use_container_width=True)
    st.markdown(saved["planner_result"])
    if saved.get("human_feedback_history"):
        with st.expander("已采纳的修改意见", expanded=False):
            for feedback in saved["human_feedback_history"]:
                st.markdown(feedback["comment"])

result = st.session_state.result

if result:
    st.divider()
    st.subheader("🤖 多智能体执行结果")

    with st.expander("🏛️ Attraction Agent", expanded=False):
        st.markdown(result.get("attractions_result", "暂无结果"))

    if requires_hotel(result):
        with st.expander("🏨 Hotel Agent", expanded=False):
            st.markdown(result.get("hotel_result", "暂无结果"))

    with st.expander("🚇 Transport Agent", expanded=False):
        st.markdown(result.get("transport_result", "暂无结果"))
        if result.get("transport_segments"):
            st.dataframe(result["transport_segments"], hide_index=True, use_container_width=True)

    st.subheader("📅 待确认旅游方案" if st.session_state.running else "📅 最终旅游方案")
    st.markdown(result.get("planner_result", ""))

    st.subheader("🔍 AI Reviewer")
    if result.get("review_passed"):
        st.success("✅ AI 审核通过")
    else:
        st.warning("⚠️ AI 审核未通过")

    st.markdown(result.get("review_result", ""))

    review_retry_count = result.get("review_retry_count", 0)
    max_review_retries = result.get("max_review_retries", 2)
    if not result.get("review_passed") and review_retry_count >= max_review_retries:
        st.info("AI 已达到最大自动修改次数，当前方案交由人工审核判断。")

    if st.session_state.version > 0:
        st.caption(f"当前方案版本：Version {st.session_state.version}")

    if st.session_state.running and not st.session_state.generation_failed:
        st.divider()
        st.subheader("👤 人工审核")
        st.write("请确认是否接受当前旅游方案。")

        col1, col2 = st.columns(2)

        with col1:
            approve_button = st.button(
                "重试保存最终方案" if result.get("human_review_passed") else "✅ 同意方案",
                use_container_width=True,
                key="approve_plan",
            )

        with col2:
            reject_button = st.button(
                "❌ 修改方案",
                use_container_width=True,
                key="reject_plan",
                disabled=result.get("human_review_passed", False),
            )

        if st.session_state.confirmation_error:
            st.error(st.session_state.confirmation_error)
            if result.get("human_review_passed"):
                st.warning("人工确认已完成但最终方案尚未保存，重试保存不会再次生成方案。")
            else:
                st.warning("人工确认尚未完成，请重试确认。")

        if approve_button:
            comment = "人工审核通过"

            try:
                if not result.get("human_review_passed"):
                    with st.spinner("正在确认最终旅游方案..."):
                        result = _resume_graph(approved=True, comment=comment)
                    st.session_state.result = result
                    st.session_state.show_feedback = False
                saved = save_confirmed_plan(
                    session_id=st.session_state.session_id, state=result, comment=comment,
                )
                st.session_state.current_plan_id = saved["plan_id"]
                st.session_state.version = saved["version"]
                st.session_state.confirmation_error = ""

                st.session_state.result = result
                st.session_state.running = False
                st.session_state.show_feedback = False
                st.rerun()

            except Exception as e:
                st.session_state.confirmation_error = f"确认或保存最终方案失败：{e}"
                st.rerun()

        if reject_button:
            st.session_state.show_feedback = True
            st.rerun()

        if st.session_state.get("show_feedback", False):
            st.divider()
            st.write("### ✏️ 修改意见")

            revision_options = {
                "仅调整行程": "planner",
                "更换景点": "attraction",
                "更换交通方式": "transport",
            }
            if requires_hotel(result):
                revision_options["更换酒店"] = "hotel"
            revision_key = (
                f"{st.session_state.thread_id}_"
                f"{result.get('human_revision_count', 0)}"
            )
            revision_label = st.selectbox(
                "修改类型",
                list(revision_options),
                key=f"revision_scope_{revision_key}",
            )
            revision_scope = revision_options[revision_label]
            revision_parameters = {}

            if revision_scope == "attraction":
                st.caption("会重新推荐景点，并同步更新酒店和交通。")
                previous_exclusions = result.get("attraction_excluded_names", [])
                attraction_options = list(dict.fromkeys(
                    result.get("attraction_names", []) + previous_exclusions
                ))
                revision_parameters["attraction_excluded_names"] = st.multiselect(
                    "不再安排的景点，可不选并在修改意见中描述新偏好",
                    attraction_options,
                    default=previous_exclusions,
                    key=f"excluded_attractions_{revision_key}",
                )
            elif revision_scope == "hotel":
                st.caption("会排除当前酒店，重新查询住宿，并同步更新交通。")
                current_keyword = result.get("hotel_search_keyword", "酒店")
                revision_parameters["hotel_keyword"] = st.text_input(
                    "酒店品牌或名称关键词，可留空",
                    value="" if current_keyword == "酒店" else current_keyword,
                    placeholder="例如：全季、如家",
                    key=f"hotel_keyword_{revision_key}",
                )
            elif revision_scope == "transport":
                st.caption("保留景点和酒店，重新查询交通。城际搜索不提供实时库存或订票。")
                city_options = {"公共交通": "transit", "自驾": "driving", "步行": "walking"}
                city_values = list(city_options.values())
                current_city = result.get("city_transport_mode", "transit")
                city_label = st.selectbox(
                    "市内交通方式",
                    list(city_options),
                    index=city_values.index(current_city) if current_city in city_values else 0,
                    key=f"city_mode_{revision_key}",
                )
                intercity_options = {"高铁 / 火车": "train", "飞机": "flight"}
                intercity_values = list(intercity_options.values())
                current_intercity = result.get("intercity_transport_mode", "train")
                intercity_label = st.selectbox(
                    "城际交通方式（去程和返程）",
                    list(intercity_options),
                    index=(intercity_values.index(current_intercity)
                           if current_intercity in intercity_values else 0),
                    key=f"intercity_mode_{revision_key}",
                )
                revision_parameters.update({
                    "city_transport_mode": city_options[city_label],
                    "intercity_transport_mode": intercity_options[intercity_label],
                })
                revision_parameters["allow_walking_fallback"] = st.checkbox(
                    "修改后允许步行兜底", value=result.get("allow_walking_fallback", True),
                    key=f"allow_walking_{revision_key}",
                )
                for field, label, divisor, default in (
                    ("max_walking_distance_m", "修改后单段步行上限（公里）", 1000, 2000),
                    ("max_walking_duration_s", "修改后单段步行上限（分钟）", 60, 1800),
                    ("max_total_walking_distance_m", "修改后累计步行上限（公里）", 1000, 4000),
                    ("max_total_walking_duration_s", "修改后累计步行上限（分钟）", 60, 3600),
                ):
                    revision_parameters[field] = st.number_input(
                        label, min_value=0.1,
                        value=float(result.get(field, default)) / divisor,
                        step=0.1, key=f"{field}_{revision_key}",
                    ) * divisor
            else:
                st.caption("只调整行程节奏或说明。要换景点、酒店或交通，请选择对应类型。")

            feedback = st.text_area(
                "请输入修改意见",
                placeholder="例如：我不想去回民街；酒店换成全季；行程不要太赶……",
                key=f"feedback_text_{revision_key}",
            )

            submit_feedback = st.button(
                "🔄 提交修改意见",
                use_container_width=True,
                key="submit_feedback",
            )

            if submit_feedback:
                if not feedback.strip():
                    st.warning("请输入修改意见")
                else:
                    details = [f"修改类型：{revision_label}", feedback.strip()]
                    for key, value in revision_parameters.items():
                        details.append(f"{key}：{value}")
                    decision = {
                        "approved": False,
                        "comment": "\n".join(details),
                        "revision_scope": revision_scope,
                        **revision_parameters,
                    }
                    try:
                        # Validate before writing the audit record or resuming the graph.
                        build_revision_update(result, decision)
                        st.session_state.pending_generation = True
                        with st.spinner("正在按修改类型更新旅游方案..."):
                            result = _resume_graph(**decision)
                        _stage_candidate(result)
                        st.rerun()

                    except Exception as e:
                        if st.session_state.pending_generation:
                            st.session_state.generation_failed = True
                            st.session_state.last_generation_error = f"重新生成方案失败：{e}"
                        st.error(f"重新生成方案失败：{e}")
                        if st.session_state.generation_failed:
                            st.rerun()


if result and not st.session_state.running and not st.session_state.generation_failed:
    st.divider()
    st.success("🎉 当前旅游方案已最终确认")

    if st.session_state.version > 0:
        st.write(f"最终方案版本：**Version {st.session_state.version}**")
