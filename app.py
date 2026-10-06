import uuid
from datetime import date, timedelta

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from config.graph import builder
from database.repository import (
    complete_travel_session,
    create_ai_review,
    create_human_review,
    create_travel_plan,
    create_travel_session,
    delete_travel_plan,
    get_plan_versions,
    mark_plan_final,
)

st.set_page_config(
    page_title="AI 多智能体旅游助手",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ AI 多智能体旅游助手")
st.caption("基于 LangGraph + 多智能体 + Tool Calling + Human-in-the-loop + PostgreSQL")


if "graph" not in st.session_state:
    memory = InMemorySaver()
    st.session_state.graph = builder.compile(checkpointer=memory)


DEFAULT_SESSION_STATE = {
    "thread_id": lambda: f"travel_{uuid.uuid4().hex[:8]}",
    "result": None,
    "running": False,
    "show_feedback": False,
    "session_id": None,
    "current_plan_id": None,
    "version": 0,
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

    return {
        "session_id": None,
        "thread_id": st.session_state.thread_id,
        "trigger_reason": "initial",
        "agent_run_id": None,
        "user_request": user_request,
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date.strftime("%Y-%m-%d"),
        "return_date": return_date.strftime("%Y-%m-%d"),
        "days": int(days),
        "people": int(people),
        "budget": float(budget),
        "preferences": preferences,
        "constraints": constraints,
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
        "max_human_revisions": 3,
        "next_agent": "",
    }


def _save_plan_and_review(result):
    st.session_state.version += 1
    plan_id = create_travel_plan(
        session_id=st.session_state.session_id,
        state=result,
        version=st.session_state.version,
    )
    st.session_state.current_plan_id = plan_id
    create_ai_review(plan_id=plan_id, state=result)


def _resume_graph(approved: bool, comment: str):
    return graph.invoke(
        Command(
            resume={
                "approved": approved,
                "comment": comment,
            }
        ),
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
    generate_button = st.button("🚀 生成旅游方案", use_container_width=True)

    if st.session_state.session_id:
        st.divider()
        st.header("📚 历史版本")

        try:
            versions = get_plan_versions(st.session_state.session_id)
        except Exception as e:
            st.error(f"读取历史版本失败：{e}")
            versions = []

        for item in versions:
            title = f"Version {item['version']}"
            if item["is_final"]:
                title += "｜最终版"

            with st.expander(title, expanded=False):
                st.caption(f"Plan ID：{item['plan_id']}")
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
                    disabled=item["is_final"],
                )

                delete_version = st.button(
                    "删除此版本",
                    key=f"delete_plan_{item['plan_id']}",
                    use_container_width=True,
                )

                if set_final:
                    try:
                        mark_plan_final(item["plan_id"])
                        st.session_state.current_plan_id = item["plan_id"]
                        st.success("已设为最终版")
                        st.rerun()
                    except Exception as e:
                        st.error(f"设置最终版失败：{e}")

                if delete_version:
                    try:
                        delete_travel_plan(item["plan_id"])
                        if st.session_state.current_plan_id == item["plan_id"]:
                            st.session_state.current_plan_id = None
                        st.success("已删除该版本")
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除版本失败：{e}")


if generate_button:
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
    )

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
        with st.spinner("多智能体正在规划旅游方案..."):
            result = graph.invoke(initial_state, config=_graph_config())
    except Exception as e:
        st.error(f"LangGraph 执行失败：{e}")
        st.stop()

    try:
        _save_plan_and_review(result)
    except Exception as e:
        st.error(f"保存旅游方案失败：{e}")
        st.stop()

    st.session_state.result = result
    st.session_state.running = True
    st.session_state.show_feedback = False
    st.rerun()


result = st.session_state.result

if result:
    st.divider()
    st.subheader("🤖 多智能体执行结果")

    with st.expander("🏛️ Attraction Agent", expanded=False):
        st.markdown(result.get("attractions_result", "暂无结果"))

    with st.expander("🏨 Hotel Agent", expanded=False):
        st.markdown(result.get("hotel_result", "暂无结果"))

    with st.expander("🚇 Transport Agent", expanded=False):
        st.markdown(result.get("transport_result", "暂无结果"))

    st.subheader("📅 最终旅游方案")
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

    if st.session_state.running:
        st.divider()
        st.subheader("👤 人工审核")
        st.write("请确认是否接受当前旅游方案。")

        col1, col2 = st.columns(2)

        with col1:
            approve_button = st.button(
                "✅ 同意方案",
                use_container_width=True,
                key="approve_plan",
            )

        with col2:
            reject_button = st.button(
                "❌ 修改方案",
                use_container_width=True,
                key="reject_plan",
            )

        if approve_button:
            comment = "人工审核通过"

            try:
                create_human_review(
                    plan_id=st.session_state.current_plan_id,
                    approved=True,
                    comment=comment,
                )

                with st.spinner("正在确认最终旅游方案..."):
                    result = _resume_graph(approved=True, comment=comment)

                mark_plan_final(st.session_state.current_plan_id)
                complete_travel_session(st.session_state.session_id)

                st.session_state.result = result
                st.session_state.running = False
                st.session_state.show_feedback = False
                st.rerun()

            except Exception as e:
                st.error(f"确认方案失败：{e}")

        if reject_button:
            st.session_state.show_feedback = True
            st.rerun()

        if st.session_state.get("show_feedback", False):
            st.divider()
            st.write("### ✏️ 修改意见")

            feedback = st.text_area(
                "请输入修改意见",
                placeholder="例如：我不想去回民街；酒店换成全季；行程不要太赶……",
                key="feedback_text",
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
                    try:
                        create_human_review(
                            plan_id=st.session_state.current_plan_id,
                            approved=False,
                            comment=feedback.strip(),
                        )

                        with st.spinner("Planner 正在根据您的意见重新生成方案..."):
                            result = _resume_graph(
                                approved=False,
                                comment=feedback.strip(),
                            )

                        _save_plan_and_review(result)

                        st.session_state.result = result
                        st.session_state.show_feedback = False
                        st.rerun()

                    except Exception as e:
                        st.error(f"重新生成方案失败：{e}")


if result and not st.session_state.running:
    st.divider()
    st.success("🎉 当前旅游方案已最终确认")

    if st.session_state.version > 0:
        st.write(f"最终方案版本：**Version {st.session_state.version}**")
