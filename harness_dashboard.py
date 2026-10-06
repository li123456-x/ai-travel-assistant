import pandas as pd
import streamlit as st

from database.harness_stats import (
    get_harness_summary,
    get_agent_stats,
    get_step_stats,
    get_recent_failures,
    get_trigger_reason_stats,
    get_recent_agent_runs,
)


st.set_page_config(
    page_title="Harness Dashboard",
    page_icon="📊",
    layout="wide",
)


st.title("📊 Agent Harness Dashboard")

st.caption(
    "用于查看多智能体运行状态、性能、失败、重试和触发原因。"
)


# ==================================================
# 刷新
# ==================================================

if st.button("刷新数据"):
    st.rerun()


# ==================================================
# 总览
# ==================================================

summary = get_harness_summary()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Agent运行次数",
    summary["total_runs"],
)

col2.metric(
    "成功率",
    f'{summary["success_rate"]:.2f}%',
)

col3.metric(
    "失败次数",
    summary["failed_runs"],
)

col4.metric(
    "平均耗时",
    f'{summary["avg_duration"]:.2f}s',
)

col5.metric(
    "总重试次数",
    summary["total_retries"],
)


col6, col7 = st.columns(2)

with col6:
    st.metric(
        "最慢 Agent",
        summary["slowest_agent"],
        f'{summary["slowest_agent_duration"]:.2f}s 平均',
    )

with col7:
    st.metric(
        "最慢 Step",
        summary["slowest_step"],
        f'{summary["slowest_step_duration"]:.2f}s 平均',
    )


st.divider()


# ==================================================
# Agent统计
# ==================================================

st.subheader("Agent 运行统计")

agent_stats = get_agent_stats()

if agent_stats:
    agent_df = pd.DataFrame(
        agent_stats
    )

    st.dataframe(
        agent_df,
        use_container_width=True,
        hide_index=True,
    )

    chart_df = (
        agent_df[
            [
                "agent_name",
                "avg_duration",
            ]
        ]
        .set_index("agent_name")
    )

    st.bar_chart(
        chart_df
    )

else:
    st.info(
        "暂无 Agent 运行数据。"
    )


st.divider()


# ==================================================
# Tool / LLM统计
# ==================================================

st.subheader("Tool / LLM Step 统计")

step_stats = get_step_stats()

if step_stats:
    step_df = pd.DataFrame(
        step_stats
    )

    st.dataframe(
        step_df,
        use_container_width=True,
        hide_index=True,
    )

    step_chart_df = (
        step_df[
            [
                "step_name",
                "avg_duration",
            ]
        ]
        .set_index("step_name")
    )

    st.bar_chart(
        step_chart_df
    )

else:
    st.info(
        "暂无 execution_steps 数据。"
    )


st.divider()


# ==================================================
# Trigger Reason
# ==================================================

left, right = st.columns(2)

with left:
    st.subheader(
        "触发原因分布"
    )

    trigger_stats = (
        get_trigger_reason_stats()
    )

    if trigger_stats:
        trigger_df = pd.DataFrame(
            trigger_stats
        )

        st.dataframe(
            trigger_df,
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            trigger_df.set_index(
                "trigger_reason"
            )
        )

    else:
        st.info(
            "暂无 trigger_reason 数据。"
        )


# ==================================================
# 最近失败
# ==================================================

with right:
    st.subheader(
        "最近失败记录"
    )

    failures = get_recent_failures(
        limit=20
    )

    if failures:
        failure_df = pd.DataFrame(
            failures
        )

        st.dataframe(
            failure_df,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.success(
            "目前没有失败记录。"
        )


st.divider()


# ==================================================
# 最近运行明细
# ==================================================

st.subheader(
    "最近 Agent Runs"
)

recent_runs = get_recent_agent_runs(
    limit=30
)

if recent_runs:
    recent_df = pd.DataFrame(
        recent_runs
    )

    st.dataframe(
        recent_df,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "暂无运行数据。"
    )
