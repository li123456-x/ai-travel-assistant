from sqlalchemy import func, case

from config.database import get_db
from database.models import AgentRun, ExecutionStep


def get_harness_summary():
    """
    Harness 总览指标：
    - Agent 总运行次数
    - 成功率
    - 失败次数
    - Agent 平均耗时
    - 总重试次数
    - 最慢 Agent
    - 最慢 Step
    """
    db = get_db()

    try:
        total_runs = (
            db.query(func.count(AgentRun.id))
            .scalar()
            or 0
        )

        success_runs = (
            db.query(func.count(AgentRun.id))
            .filter(AgentRun.success.is_(True))
            .scalar()
            or 0
        )

        failed_runs = (
            db.query(func.count(AgentRun.id))
            .filter(AgentRun.success.is_(False))
            .scalar()
            or 0
        )

        avg_duration = (
            db.query(func.avg(AgentRun.duration))
            .filter(AgentRun.duration.isnot(None))
            .scalar()
            or 0
        )

        total_retries = (
            db.query(
                func.coalesce(
                    func.sum(AgentRun.retry_count),
                    0,
                )
            )
            .scalar()
            or 0
        )

        slowest_agent = (
            db.query(
                AgentRun.agent_name,
                func.avg(AgentRun.duration).label(
                    "avg_duration"
                ),
            )
            .filter(AgentRun.duration.isnot(None))
            .group_by(AgentRun.agent_name)
            .order_by(
                func.avg(AgentRun.duration).desc()
            )
            .first()
        )

        slowest_step = (
            db.query(
                ExecutionStep.step_name,
                func.avg(
                    ExecutionStep.duration
                ).label("avg_duration"),
            )
            .filter(
                ExecutionStep.duration.isnot(None)
            )
            .group_by(
                ExecutionStep.step_name
            )
            .order_by(
                func.avg(
                    ExecutionStep.duration
                ).desc()
            )
            .first()
        )

        success_rate = (
            success_runs / total_runs * 100
            if total_runs
            else 0
        )

        return {
            "total_runs": int(total_runs),
            "success_rate": round(
                float(success_rate),
                2,
            ),
            "failed_runs": int(failed_runs),
            "avg_duration": round(
                float(avg_duration),
                2,
            ),
            "total_retries": int(total_retries),
            "slowest_agent": (
                slowest_agent.agent_name
                if slowest_agent
                else "-"
            ),
            "slowest_agent_duration": round(
                float(
                    slowest_agent.avg_duration
                ),
                2,
            )
            if slowest_agent
            else 0,
            "slowest_step": (
                slowest_step.step_name
                if slowest_step
                else "-"
            ),
            "slowest_step_duration": round(
                float(
                    slowest_step.avg_duration
                ),
                2,
            )
            if slowest_step
            else 0,
        }

    finally:
        db.close()


def get_agent_stats():
    """
    每个 Agent 的运行统计。
    """
    db = get_db()

    try:
        rows = (
            db.query(
                AgentRun.agent_name.label(
                    "agent_name"
                ),
                func.count(
                    AgentRun.id
                ).label("runs"),
                func.avg(
                    AgentRun.duration
                ).label("avg_duration"),
                func.sum(
                    case(
                        (
                            AgentRun.success.is_(True),
                            1,
                        ),
                        else_=0,
                    )
                ).label("success_runs"),
                func.coalesce(
                    func.sum(
                        AgentRun.retry_count
                    ),
                    0,
                ).label("retries"),
            )
            .group_by(
                AgentRun.agent_name
            )
            .order_by(
                func.avg(
                    AgentRun.duration
                ).desc()
            )
            .all()
        )

        result = []

        for row in rows:
            runs = int(row.runs or 0)
            success_runs = int(
                row.success_runs or 0
            )

            result.append(
                {
                    "agent_name":
                        row.agent_name,
                    "runs": runs,
                    "avg_duration": round(
                        float(
                            row.avg_duration
                            or 0
                        ),
                        2,
                    ),
                    "success_rate": round(
                        (
                            success_runs
                            / runs
                            * 100
                        )
                        if runs
                        else 0,
                        2,
                    ),
                    "retries": int(
                        row.retries or 0
                    ),
                }
            )

        return result

    finally:
        db.close()


def get_step_stats():
    """
    Tool / LLM Step 统计。
    """
    db = get_db()

    try:
        rows = (
            db.query(
                ExecutionStep.step_type.label(
                    "step_type"
                ),
                ExecutionStep.step_name.label(
                    "step_name"
                ),
                func.count(
                    ExecutionStep.id
                ).label("runs"),
                func.avg(
                    ExecutionStep.duration
                ).label("avg_duration"),
                func.sum(
                    case(
                        (
                            ExecutionStep.success.is_(
                                True
                            ),
                            1,
                        ),
                        else_=0,
                    )
                ).label("success_runs"),
            )
            .group_by(
                ExecutionStep.step_type,
                ExecutionStep.step_name,
            )
            .order_by(
                func.avg(
                    ExecutionStep.duration
                ).desc()
            )
            .all()
        )

        result = []

        for row in rows:
            runs = int(row.runs or 0)
            success_runs = int(
                row.success_runs or 0
            )

            result.append(
                {
                    "step_type":
                        row.step_type,
                    "step_name":
                        row.step_name,
                    "runs": runs,
                    "avg_duration": round(
                        float(
                            row.avg_duration
                            or 0
                        ),
                        2,
                    ),
                    "success_rate": round(
                        (
                            success_runs
                            / runs
                            * 100
                        )
                        if runs
                        else 0,
                        2,
                    ),
                }
            )

        return result

    finally:
        db.close()


def get_recent_failures(limit=20):
    """
    最近 Agent 失败记录。
    """
    db = get_db()

    try:
        rows = (
            db.query(AgentRun)
            .filter(
                AgentRun.success.is_(False)
            )
            .order_by(
                AgentRun.id.desc()
            )
            .limit(limit)
            .all()
        )

        return [
            {
                "id": row.id,
                "session_id":
                    row.session_id,
                "thread_id":
                    row.thread_id,
                "agent_name":
                    row.agent_name,
                "trigger_reason":
                    row.trigger_reason,
                "retry_count":
                    row.retry_count,
                "duration": round(
                    float(
                        row.duration or 0
                    ),
                    2,
                ),
                "error_message":
                    row.error_message,
                "started_at":
                    row.started_at,
            }
            for row in rows
        ]

    finally:
        db.close()


def get_trigger_reason_stats():
    """
    trigger_reason 分布。
    """
    db = get_db()

    try:
        rows = (
            db.query(
                func.coalesce(
                    AgentRun.trigger_reason,
                    "unknown",
                ).label(
                    "trigger_reason"
                ),
                func.count(
                    AgentRun.id
                ).label("runs"),
            )
            .group_by(
                func.coalesce(
                    AgentRun.trigger_reason,
                    "unknown",
                )
            )
            .order_by(
                func.count(
                    AgentRun.id
                ).desc()
            )
            .all()
        )

        return [
            {
                "trigger_reason":
                    row.trigger_reason,
                "runs":
                    int(row.runs or 0),
            }
            for row in rows
        ]

    finally:
        db.close()


def get_recent_agent_runs(limit=30):
    """
    最近 AgentRun，用于 Dashboard 明细查看。
    """
    db = get_db()

    try:
        rows = (
            db.query(AgentRun)
            .order_by(
                AgentRun.id.desc()
            )
            .limit(limit)
            .all()
        )

        return [
            {
                "id": row.id,
                "session_id":
                    row.session_id,
                "agent_name":
                    row.agent_name,
                "trigger_reason":
                    row.trigger_reason,
                "duration": round(
                    float(
                        row.duration or 0
                    ),
                    2,
                ),
                "success":
                    row.success,
                "retry_count":
                    row.retry_count,
                "error_message":
                    row.error_message,
                "started_at":
                    row.started_at,
            }
            for row in rows
        ]

    finally:
        db.close()
