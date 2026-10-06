from config.database import SessionLocal
from database.repository import create_execution_step


def save_execution_step(
    agent_run_id,
    step_type,
    step_name,
    duration,
    success=True,
    error_message=None
):
    """
    保存 Tool / LLM 执行记录。

    注意：
    Harness 日志写入失败不能影响 Agent 正常业务。
    """

    if agent_run_id is None:
        print(
            f"[Harness] 无 agent_run_id，"
            f"跳过 Step 记录：{step_name}"
        )
        return

    db = SessionLocal()

    try:

        create_execution_step(
            db=db,
            agent_run_id=agent_run_id,
            step_type=step_type,
            step_name=step_name,
            duration=duration,
            success=success,
            error_message=error_message
        )

    except Exception as e:

        db.rollback()

        print(
            f"[Harness] Step日志写入失败："
            f"{step_name}，错误：{e}"
        )

    finally:

        db.close()