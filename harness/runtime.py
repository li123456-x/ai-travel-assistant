import time
import traceback

from config.database import get_db
from database.repository import create_agent_run, finish_agent_run


NON_RETRYABLE_ERRORS = (
    KeyError,
    TypeError,
    AttributeError,
    SyntaxError,
)

RETRYABLE_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
    "service unavailable",
    "rate limit",
    "ratelimit",
    "too many requests",
    "429",
    "502",
    "503",
    "504",
    "qps",
    "cuqps_has_exceeded_the_limit",
    "outputparser",
    "validationerror",
)


def _is_retryable_error(error: Exception) -> bool:
    if isinstance(error, NON_RETRYABLE_ERRORS):
        return False

    error_name = type(error).__name__.lower()
    error_message = str(error).lower()
    error_text = f"{error_name} {error_message}"

    return any(marker in error_text for marker in RETRYABLE_MARKERS)


def _format_error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def run_with_harness(
    agent_name,
    agent_func,
    state,
    max_retries=1,
    retry_delay=1.0,
):
    total_start = time.time()

    session_id = state.get("session_id")
    thread_id = state.get("thread_id")
    trigger_reason = state.get("trigger_reason", "initial")

    print("\n" + "=" * 60)
    print(f"[Harness] 开始运行 Agent：{agent_name}")
    print(f"[Harness] session_id：{session_id}")
    print(f"[Harness] thread_id：{thread_id}")
    print(f"[Harness] trigger_reason：{trigger_reason}")
    print("=" * 60)

    db = get_db()
    agent_run = None
    agent_state = dict(state)
    last_error = None
    retry_count = 0

    try:
        try:
            agent_run = create_agent_run(
                db=db,
                session_id=session_id,
                thread_id=thread_id,
                agent_name=agent_name,
                trigger_reason=trigger_reason,
            )
            agent_state["agent_run_id"] = agent_run.id
        except Exception as e:
            db.rollback()
            print("[Harness] Agent日志创建失败：", e)

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    retry_count += 1
                    wait_seconds = retry_delay * (2 ** (attempt - 1))
                    print(
                        f"[Harness] {agent_name} 第 {retry_count} 次重试，"
                        f"{wait_seconds:.1f}s 后执行"
                    )
                    time.sleep(wait_seconds)

                result = agent_func(agent_state)
                duration = time.time() - total_start

                if agent_run is not None:
                    try:
                        finish_agent_run(
                            db=db,
                            run_id=agent_run.id,
                            duration=duration,
                            success=True,
                            retry_count=retry_count,
                            error_message=None,
                        )
                    except Exception as e:
                        db.rollback()
                        print("[Harness] Agent日志更新失败：", e)

                print(f"[Harness] {agent_name} 执行成功")
                print(f"[Harness] retry_count：{retry_count}")
                print(f"[Harness] 总耗时：{duration:.2f}s")

                return result

            except Exception as e:
                last_error = e

                print(f"[Harness] {agent_name} 执行失败")
                print(f"[Harness] 错误类型：{type(e).__name__}")
                print(f"[Harness] 错误：{e}")
                traceback.print_exc()

                retryable = _is_retryable_error(e)
                retries_left = attempt < max_retries

                if not retryable or not retries_left:
                    if not retryable:
                        print("[Harness] 该错误判定为不可重试，停止重试")
                    break

                print("[Harness] 该错误判定为临时错误，可以重试")

        duration = time.time() - total_start

        if agent_run is not None:
            try:
                finish_agent_run(
                    db=db,
                    run_id=agent_run.id,
                    duration=duration,
                    success=False,
                    retry_count=retry_count,
                    error_message=(
                        _format_error(last_error)
                        if last_error
                        else "UnknownError"
                    ),
                )
            except Exception as e:
                db.rollback()
                print("[Harness] Agent日志更新失败：", e)

        print(f"[Harness] {agent_name} 最终执行失败")
        print(f"[Harness] retry_count：{retry_count}")

        if last_error is not None:
            raise last_error

        raise RuntimeError(f"{agent_name} 执行失败，但未捕获到异常信息")

    finally:
        db.close()
