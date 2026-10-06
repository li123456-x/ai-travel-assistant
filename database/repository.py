from datetime import datetime

from config.database import get_db
from database.models import (
    AIReview,
    AgentRun,
    ExecutionStep,
    HumanReview,
    TravelPlan,
    TravelSession,
)


def _commit_refresh(db, obj):
    db.commit()
    db.refresh(obj)
    return obj


def create_travel_session(thread_id: str, state: dict):
    db = get_db()
    try:
        travel_session = TravelSession(
            thread_id=thread_id,
            user_request=state["user_request"],
            origin=state["origin"],
            destination=state["destination"],
            departure_date=state["departure_date"],
            return_date=state["return_date"],
            days=state["days"],
            people=state["people"],
            budget=state["budget"],
            preferences=",".join(state.get("preferences", [])),
            constraints=",".join(state.get("constraints", [])),
            status="planning",
        )
        db.add(travel_session)
        _commit_refresh(db, travel_session)
        return travel_session.id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def complete_travel_session(session_id: int):
    db = get_db()
    try:
        travel_session = db.get(TravelSession, session_id)
        if travel_session is None:
            return False

        travel_session.status = "completed"
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_travel_plan(session_id: int, state: dict, version: int):
    db = get_db()
    try:
        plan = TravelPlan(
            session_id=session_id,
            version=version,
            attractions_result=state.get("attractions_result", ""),
            hotel_result=state.get("hotel_result", ""),
            transport_result=state.get("transport_result", ""),
            planner_result=state.get("planner_result", ""),
            is_final=False,
        )
        db.add(plan)
        _commit_refresh(db, plan)
        return plan.id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def mark_plan_final(plan_id: int):
    db = get_db()
    try:
        plan = db.get(TravelPlan, plan_id)
        if plan is None:
            return False

        (
            db.query(TravelPlan)
            .filter(TravelPlan.session_id == plan.session_id)
            .update({"is_final": False})
        )

        plan.is_final = True
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_ai_review(plan_id: int, state: dict):
    db = get_db()
    try:
        review = AIReview(
            plan_id=plan_id,
            review_passed=state.get("review_passed", False),
            review_result=state.get("review_result", ""),
        )
        db.add(review)
        _commit_refresh(db, review)
        return review.id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_human_review(plan_id: int, approved: bool, comment: str):
    db = get_db()
    try:
        review = HumanReview(
            plan_id=plan_id,
            approved=approved,
            comment=comment,
        )
        db.add(review)
        _commit_refresh(db, review)
        return review.id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_agent_run(
    db,
    session_id,
    thread_id,
    agent_name,
    trigger_reason=None,
):
    run = AgentRun(
        session_id=session_id,
        thread_id=thread_id,
        agent_name=agent_name,
        trigger_reason=trigger_reason,
        started_at=datetime.now(),
        success=False,
        retry_count=0,
    )

    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def finish_agent_run(
    db,
    run_id,
    duration,
    success,
    retry_count=0,
    error_message=None,
):
    run = db.query(AgentRun).filter(AgentRun.id == run_id).first()

    if run is None:
        return None

    run.finished_at = datetime.now()
    run.duration = duration
    run.success = success
    run.retry_count = retry_count
    run.error_message = error_message

    db.commit()
    db.refresh(run)
    return run


def create_execution_step(
    db,
    agent_run_id,
    step_type,
    step_name,
    duration,
    success=True,
    error_message=None,
):
    step = ExecutionStep(
        agent_run_id=agent_run_id,
        step_type=step_type,
        step_name=step_name,
        duration=duration,
        success=success,
        error_message=error_message,
    )

    db.add(step)
    db.commit()
    db.refresh(step)
    return step


def record_execution_step_safe(
    agent_run_id,
    step_type,
    step_name,
    duration,
    success=True,
    error_message=None,
):
    if agent_run_id is None:
        print(f"[Harness] 未获取 agent_run_id，跳过 Step 记录：{step_name}")
        return None

    db = get_db()
    try:
        return create_execution_step(
            db=db,
            agent_run_id=agent_run_id,
            step_type=step_type,
            step_name=step_name,
            duration=duration,
            success=success,
            error_message=error_message,
        )
    except Exception as e:
        db.rollback()
        print(f"[Harness] Step 日志写入失败：{step_name}，错误：{e}")
        return None
    finally:
        db.close()


def get_plan_versions(session_id: int):
    db = get_db()
    try:
        plans = (
            db.query(TravelPlan)
            .filter(TravelPlan.session_id == session_id)
            .order_by(TravelPlan.version.asc())
            .all()
        )

        result = []

        for plan in plans:
            latest_ai_review = (
                db.query(AIReview)
                .filter(AIReview.plan_id == plan.id)
                .order_by(AIReview.id.desc())
                .first()
            )

            latest_human_review = (
                db.query(HumanReview)
                .filter(HumanReview.plan_id == plan.id)
                .order_by(HumanReview.id.desc())
                .first()
            )

            result.append({
                "plan_id": plan.id,
                "version": plan.version,
                "planner_result": plan.planner_result,
                "review_passed": (
                    latest_ai_review.review_passed
                    if latest_ai_review
                    else None
                ),
                "review_result": (
                    latest_ai_review.review_result
                    if latest_ai_review
                    else ""
                ),
                "human_approved": (
                    latest_human_review.approved
                    if latest_human_review
                    else None
                ),
                "human_comment": (
                    latest_human_review.comment
                    if latest_human_review
                    else ""
                ),
                "is_final": plan.is_final,
                "created_at": plan.created_at,
            })

        return result
    finally:
        db.close()


def get_next_plan_version(session_id: int) -> int:
    db = get_db()
    try:
        latest_plan = (
            db.query(TravelPlan)
            .filter(TravelPlan.session_id == session_id)
            .order_by(TravelPlan.version.desc())
            .first()
        )

        if latest_plan is None:
            return 1

        return latest_plan.version + 1
    finally:
        db.close()


def create_manual_travel_plan(
    session_id: int,
    planner_result: str,
    version: int | None = None,
):
    db = get_db()
    try:
        if version is None:
            latest_plan = (
                db.query(TravelPlan)
                .filter(TravelPlan.session_id == session_id)
                .order_by(TravelPlan.version.desc())
                .first()
            )
            version = 1 if latest_plan is None else latest_plan.version + 1

        plan = TravelPlan(
            session_id=session_id,
            version=version,
            attractions_result="",
            hotel_result="",
            transport_result="",
            planner_result=planner_result,
            is_final=False,
        )
        db.add(plan)
        _commit_refresh(db, plan)
        return plan.id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def delete_travel_plan(plan_id: int):
    db = get_db()
    try:
        plan = db.get(TravelPlan, plan_id)
        if plan is None:
            return False

        db.delete(plan)
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
