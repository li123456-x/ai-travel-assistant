from datetime import datetime

from sqlalchemy import func, or_

from config.database import get_db
from config.memory import (
    build_memory_payload, build_memory_record, confirmation_signature, memory_signature,
)
from database.models import (
    AIReview,
    AgentRun,
    ExecutionStep,
    HumanReview,
    TravelPlan,
    TravelPlanMemory,
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
        db.flush()
        _add_plan_memory(db, plan.id, state)
        _commit_refresh(db, plan)
        return plan.id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_plan_with_review(session_id: int, state: dict, version: int) -> int:
    """一个事务保存完整方案及其 AI 审核，避免只保存方案的半完成版本。"""
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
        # flush 获得主键，但尚未提交；后面的审核写入失败时方案也能回滚。
        db.flush()
        plan_id = plan.id
        db.add(
            AIReview(
                plan_id=plan_id,
                review_passed=state.get("review_passed", False),
                review_result=state.get("review_result", ""),
            )
        )
        _add_plan_memory(db, plan_id, state)
        db.commit()
        return plan_id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _add_plan_memory(db, plan_id: int, state: dict) -> TravelPlanMemory | None:
    record = build_memory_record(state)
    if record is None:
        return
    snapshot = {
        "days": state.get("days", record["request_payload"]["days"]),
        "needs_hotel": state.get("days", record["request_payload"]["days"]) > 1,
        "hotel_name": state.get("hotel_name", ""),
        "attraction_names": state.get("attraction_names", []),
        "transport_segments": state.get("transport_segments", []),
        "effective_request_payload": build_memory_payload(state),
        "human_feedback_history": state.get("human_feedback_history", []),
        "revision_requests": {
            scope: state.get(f"{scope}_revision_request", "")
            for scope in ("attraction", "hotel", "transport", "planner")
        },
        "attraction_excluded_names": state.get("attraction_excluded_names", []),
        "hotel_excluded_names": state.get("hotel_excluded_names", []),
        "hotel_search_keyword": state.get("hotel_search_keyword", "酒店"),
    }
    memory = TravelPlanMemory(plan_id=plan_id, result_snapshot=snapshot, **record)
    db.add(memory)
    return memory


def save_confirmed_plan(session_id: int, state: dict, comment: str) -> dict:
    """Persist one human-confirmed result atomically; retries do not add versions."""
    if state.get("human_review_passed") is not True or state.get("__interrupt__"):
        raise ValueError("只有已完成人工确认的方案才能保存")
    if not isinstance(state.get("planner_result"), str) or not state["planner_result"].strip():
        raise ValueError("最终方案内容不能为空")
    token = confirmation_signature(state)
    db = get_db()
    try:
        session = (
            db.query(TravelSession).filter(TravelSession.id == session_id)
            .with_for_update().one_or_none()
        )
        if session is None or state.get("thread_id") != session.thread_id:
            raise ValueError("旅行任务不存在或不属于当前工作流")
        final = (
            db.query(TravelPlan).filter_by(session_id=session_id, is_final=True)
            .order_by(TravelPlan.id.desc()).first()
        )
        if final and final.memory and final.memory.result_snapshot.get("confirmation_signature") == token:
            latest = db.query(HumanReview).filter_by(plan_id=final.id).order_by(HumanReview.id.desc()).first()
            if session.status == "completed" and latest and latest.approved:
                return {"plan_id": final.id, "version": final.version}
        if session.status == "completed":
            raise ValueError("该任务已完成，请创建新任务，不重复写入最终版本")
        version = (db.query(func.max(TravelPlan.version)).filter_by(session_id=session_id).scalar() or 0) + 1
        db.query(TravelPlan).filter_by(session_id=session_id).update(
            {"is_final": False}, synchronize_session=False,
        )
        plan = TravelPlan(
            session_id=session_id, version=version, is_final=True,
            **{key: state.get(key, "") for key in (
                "attractions_result", "hotel_result", "transport_result", "planner_result",
            )},
        )
        db.add(plan)
        db.flush()
        plan_id = plan.id
        db.add(AIReview(
            plan_id=plan_id, review_passed=state.get("review_passed", False),
            review_result=state.get("review_result", ""),
        ))
        db.add(HumanReview(plan_id=plan_id, approved=True, comment=comment))
        memory = _add_plan_memory(db, plan_id, state)
        memory.result_snapshot = {**memory.result_snapshot, "confirmation_signature": token}
        session.status = "completed"
        db.commit()
        return {"plan_id": plan_id, "version": version}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def find_reusable_plan(state: dict) -> dict | None:
    payload = build_memory_payload(state)
    signature = memory_signature(payload)
    db = get_db()
    try:
        latest_human_review = (
            db.query(func.max(HumanReview.id))
            .filter(HumanReview.plan_id == TravelPlan.id)
            .correlate(TravelPlan).scalar_subquery()
        )
        rows = (
            db.query(TravelPlanMemory, TravelPlan, TravelSession, HumanReview)
            .join(TravelPlan, TravelPlanMemory.plan_id == TravelPlan.id)
            .join(TravelSession, TravelPlan.session_id == TravelSession.id)
            .join(HumanReview, HumanReview.plan_id == TravelPlan.id)
            .filter(
                TravelPlanMemory.request_signature == signature,
                TravelPlan.is_final.is_(True),
                TravelSession.status == "completed",
                HumanReview.id == latest_human_review,
                HumanReview.approved.is_(True),
            )
            .order_by(TravelPlan.created_at.desc(), TravelPlan.id.desc())
        )
        for memory, plan, session, human_review in rows:
            if memory.request_payload != payload or not plan.planner_result.strip():
                continue
            latest_ai = (
                db.query(AIReview).filter(AIReview.plan_id == plan.id)
                .order_by(AIReview.id.desc()).first()
            )
            result = {
                **memory.result_snapshot,
                "attractions_result": plan.attractions_result or "",
                "hotel_result": plan.hotel_result or "",
                "transport_result": plan.transport_result or "",
                "planner_result": plan.planner_result,
                "review_passed": latest_ai.review_passed if latest_ai else False,
                "review_result": latest_ai.review_result if latest_ai else "",
            }
            return {
                "session_id": session.id, "plan_id": plan.id, "version": plan.version,
                "created_at": plan.created_at, "request_payload": payload,
                "human_comment": human_review.comment or "", "result": result,
            }
        return None
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


def get_travel_sessions(search: str = ""):
    db = get_db()
    try:
        query = db.query(TravelSession).filter(TravelSession.plans.any())
        search = search.strip()
        if search:
            query = query.filter(or_(
                TravelSession.origin.contains(search, autoescape=True),
                TravelSession.destination.contains(search, autoescape=True),
                TravelSession.user_request.contains(search, autoescape=True),
            ))
        sessions = query.order_by(
            TravelSession.created_at.desc(), TravelSession.id.desc()
        ).all()
        return [{
            "session_id": item.id,
            "origin": item.origin,
            "destination": item.destination,
            "departure_date": item.departure_date,
            "return_date": item.return_date,
            "status": item.status,
            "created_at": item.created_at,
        } for item in sessions]
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
