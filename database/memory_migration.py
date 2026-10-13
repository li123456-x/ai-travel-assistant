"""Upgrade old memory keys only when the original request is recoverable."""

from collections import defaultdict

from config.database import get_db
from config.memory import build_memory_payload, memory_signature, request_state_from_payload
from database.models import TravelPlan, TravelPlanMemory, TravelSession


def _initial_legacy_payload(payload: dict) -> bool:
    revisions = payload.get("revision_requests")
    return (
        payload.get("schema_version") == 1
        and isinstance(revisions, dict)
        and all(isinstance(value, str) and not value.strip() for value in revisions.values())
        and payload.get("attraction_excluded_names") == []
        and payload.get("hotel_excluded_names") == []
        and payload.get("hotel_search_keyword", "") in {"", "酒店"}
    )


def migrate_plan_memory(*, apply: bool = False) -> dict:
    db = get_db()
    report = {"applied": apply, "updated_plan_ids": [], "skipped_session_ids": []}
    try:
        rows = (
            db.query(TravelPlanMemory, TravelPlan, TravelSession)
            .join(TravelPlan, TravelPlanMemory.plan_id == TravelPlan.id)
            .join(TravelSession, TravelPlan.session_id == TravelSession.id)
            .order_by(TravelSession.id, TravelPlan.version, TravelPlan.id).all()
        )
        groups = defaultdict(list)
        for row in rows:
            groups[row[2].id].append(row)
        for session_id, records in groups.items():
            legacy = [row for row in records if row[0].request_payload.get("schema_version") == 1]
            if not legacy:
                continue
            source = next((row for row in records if _initial_legacy_payload(row[0].request_payload)), None)
            if source is None:
                report["skipped_session_ids"].append(session_id)
                continue
            memory, original_plan, session = source
            try:
                initial_state = request_state_from_payload(memory.request_payload)
                payload = build_memory_payload(initial_state)
                stored_request = {
                    **initial_state,
                    "origin": session.origin, "destination": session.destination,
                    "departure_date": session.departure_date, "return_date": session.return_date,
                    "days": session.days, "people": session.people, "budget": session.budget,
                    "preferences": [value.strip() for value in (session.preferences or "").split(",") if value.strip()],
                    "constraints": [value.strip() for value in (session.constraints or "").split(",") if value.strip()],
                }
                if payload != build_memory_payload(stored_request):
                    raise ValueError("Original snapshot disagrees with the task request")
            except (TypeError, ValueError, KeyError):
                report["skipped_session_ids"].append(session_id)
                continue
            for old_memory, plan, _ in legacy:
                report["updated_plan_ids"].append(plan.id)
                if not apply:
                    continue
                old_payload = old_memory.request_payload
                snapshot = {
                    **old_memory.result_snapshot,
                    "effective_request_payload": build_memory_payload(request_state_from_payload(old_payload)),
                    "revision_requests": old_payload.get("revision_requests", {}),
                    "attraction_excluded_names": old_payload.get("attraction_excluded_names", []),
                    "hotel_excluded_names": old_payload.get("hotel_excluded_names", []),
                    "hotel_search_keyword": old_payload.get("hotel_search_keyword", "酒店"),
                    "initial_request_source_plan_id": original_plan.id,
                }
                old_memory.result_snapshot = snapshot
                old_memory.request_payload = payload
                old_memory.request_signature = memory_signature(payload)
        if apply:
            db.commit()
        return report
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
