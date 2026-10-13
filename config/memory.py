"""Exact request matching for persisted, confirmed plan snapshots, not vector RAG."""

from datetime import date
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import unicodedata

from config.trip_policy import walking_policy


MEMORY_SCHEMA_VERSION = 2

INITIAL_REQUEST_FIELDS = (
    "origin", "destination", "departure_date", "return_date", "days", "people",
    "budget", "preferences", "constraints", "city_transport_mode",
    "intercity_transport_mode", "allow_walking_fallback", "max_walking_distance_m",
    "max_walking_duration_s", "max_total_walking_distance_m", "max_total_walking_duration_s",
)


def _normalized_text(value) -> str:
    if not isinstance(value, str):
        raise ValueError("Memory text fields must be strings")
    return " ".join(unicodedata.normalize("NFKC", value).strip().split()).casefold()


def _normalized_list(value, *, ordered=False) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("Memory list fields must be lists")
    result = list(dict.fromkeys(_normalized_text(item) for item in value if item))
    return result if ordered else sorted(result)


def build_memory_payload(state: dict) -> dict:
    start = date.fromisoformat(str(state["departure_date"]))
    end = date.fromisoformat(str(state["return_date"]))
    days = (end - start).days + 1
    if days < 1 or int(state.get("days", days)) != days:
        raise ValueError("Travel dates and days are inconsistent")
    people = state["people"]
    if isinstance(people, bool) or int(people) != float(people) or int(people) < 1:
        raise ValueError("People must be a positive integer")
    try:
        if isinstance(state["budget"], bool):
            raise ValueError("Budget cannot be a boolean")
        budget = Decimal(str(state["budget"]))
    except InvalidOperation as exc:
        raise ValueError("Invalid budget") from exc
    if not budget.is_finite() or budget < 0:
        raise ValueError("Budget must be a nonnegative finite number")
    origin = _normalized_text(state["origin"])
    destination = _normalized_text(state["destination"])
    if not origin or not destination:
        raise ValueError("Origin and destination are required")
    city_mode = state.get("city_transport_mode", "transit")
    intercity_mode = state.get("intercity_transport_mode", "train")
    if city_mode not in {"transit", "driving", "walking"}:
        raise ValueError("Invalid city transport mode")
    if intercity_mode not in {"train", "flight"}:
        raise ValueError("Invalid intercity transport mode")
    payload = {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "origin": origin,
        "destination": destination,
        "departure_date": start.isoformat(),
        "return_date": end.isoformat(),
        "days": days,
        "people": int(people),
        "budget": format(budget.normalize(), "f"),
        "preferences": _normalized_list(state.get("preferences", [])),
        "constraints": _normalized_list(state.get("constraints", []), ordered=True),
        "needs_hotel": days > 1,
        "city_transport_mode": city_mode,
        "intercity_transport_mode": intercity_mode,
        "walking_policy": walking_policy(state),
    }
    return payload


def capture_initial_request(state: dict) -> dict:
    build_memory_payload(state)
    return deepcopy({key: state[key] for key in INITIAL_REQUEST_FIELDS if key in state})


def request_state_from_payload(payload: dict) -> dict:
    """Decode persisted conditions without inventing missing original settings."""
    state = {key: deepcopy(payload[key]) for key in INITIAL_REQUEST_FIELDS if key in payload}
    if not isinstance(payload.get("walking_policy"), dict):
        raise ValueError("Stored request lacks its walking policy")
    state.update(payload["walking_policy"])
    build_memory_payload(state)
    return state


def memory_signature(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_memory_record(state: dict) -> dict | None:
    state = state.get("initial_request_state", state)
    if not isinstance(state, dict):
        raise ValueError("Initial request must be a dictionary")
    required = {"origin", "destination", "departure_date", "return_date", "people", "budget"}
    if not required.issubset(state):
        # Legacy/manual saves without the original query must not guess a cache key.
        return None
    payload = build_memory_payload(state)
    return {"request_signature": memory_signature(payload), "request_payload": payload}


def confirmation_signature(state: dict) -> str:
    record = build_memory_record(state)
    if record is None:
        raise ValueError("Missing initial request for final confirmation")
    return memory_signature({
        "request_signature": record["request_signature"],
        "results": {key: state.get(key) for key in (
            "attractions_result", "hotel_result", "transport_result", "planner_result",
            "review_result", "review_passed", "hotel_name", "attraction_names", "transport_segments",
        )},
        "effective_conditions": build_memory_payload(state),
        "human_feedback_history": state.get("human_feedback_history", []),
    })
