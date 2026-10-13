"""Shared, deterministic lodging and transit-fallback policies."""

from datetime import date
import math


DEFAULT_MAX_WALKING_DISTANCE_M = 2000.0
DEFAULT_MAX_WALKING_DURATION_S = 1800.0
DEFAULT_MAX_TOTAL_WALKING_DISTANCE_M = 4000.0
DEFAULT_MAX_TOTAL_WALKING_DURATION_S = 3600.0


def requires_hotel(state: dict) -> bool:
    days = state.get("days")
    if days is None and state.get("departure_date") and state.get("return_date"):
        start = date.fromisoformat(str(state["departure_date"]))
        end = date.fromisoformat(str(state["return_date"]))
        days = (end - start).days + 1
    if days is None:
        # Partial node updates lack request fields; retain the old validation default.
        return True
    if isinstance(days, bool) or int(days) != float(days) or int(days) < 1:
        raise ValueError("Travel days must be a positive integer")
    return int(days) > 1


def skipped_hotel_update() -> dict:
    return {
        "needs_hotel": False,
        "hotel_status": "skipped",
        "hotel_name": "",
        "hotel_result": "一日游当天往返，不查询或安排酒店。",
    }


def walking_policy(state: dict) -> dict:
    allowed = state.get("allow_walking_fallback", True)
    if type(allowed) is not bool:
        raise ValueError("allow_walking_fallback must be a boolean")
    limits = {}
    for key, default in (
        ("max_walking_distance_m", DEFAULT_MAX_WALKING_DISTANCE_M),
        ("max_walking_duration_s", DEFAULT_MAX_WALKING_DURATION_S),
        ("max_total_walking_distance_m", DEFAULT_MAX_TOTAL_WALKING_DISTANCE_M),
        ("max_total_walking_duration_s", DEFAULT_MAX_TOTAL_WALKING_DURATION_S),
    ):
        raw = state.get(key, default)
        if isinstance(raw, bool):
            raise ValueError(f"{key} must be a positive finite number")
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a positive finite number")
        limits[key] = value
    return {"allow_walking_fallback": allowed, **limits}
