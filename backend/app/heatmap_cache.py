from __future__ import annotations

import time
from typing import Any, Optional

# user_id -> (monotonic_ts, cells)
_cache: dict[int, tuple[float, list[Any]]] = {}


def get(user_id: int, *, max_age_s: float = 180.0) -> Optional[list[Any]]:
    hit = _cache.get(user_id)
    if hit is None:
        return None
    ts, cells = hit
    if time.monotonic() - ts > max_age_s:
        return None
    return cells


def put(user_id: int, cells: list[Any]) -> None:
    _cache[user_id] = (time.monotonic(), cells)


def invalidate(user_id: int) -> None:
    _cache.pop(user_id, None)
