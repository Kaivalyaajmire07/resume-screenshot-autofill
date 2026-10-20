"""Retention policy: by default only today's history is kept (the original behaviour)."""
from __future__ import annotations

import time
from datetime import date, timedelta

from . import config
from .db import HistoryRepository

_last_run: dict[int, float] = {}


def cutoff_day(today: date | None = None, keep_days: int | None = None) -> date:
    """Records strictly older than this day are deleted."""
    today = today or date.today()
    keep = config.RETENTION_DAYS if keep_days is None else keep_days
    return today - timedelta(days=max(0, keep))


def purge_expired(repo: HistoryRepository, today: date | None = None, keep_days: int | None = None,
                  min_interval: float = 0.0) -> int:
    """Delete expired records; ``min_interval`` throttles calls made on every Streamlit rerun."""
    if min_interval:
        now, key = time.monotonic(), id(repo)
        if now - _last_run.get(key, -1e9) < min_interval:
            return 0
        _last_run[key] = now
    return repo.delete_before(cutoff_day(today, keep_days))
