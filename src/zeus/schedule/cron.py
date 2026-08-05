"""Cron expression handling. See spec §9.1.

Cron is used rather than a simple 'daily at HH:MM' format because later
slices need interval cadences such as '*/30 * * * *' that a daily format
cannot express.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from croniter import croniter

_HHMM = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def hhmm_to_cron(hhmm: str) -> str:
    """Convert the user-facing 'HH:MM' config value to a cron expression."""
    match = _HHMM.match(hhmm)
    if not match:
        raise ValueError(f"invalid time of day: {hhmm!r} (expected 'HH:MM')")
    hour, minute = match.groups()
    return f"{int(minute)} {int(hour)} * * *"


def next_occurrence(expression: str, after_utc: datetime, tz: ZoneInfo) -> datetime:
    """Next firing strictly after `after_utc`, evaluated in local wall-clock."""
    local = after_utc.astimezone(tz)
    nxt = croniter(expression, local).get_next(datetime)
    return nxt.astimezone(timezone.utc)


def occurrences_between(
    expression: str, start_utc: datetime, end_utc: datetime, tz: ZoneInfo
) -> list[datetime]:
    """All firings in (start_utc, end_utc], evaluated in local wall-clock."""
    found: list[datetime] = []
    cursor = start_utc
    while True:
        cursor = next_occurrence(expression, cursor, tz)
        if cursor > end_utc:
            return found
        found.append(cursor)
