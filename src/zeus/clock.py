"""Time handling. See spec §6.3: store UTC, schedule in local wall-clock."""
from __future__ import annotations

import os
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

_LOCALTIME = Path("/etc/localtime")


def resolve_timezone(name: str) -> ZoneInfo:
    """Resolve a config timezone value into a DST-aware ZoneInfo.

    'system' follows the /etc/localtime symlink to recover the IANA zone
    name. A fixed-offset tzinfo (what datetime.astimezone() yields) is NOT
    acceptable here: it cannot represent a future DST transition.
    """
    if name != "system":
        return ZoneInfo(name)
    try:
        parts = _LOCALTIME.resolve().parts
        index = len(parts) - 1 - parts[::-1].index("zoneinfo")
        return ZoneInfo("/".join(parts[index + 1 :]))
    except (OSError, ValueError):
        pass
    env = os.environ.get("TZ")
    if env:
        try:
            return ZoneInfo(env)
        except Exception:
            pass
    return ZoneInfo("UTC")


def to_utc_iso(dt: datetime) -> str:
    """Serialize an aware datetime as an ISO-8601 UTC string."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime rejected; all timestamps must be aware")
    return dt.astimezone(timezone.utc).isoformat()


def from_utc_iso(text: str) -> datetime:
    """Parse an ISO-8601 string back into an aware UTC datetime."""
    return datetime.fromisoformat(text).astimezone(timezone.utc)


class Clock(Protocol):
    def now_utc(self) -> datetime: ...
    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def sleep(self, seconds: float) -> None:
        _time.sleep(seconds)


class FakeClock:
    """Deterministic clock. sleep() advances time instead of blocking."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FakeClock requires an aware datetime")
        self._now = start.astimezone(timezone.utc)
        self.slept: list[float] = []

    def now_utc(self) -> datetime:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self._now += timedelta(seconds=seconds)

    def advance(self, delta: timedelta) -> None:
        self._now += delta
