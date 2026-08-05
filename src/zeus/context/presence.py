"""Context gate. See spec §8.

Every probe degrades to the *safe* value on failure (not locked, not idle,
no focus, no call) so that a broken probe never silences ZEUS permanently
and never crashes the daemon.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path

from zeus.config import ContextConfig

log = logging.getLogger(__name__)

ASSERTIONS = Path.home() / "Library/DoNotDisturb/DB/Assertions.json"
_IDLE_RE = re.compile(r'"HIDIdleTime"\s*=\s*(\d+)')


class Verdict(Enum):
    SPEAK = "speak"
    NOTIFY = "notify"
    DEFER = "defer"


@dataclass(frozen=True)
class Signals:
    screen_locked: bool
    idle: timedelta
    focus_active: bool
    call_app_running: bool


def decide(signals: Signals, idle_threshold: timedelta) -> Verdict:
    """Pure implementation of the spec §8 decision table."""
    if signals.screen_locked or signals.idle > idle_threshold:
        return Verdict.DEFER
    if signals.focus_active or signals.call_app_running:
        return Verdict.NOTIFY
    return Verdict.SPEAK


def screen_locked() -> bool:
    try:
        import Quartz  # imported lazily: pyobjc is slow to load

        session = Quartz.CGSessionCopyCurrentDictionary()
        return bool(session and session.get("CGSSessionScreenIsLocked", False))
    except Exception:
        log.debug("screen_locked probe failed", exc_info=True)
        return False


def _raw_idle_nanoseconds() -> int:
    output = subprocess.run(
        ["ioreg", "-c", "IOHIDSystem"],
        capture_output=True, text=True, timeout=5, check=True,
    ).stdout
    match = _IDLE_RE.search(output)
    if not match:
        raise ValueError("HIDIdleTime not found in ioreg output")
    return int(match.group(1))


def idle_time() -> timedelta:
    try:
        return timedelta(seconds=_raw_idle_nanoseconds() / 1_000_000_000)
    except Exception:
        log.debug("idle_time probe failed", exc_info=True)
        return timedelta(0)


def focus_active() -> bool:
    """True when a macOS Focus mode is asserted.

    RISK R3: verified that Assertions.json is ABSENT with no Focus active.
    The converse — that it is present with records while Focus IS active —
    must be confirmed manually before this probe is trusted (see Task 7,
    Step 5).
    """
    try:
        if not ASSERTIONS.exists():
            return False
        data = json.loads(ASSERTIONS.read_text())
        records = data.get("data", [{}])[0].get("storeAssertionRecords", [])
        return bool(records)
    except Exception:
        log.debug("focus_active probe failed", exc_info=True)
        return False


def call_app_running(names: list[str]) -> bool:
    for name in names:
        try:
            found = subprocess.run(
                ["pgrep", "-x", name], capture_output=True, timeout=5
            )
            if found.returncode == 0:
                return True
        except Exception:
            log.debug("call_app probe failed for %s", name, exc_info=True)
    return False


class Presence:
    def __init__(self, config: ContextConfig) -> None:
        self._config = config

    def read(self) -> Signals:
        return Signals(
            screen_locked=screen_locked(),
            idle=idle_time(),
            focus_active=focus_active(),
            call_app_running=call_app_running(self._config.call_apps),
        )

    def verdict(self) -> Verdict:
        return decide(self.read(), self._config.idle_threshold)
