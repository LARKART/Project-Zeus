"""Generic recurring-job scheduler with startup catch-up. See spec §9."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from zeus.clock import Clock
from zeus.memory.store import Store
from zeus.schedule.cron import next_occurrence, occurrences_between

Handler = Callable[[datetime], None]

_MAX_SLEEP = 60.0
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MissedRun:
    job: str
    scheduled_for: datetime
    same_local_day: bool


class Scheduler:
    """Fires named jobs on cron schedules evaluated in local wall-clock time.

    Deliberately knows nothing about check-ins: Slice 4's watchman registers
    its own jobs here without modification. Policy for what to do about a
    missed run lives with the caller.
    """

    def __init__(self, store: Store, clock: Clock, tz: ZoneInfo) -> None:
        self._store = store
        self._clock = clock
        self._tz = tz
        self._handlers: dict[str, Handler] = {}
        self._schedules: dict[str, str] = {}

    def register(self, name: str, schedule: str, handler: Handler) -> None:
        self._handlers[name] = handler
        self._schedules[name] = schedule
        self._store.upsert_job(name, schedule)

    def catch_up(self) -> list[MissedRun]:
        """Occurrences that elapsed between the last heartbeat and now."""
        since = self._store.heartbeat()
        if since is None:
            return []
        now = self._clock.now_utc()
        today = now.astimezone(self._tz).date()
        missed: list[MissedRun] = []
        for name, expression in self._schedules.items():
            for when in occurrences_between(expression, since, now, self._tz):
                missed.append(
                    MissedRun(
                        job=name,
                        scheduled_for=when,
                        same_local_day=when.astimezone(self._tz).date() == today,
                    )
                )
        missed.sort(key=lambda run: run.scheduled_for)
        return missed

    def run_pending(self, now_utc: datetime) -> list[str]:
        """Fire any job whose next occurrence has arrived."""
        fired: list[str] = []
        by_name = {job.name: job for job in self._store.jobs()}
        for name, expression in self._schedules.items():
            job = by_name.get(name)
            baseline = job.last_run_at if job and job.last_run_at else None
            if baseline is None:
                # First sight of this job: set the baseline, do not fire.
                self._store.set_job_run(name, now_utc)
                continue
            due = next_occurrence(expression, baseline, self._tz)
            if due <= now_utc:
                # set_job_run BEFORE the handler, deliberately: the occurrence is
                # consumed whether or not the handler succeeds. Persisting after
                # would make a permanently-failing job re-fire on every poll.
                # Retry policy for check-ins lives in the Task 8 state machine.
                self._store.set_job_run(name, due)
                try:
                    self._handlers[name](due)
                except Exception:
                    # One job must never abort the sweep for the jobs after it.
                    _log.exception("scheduled job %r failed at %s", name, due)
                fired.append(name)
        return fired

    def seconds_until_next(self, now_utc: datetime) -> float:
        """Sleep interval, capped so clock jumps are noticed promptly."""
        if not self._schedules:
            return _MAX_SLEEP
        soonest = min(
            next_occurrence(expression, now_utc, self._tz)
            for expression in self._schedules.values()
        )
        delta = (soonest - now_utc).total_seconds()
        return max(0.0, min(delta, _MAX_SLEEP))
