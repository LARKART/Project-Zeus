from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from zeus.clock import FakeClock
from zeus.memory.store import Store
from zeus.schedule.scheduler import Scheduler

LAGOS = ZoneInfo("Africa/Lagos")
NOW = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)  # 13:00 Lagos


@pytest.fixture
def scheduler(tmp_path):
    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    return Scheduler(store, clock, LAGOS), store, clock


def test_register_persists_the_job(scheduler):
    sched, store, _ = scheduler
    sched.register("checkin_morning", "0 11 * * *", lambda when: None)
    assert [j.name for j in store.jobs()] == ["checkin_morning"]


def test_catch_up_returns_nothing_without_a_heartbeat(scheduler):
    sched, _, _ = scheduler
    sched.register("checkin_morning", "0 11 * * *", lambda when: None)
    assert sched.catch_up() == []


def test_catch_up_flags_a_missed_run_on_the_same_local_day(scheduler):
    sched, store, clock = scheduler
    sched.register("checkin_morning", "0 11 * * *", lambda when: None)
    # Heartbeat at 09:00 Lagos today; now is 13:00 Lagos. 11:00 was missed.
    clock.advance(timedelta(hours=-4))
    store.set_heartbeat()
    clock.advance(timedelta(hours=4))

    missed = sched.catch_up()
    assert len(missed) == 1
    assert missed[0].job == "checkin_morning"
    assert missed[0].same_local_day is True


def test_catch_up_flags_a_stale_run_from_a_previous_day(scheduler):
    sched, store, clock = scheduler
    sched.register("checkin_morning", "0 11 * * *", lambda when: None)
    clock.advance(timedelta(days=-2))
    store.set_heartbeat()
    clock.advance(timedelta(days=2))

    missed = sched.catch_up()
    assert len(missed) == 2                      # two 11:00s elapsed
    assert missed[0].same_local_day is False     # the older one
    assert missed[-1].same_local_day is True     # today's


def test_run_pending_fires_the_handler_once(scheduler):
    sched, _, _ = scheduler
    fired: list[datetime] = []
    sched.register("checkin_morning", "0 11 * * *", fired.append)

    # First call establishes the baseline; nothing has come due yet.
    assert sched.run_pending(NOW) == []
    # Advance past tomorrow's 11:00 Lagos (== 10:00 UTC).
    later = datetime(2026, 8, 6, 10, 30, tzinfo=timezone.utc)
    assert sched.run_pending(later) == ["checkin_morning"]
    assert len(fired) == 1
    # Firing is not repeated for the same occurrence.
    assert sched.run_pending(later) == []


def test_seconds_until_next_is_capped_at_sixty(scheduler):
    sched, _, _ = scheduler
    sched.register("checkin_morning", "0 11 * * *", lambda when: None)
    assert sched.seconds_until_next(NOW) == 60.0


def test_seconds_until_next_when_no_jobs(scheduler):
    sched, _, _ = scheduler
    assert sched.seconds_until_next(NOW) == 60.0
