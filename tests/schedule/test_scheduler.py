from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from zeus.clock import FakeClock
from zeus.memory.store import Store
from zeus.schedule.cron import hhmm_to_cron
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


def test_same_local_day_uses_local_not_utc_date(scheduler):
    sched, store, clock = scheduler
    schedule = hhmm_to_cron("00:30")  # fires at 00:30 Lagos local, daily
    sched.register("checkin_midnight", schedule, lambda when: None)

    # Heartbeat at 21:00 Lagos on 2026-08-05 (20:00 UTC).
    clock.advance(timedelta(hours=8))
    store.set_heartbeat()
    # Now at 01:15 Lagos on 2026-08-06 (00:15 UTC): same LOCAL day as the
    # missed occurrence (2026-08-06 00:30 Lagos == 2026-08-05 23:30 UTC),
    # but a different UTC calendar day (2026-08-05 vs 2026-08-06). A
    # comparison that skips .astimezone(tz) and compares raw UTC dates
    # would wrongly say False here.
    clock.advance(timedelta(hours=4, minutes=15))

    missed = sched.catch_up()
    assert len(missed) == 1
    assert missed[0].scheduled_for == datetime(2026, 8, 5, 23, 30, tzinfo=timezone.utc)
    assert missed[0].same_local_day is True


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


def test_failing_handler_does_not_abort_the_sweep(scheduler):
    sched, store, _ = scheduler
    ran: list[datetime] = []

    def raise_error(when):
        raise RuntimeError("boom")

    # Registration order matters: the raising job must run first so the
    # test actually proves the sweep continues past it.
    sched.register("checkin_a", "0 11 * * *", raise_error)
    sched.register("checkin_b", "0 11 * * *", ran.append)

    # First call establishes the baseline for both; nothing has fired yet.
    assert sched.run_pending(NOW) == []
    # Advance past tomorrow's 11:00 Lagos (== 10:00 UTC).
    later = datetime(2026, 8, 6, 10, 30, tzinfo=timezone.utc)

    fired = sched.run_pending(later)

    assert fired == ["checkin_a", "checkin_b"]
    assert len(ran) == 1  # checkin_b's handler ran despite checkin_a's raising

    by_name = {job.name: job for job in store.jobs()}
    assert by_name["checkin_a"].last_run_at is not None


def test_seconds_until_next_is_capped_at_sixty(scheduler):
    sched, _, _ = scheduler
    sched.register("checkin_morning", "0 11 * * *", lambda when: None)
    assert sched.seconds_until_next(NOW) == 60.0


def test_seconds_until_next_when_no_jobs(scheduler):
    sched, _, _ = scheduler
    assert sched.seconds_until_next(NOW) == 60.0


def test_seconds_until_next_returns_real_interval_under_the_cap(scheduler):
    sched, _, _ = scheduler
    sched.register("every_minute", "* * * * *", lambda when: None)
    # 20 seconds before the next minute boundary.
    now = datetime(2026, 8, 5, 12, 0, 40, tzinfo=timezone.utc)
    result = sched.seconds_until_next(now)
    assert 0 < result < 60
    assert result == pytest.approx(20.0)
