from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from zeus.clock import FakeClock
from zeus.memory.store import Store

START = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "zeus.db", FakeClock(START))


def test_wal_mode_is_enabled(store):
    mode = store.connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_goal_roundtrip(store):
    store.set_goal("2026-08-05", "Finish the auth flow")
    goal = store.get_goal("2026-08-05")
    assert goal.text == "Finish the auth flow"
    assert goal.status == "pending"
    assert goal.set_at == START


def test_set_goal_replaces_same_day(store):
    store.set_goal("2026-08-05", "First")
    store.set_goal("2026-08-05", "Second")
    assert store.get_goal("2026-08-05").text == "Second"


def test_set_goal_returns_updated_row_id(store):
    # Regression for F1: cur.lastrowid is only bumped by a real INSERT, so on
    # the ON CONFLICT DO UPDATE branch it silently returns the id of whatever
    # row was last *inserted* on this connection -- not the row that was just
    # updated. Interleave a second date's insert so a stale lastrowid would
    # be caught: without the fix, re-setting A after inserting B returns B's
    # id instead of A's.
    a_id = store.set_goal("2026-08-05", "A")
    b_id = store.set_goal("2026-08-06", "B")
    updated_id = store.set_goal("2026-08-05", "A v2")
    assert updated_id == a_id
    assert updated_id != b_id
    assert store.get_goal("2026-08-05").id == a_id


def test_update_goal_status(store):
    goal_id = store.set_goal("2026-08-05", "Ship it")
    store.update_goal(goal_id, status="partial", notes="tests missing")
    goal = store.get_goal("2026-08-05")
    assert goal.status == "partial"
    assert goal.notes == "tests missing"
    assert goal.reviewed_at == START


def test_invalid_goal_status_rejected(store):
    goal_id = store.set_goal("2026-08-05", "Ship it")
    with pytest.raises(Exception):
        store.update_goal(goal_id, status="banana")


def test_update_goal_preserves_existing_notes(store):
    # Regression for F3: update_goal wrote notes unconditionally, so calling
    # it without the optional notes arg (its own default invites this)
    # clobbered notes from an earlier review.
    goal_id = store.set_goal("2026-08-05", "Ship it")
    store.update_goal(goal_id, status="partial", notes="went well")
    store.update_goal(goal_id, status="missed")
    goal = store.get_goal("2026-08-05")
    assert goal.notes == "went well"
    assert goal.status == "missed"


def test_checkin_lifecycle(store):
    cid = store.open_checkin("morning", START)
    assert store.get_checkin(cid).outcome == "deferred"
    store.update_checkin(cid, outcome="answered", attempts=1, fired_at=START)
    checkin = store.get_checkin(cid)
    assert checkin.outcome == "answered"
    assert checkin.attempts == 1
    assert checkin.fired_at == START


def test_find_open_checkin_returns_unresolved(store):
    cid = store.open_checkin("morning", START)
    found = store.find_open_checkin("morning", "2026-08-05")
    assert found is not None
    assert found.id == cid


def test_find_open_checkin_ignores_settled(store):
    answered_id = store.open_checkin("morning", START)
    store.update_checkin(answered_id, outcome="answered", attempts=1)
    assert store.find_open_checkin("morning", "2026-08-05") is None

    no_answer_id = store.open_checkin("morning", START)
    store.update_checkin(no_answer_id, outcome="no_answer", attempts=1)
    found = store.find_open_checkin("morning", "2026-08-05")
    assert found is not None
    assert found.id == no_answer_id


def test_find_open_checkin_uses_local_not_utc_date(store):
    # 21:00 local in America/New_York (EDT, UTC-4 in August) is 01:00 UTC the
    # *next* day. checkins.scheduled_for is stored as UTC, so a lookup that
    # compared date(scheduled_for) against the local calendar date would
    # miss this check-in entirely -- it must be keyed on local_date instead.
    local_dt = datetime(2026, 8, 5, 21, 0, tzinfo=ZoneInfo("America/New_York"))
    store.open_checkin("evening", local_dt)
    assert store.find_open_checkin("evening", "2026-08-05") is not None
    assert store.find_open_checkin("evening", "2026-08-06") is None


def test_action_log(store):
    store.log_action("save_goal", {"text": "x"}, {"ok": True}, True, 42)
    store.log_action("save_goal", {"text": "y"}, None, False, 7, error="boom")
    actions = store.recent_actions()
    assert len(actions) == 2
    assert actions[0].ok is False and actions[0].error == "boom"
    assert actions[1].tool == "save_goal" and actions[1].args["text"] == "x"


def test_conversation_and_messages(store):
    conv = store.start_conversation("schedule")
    store.add_message(conv, "assistant", "Morning. What's the goal?")
    store.add_message(conv, "user", "Finish auth.")
    store.end_conversation(conv)
    rows = store.messages(conv)
    assert [r.role for r in rows] == ["assistant", "user"]


def test_facts(store):
    store.set_fact("wake_hour", "07:30", source="observed")
    assert store.get_fact("wake_hour") == "07:30"
    store.set_fact("wake_hour", "08:00", source="observed")
    assert store.get_fact("wake_hour") == "08:00"


def test_jobs_and_heartbeat(store):
    store.upsert_job("checkin_morning", "0 11 * * *")
    store.upsert_job("checkin_morning", "0 9 * * *")  # idempotent update
    jobs = store.jobs()
    assert len(jobs) == 1 and jobs[0].schedule == "0 9 * * *"

    assert store.heartbeat() is None
    store.set_heartbeat()
    assert store.heartbeat() == START
