from datetime import datetime, timezone

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


def test_checkin_lifecycle(store):
    cid = store.open_checkin("morning", START)
    assert store.get_checkin(cid).outcome == "deferred"
    store.update_checkin(cid, outcome="answered", attempts=1, fired_at=START)
    checkin = store.get_checkin(cid)
    assert checkin.outcome == "answered"
    assert checkin.attempts == 1
    assert checkin.fired_at == START


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
