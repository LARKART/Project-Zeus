from datetime import datetime, timezone

import pytest

from zeus.brain.tools import build_tool_callables, build_tools, logged_tool
from zeus.clock import FakeClock
from zeus.memory.journal import Journal
from zeus.memory.store import Store
from zoneinfo import ZoneInfo

START = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def wiring(tmp_path):
    clock = FakeClock(START)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, ZoneInfo("Africa/Lagos"))
    conv = store.start_conversation("schedule")
    return store, journal, conv


def test_logged_tool_records_a_successful_action(wiring):
    store, _, conv = wiring
    wrapped = logged_tool(store, conv, "demo", lambda value: f"got {value}")
    assert wrapped(value="x") == "got x"

    action = store.recent_actions()[0]
    assert action.tool == "demo"
    assert action.ok is True
    assert action.args == {"value": "x"}
    assert action.error is None


def test_logged_tool_captures_failures_without_raising(wiring):
    store, _, conv = wiring

    def explode():
        raise RuntimeError("nope")

    wrapped = logged_tool(store, conv, "boom", explode)
    result = wrapped()
    assert "nope" in result

    action = store.recent_actions()[0]
    assert action.ok is False
    assert "nope" in action.error


def test_save_goal_writes_goal_and_journal(wiring):
    store, journal, conv = wiring
    tools = build_tool_callables(store, journal, conv, "2026-08-05")
    tools["save_goal"](text="Finish the auth flow")

    assert store.get_goal("2026-08-05").text == "Finish the auth flow"
    assert "Finish the auth flow" in journal.read("2026-08-05")
    assert store.recent_actions()[0].tool == "save_goal"


def test_record_outcome_updates_status(wiring):
    store, journal, conv = wiring
    tools = build_tool_callables(store, journal, conv, "2026-08-05")
    tools["save_goal"](text="Ship it")
    tools["record_outcome"](status="partial", notes="tests missing")

    goal = store.get_goal("2026-08-05")
    assert goal.status == "partial"
    assert goal.notes == "tests missing"


def test_record_outcome_without_a_goal_is_reported_not_raised(wiring):
    store, journal, conv = wiring
    tools = build_tool_callables(store, journal, conv, "2026-08-05")
    result = tools["record_outcome"](status="done")
    assert "no goal" in result.lower()
    assert store.recent_actions()[0].ok is True


def test_record_outcome_rejects_an_invalid_status(wiring):
    store, journal, conv = wiring
    tools = build_tool_callables(store, journal, conv, "2026-08-05")
    tools["save_goal"](text="Ship it")
    result = tools["record_outcome"](status="banana")
    assert "banana" in result
    assert store.get_goal("2026-08-05").status == "pending"


def test_callables_and_decorated_tools_cover_the_same_surface(wiring):
    """build_tools wraps exactly the callables build_tool_callables exposes."""
    store, journal, conv = wiring
    assert set(build_tool_callables(store, journal, conv, "2026-08-05")) == {
        "save_goal", "record_outcome",
    }
    assert len(build_tools(store, journal, conv, "2026-08-05")) == 2
