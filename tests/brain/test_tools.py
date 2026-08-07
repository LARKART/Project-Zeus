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


def test_logged_tool_reports_failures_to_the_model_as_errors(wiring):
    """B7: a failing tool used to RETURN its error as a string, which the
    Tool Runner cannot tell apart from a successful result -- so it emitted
    a tool_result with no `is_error` and the model was told the call had
    worked. The runner sets is_error for a raised ToolError and uses its
    content verbatim, so the model still reads the friendly message.

    The action log was always correct; this test pins BOTH halves so the
    fix cannot regress into "loud to the model, silent in the log"."""
    from anthropic.lib.tools import ToolError

    store, _, conv = wiring

    def explode():
        raise RuntimeError("nope")

    wrapped = logged_tool(store, conv, "boom", explode)
    with pytest.raises(ToolError, match="nope") as raised:
        wrapped()
    assert "The boom tool failed" in str(raised.value)

    action = store.recent_actions()[0]
    assert action.ok is False
    assert "nope" in action.error


def test_a_raised_tool_error_is_what_the_runner_marks_as_an_error(wiring):
    """The SDK contract this depends on, pinned against the installed SDK:
    the runner catches ToolError specially and uses its `content` rather
    than repr(exc). A stub asserting our own call would prove nothing."""
    from anthropic.lib.tools import ToolError

    error = ToolError("The save_goal tool failed: disk full")
    assert error.content == "The save_goal tool failed: disk full"


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


def test_save_goal_refuses_an_empty_goal(wiring):
    """B8: text="" was stored and journalled as "Goal set: " -- a goal row
    that reads as an answered morning check-in while carrying nothing, and
    that the evening check-in then recalls back as an empty sentence."""
    store, journal, conv = wiring
    tools = build_tool_callables(store, journal, conv, "2026-08-05")

    result = tools["save_goal"](text="   ")

    assert store.get_goal("2026-08-05") is None
    assert "Goal set:" not in journal.read("2026-08-05")
    assert "no goal text" in result.lower()


def test_save_goal_truncates_an_enormous_goal(wiring):
    """B8: 100,000 characters were stored whole, and whatever is stored
    replays into every later prompt and into the journal, permanently."""
    from zeus.brain.tools import MAX_GOAL_CHARS

    store, journal, conv = wiring
    tools = build_tool_callables(store, journal, conv, "2026-08-05")

    tools["save_goal"](text="x" * 100_000)

    assert len(store.get_goal("2026-08-05").text) == MAX_GOAL_CHARS


def test_save_goal_leaves_an_ordinary_goal_untouched(wiring):
    """The guard must not mangle the normal case: no truncation, and the
    surrounding whitespace a transcript often carries is stripped."""
    store, journal, conv = wiring
    tools = build_tool_callables(store, journal, conv, "2026-08-05")

    tools["save_goal"](text="  Finish the auth flow  ")

    assert store.get_goal("2026-08-05").text == "Finish the auth flow"


def test_callables_and_decorated_tools_cover_the_same_surface(wiring):
    """build_tools wraps exactly the callables build_tool_callables exposes."""
    store, journal, conv = wiring
    assert set(build_tool_callables(store, journal, conv, "2026-08-05")) == {
        "save_goal", "record_outcome",
    }
    assert len(build_tools(store, journal, conv, "2026-08-05")) == 2
