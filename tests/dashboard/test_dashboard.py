"""Dashboard tests.

ON THE NETWORK CONSTRAINT (spec §13): no test here makes a network call.
The server tests bind and fetch 127.0.0.1 on an OS-assigned port — loopback
never leaves the machine, needs no interface to be up, and works with the
wifi off. That is the distinction the constraint draws: the ban is on tests
that depend on something outside this computer. Binding is also the only
way to check the one property that matters most here — that ZEUS listens on
loopback and not on the wildcard address.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from zeus.clock import FakeClock
from zeus.dashboard.data import _connect_readonly, read_snapshot
from zeus.dashboard.render import render_json, render_page
from zeus.dashboard.server import BIND_HOST, build_server
from zeus.memory.journal import Journal
from zeus.memory.store import Store

LOS_ANGELES = ZoneInfo("America/Los_Angeles")
LAGOS = ZoneInfo("Africa/Lagos")
# 2026-08-05 12:00 Los Angeles == 19:00 UTC. Deliberately a moment whose UTC
# date and local date agree, so the tests that care about the seam have to
# create the disagreement themselves rather than getting it by accident.
NOW = datetime(2026, 8, 5, 12, 0, tzinfo=LOS_ANGELES).astimezone(timezone.utc)


@pytest.fixture
def zeus(tmp_path):
    """A populated ZEUS, written through the real Store and Journal."""
    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LOS_ANGELES)

    goal_id = store.set_goal("2026-08-05", "ship the dashboard")
    store.update_goal(goal_id, "pending")
    for day, text, status in [
        ("2026-08-04", "write the retry ladder", "done"),
        ("2026-08-03", "fix the UTC seam", "done"),
        ("2026-08-02", "rest", "missed"),
        ("2026-08-01", "plan slice 1", "done"),
    ]:
        store.update_goal(store.set_goal(day, text), status, f"notes for {day}")

    morning = store.open_checkin(
        "morning", datetime(2026, 8, 5, 11, 0, tzinfo=LOS_ANGELES), "2026-08-05"
    )
    store.update_checkin(morning, outcome="answered", attempts=1, fired_at=NOW)
    evening = store.open_checkin(
        "evening", datetime(2026, 8, 5, 21, 0, tzinfo=LOS_ANGELES), "2026-08-05"
    )
    store.update_checkin(evening, outcome="deferred", attempts=1)

    conversation = store.start_conversation("schedule")
    store.add_message(conversation, "assistant", "What's the one thing today?")
    store.add_message(conversation, "user", "Ship the dashboard.")
    store.log_action(
        "save_goal", {"text": "ship the dashboard"}, "Saved today's goal",
        True, 12, conversation_id=conversation,
    )
    store.log_action(
        "record_outcome", {"status": "done"}, None, False, 4,
        error="no goal recorded for today", conversation_id=conversation,
    )
    store.upsert_job("checkin_morning", "0 11 * * *")
    store.set_heartbeat()
    journal.append("Goal set: ship the dashboard")
    store.close()
    return {"root": tmp_path, "db": tmp_path / "zeus.db",
            "journal_dir": tmp_path / "journal"}


def snapshot_of(zeus, now=NOW, tz=LOS_ANGELES):
    return read_snapshot(zeus["db"], zeus["journal_dir"], tz, now)


# ---- reading ----------------------------------------------------------
def test_the_snapshot_carries_every_section_the_page_shows(zeus):
    """"All of it" is the requirement, so the absence of a section is a bug."""
    snapshot = snapshot_of(zeus)
    assert snapshot.today_goal["text"] == "ship the dashboard"
    assert len(snapshot.goals) == 5
    assert len(snapshot.checkins) == 2
    assert len(snapshot.actions) == 2
    assert len(snapshot.conversations) == 1
    assert len(snapshot.conversations[0]["messages"]) == 2
    assert len(snapshot.jobs) == 1
    assert snapshot.journal and "ship the dashboard" in snapshot.journal[0]["body"]
    assert snapshot.health.status == "alive"


def test_a_failed_tool_call_is_shown_as_failed(zeus):
    """The action log is the spine; an error that renders as a success is
    worse than no log at all."""
    failed = [a for a in snapshot_of(zeus).actions if not a["ok"]]
    assert len(failed) == 1
    assert failed[0]["error"] == "no goal recorded for today"


def test_the_dashboard_cannot_write_to_the_database(zeus):
    """mode=ro is the guarantee, not a convention.

    The daemon owns this file. A page load that could take a write lock
    could stall the voice loop, so the ability is removed at the driver
    rather than left to this package's good behaviour.

    Goes through _connect_readonly rather than opening its own connection
    with mode=ro spelled out here. The first draft of this test did the
    latter, which asserted that SQLITE honours mode=ro — true, and nothing
    to do with this codebase. Dropping mode=ro from data.py left it green.
    """
    connection = _connect_readonly(zeus["db"])
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute(
                "INSERT INTO facts (key, value, learned_at, source) "
                "VALUES ('x', 'y', 'z', 'w')"
            )
    finally:
        connection.close()


def test_the_dashboard_reads_while_the_daemon_holds_the_database_open(zeus):
    """The real deployment shape: daemon writing, dashboard reading.

    This is what WAL is for (§6.1). Under the default rollback journal these
    two block each other and a page load stalls the voice loop.
    """
    writer = Store(zeus["db"], FakeClock(NOW))
    writer.set_goal("2026-08-06", "written while the dashboard reads")
    snapshot = read_snapshot(
        zeus["db"], zeus["journal_dir"], LOS_ANGELES,
        NOW + timedelta(days=1),
    )
    writer.close()
    assert snapshot.today_goal["text"] == "written while the dashboard reads"


def test_a_missing_database_renders_instead_of_raising(tmp_path):
    """§10: fail loudly, never pretend — and never 500.

    "Is ZEUS working?" is the first question this page is opened to answer,
    so it has to answer it on a machine where ZEUS has never run.
    """
    snapshot = read_snapshot(
        tmp_path / "absent.db", tmp_path / "journal", LOS_ANGELES, NOW
    )
    assert snapshot.health.status == "never"
    assert snapshot.goals == []
    assert "<html" in render_page(snapshot)


def test_a_database_path_containing_a_space_still_resolves(tmp_path):
    """`Project Zeus` has a space in it, and sqlite's URI parser is strict.

    An f-string here does not error — it silently opens a DIFFERENT, empty
    database, so the page renders perfectly and shows nothing at all.
    """
    spaced = tmp_path / "a directory with spaces"
    spaced.mkdir()
    store = Store(spaced / "zeus.db", FakeClock(NOW))
    store.set_goal("2026-08-05", "findable")
    store.close()
    snapshot = read_snapshot(
        spaced / "zeus.db", spaced / "journal", LOS_ANGELES, NOW
    )
    assert snapshot.today_goal["text"] == "findable"


# ---- streaks ----------------------------------------------------------
def test_todays_unreviewed_goal_does_not_break_the_streak(zeus):
    """Loaded at lunchtime, the page must not report a streak lost that the
    user is still in the middle of keeping — the evening review has not
    happened yet."""
    snapshot = snapshot_of(zeus)
    assert snapshot.today_goal["status"] == "pending"
    assert snapshot.streak.current == 2        # 08-04 and 08-03, not broken by today


def test_a_partial_day_does_not_count_toward_the_streak(tmp_path):
    """A streak that forgives `partial` measures intent, not outcome."""
    store = Store(tmp_path / "zeus.db", FakeClock(NOW))
    store.update_goal(store.set_goal("2026-08-05", "a"), "done")
    store.update_goal(store.set_goal("2026-08-04", "b"), "partial")
    store.update_goal(store.set_goal("2026-08-03", "c"), "done")
    store.close()
    snapshot = read_snapshot(tmp_path / "zeus.db", tmp_path / "j", LOS_ANGELES, NOW)
    assert snapshot.streak.current == 1
    assert snapshot.streak.longest == 1


# ---- rendering --------------------------------------------------------
def test_a_goal_containing_markup_is_escaped(tmp_path):
    """Every value on this page is transcribed speech or model output."""
    store = Store(tmp_path / "zeus.db", FakeClock(NOW))
    store.set_goal("2026-08-05", "<script>alert('xss')</script>")
    store.add_message(store.start_conversation("wake"), "user", "<img onerror=1>")
    store.close()
    page = render_page(
        read_snapshot(tmp_path / "zeus.db", tmp_path / "j", LOS_ANGELES, NOW)
    )
    assert "<script>alert" not in page
    assert "&lt;script&gt;alert" in page
    assert "<img onerror" not in page


def test_timestamps_render_in_the_users_zone_not_utc(tmp_path):
    """§6.3: stored UTC, displayed local.

    A 21:00 evening check-in in Los Angeles is 04:00 UTC the NEXT day. A
    page showing the raw value reports the evening ritual as having fired
    in the small hours — the same seam that has produced six defects here.
    """
    store = Store(tmp_path / "zeus.db", FakeClock(NOW))
    evening_local = datetime(2026, 8, 5, 21, 0, tzinfo=LOS_ANGELES)
    checkin = store.open_checkin("evening", evening_local, "2026-08-05")
    store.update_checkin(checkin, outcome="answered", attempts=1,
                         fired_at=evening_local)
    store.close()
    page = render_page(
        read_snapshot(tmp_path / "zeus.db", tmp_path / "j", LOS_ANGELES, NOW)
    )
    assert "2026-08-05 21:00" in page
    assert "2026-08-06 04:00" not in page


def test_the_page_carries_no_external_asset(zeus):
    """It has to render with the network unplugged — half of what it
    diagnoses is a machine whose network is broken.

    The page does carry ONE script (the pop-up poller), so the assertion is
    that nothing is FETCHED from elsewhere: no src=, no href to a CDN, no
    absolute URL at all.
    """
    page = render_page(snapshot_of(zeus))
    for marker in ("http://", "https://", "//cdn", "<script src", "<link "):
        assert marker not in page, f"the page reaches for {marker}"


def test_the_page_carries_exactly_one_script_and_the_csp_names_its_hash(zeus):
    """'unsafe-inline' would let a transcript containing a <script> run.

    The pop-up needs a poller, so script-src names that one script's hash
    instead — a one-character edit to it, or any other inline block, is
    refused by the browser.
    """
    from zeus.dashboard.render import _POLLER, poller_hash

    page = render_page(snapshot_of(zeus))
    assert page.count("<script") == 1
    assert _POLLER in page
    assert poller_hash().startswith("'sha256-")


def test_the_csp_pins_the_script_that_is_actually_served(server):
    """Guards the guard: a hash that does not match the served bytes is a
    CSP that silently blocks the feature it was added for."""
    _, body, headers = get(server, "/")
    from zeus.dashboard.render import poller_hash

    assert poller_hash() in headers["Content-Security-Policy"]
    assert "'unsafe-inline'" not in headers["Content-Security-Policy"].split(
        "script-src"
    )[1].split(";")[0]
    assert "connect-src 'self'" in headers["Content-Security-Policy"]


def test_the_page_renders_in_a_second_timezone(zeus):
    """Guards the guard: the whole page, not just the timestamp helper."""
    page = render_page(snapshot_of(zeus, tz=LAGOS))
    assert "Africa/Lagos" in page and "<html" in page


def test_the_json_endpoint_carries_the_same_snapshot(zeus):
    payload = json.loads(render_json(snapshot_of(zeus)))
    assert payload["today_goal"]["text"] == "ship the dashboard"
    assert len(payload["actions"]) == 2


# ---- serving ----------------------------------------------------------
@pytest.fixture
def server(zeus):
    instance = build_server(
        zeus["db"], zeus["journal_dir"], LOS_ANGELES, port=0,
        settings={"model": "claude-opus-5"}, now=lambda: NOW,
    )
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()
    thread.join(timeout=5)


def get(server, path):
    host, port = server.server_address[:2]
    with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=5) as response:
        return response.status, response.read().decode("utf-8"), response.headers


def test_the_dashboard_binds_loopback_and_never_the_wildcard(server):
    """The whole security model in one assertion.

    This page shows every goal, transcript and tool call with no
    authentication, which is only safe because it is unreachable from
    anywhere but this machine. 0.0.0.0 would publish all of it to whatever
    wifi the laptop is on.
    """
    assert BIND_HOST == "127.0.0.1"
    assert server.server_address[0] == "127.0.0.1"
    assert server.server_address[0] not in ("0.0.0.0", "::", "")


def test_the_page_is_served(server):
    status, body, headers = get(server, "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert "ship the dashboard" in body
    assert "claude-opus-5" in body          # settings reached the page


def test_the_api_serves_json(server):
    status, body, headers = get(server, "/api/snapshot")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert json.loads(body)["today"] == "2026-08-05"


def test_an_unknown_path_is_a_404_not_a_stack_trace(server):
    with pytest.raises(urllib.error.HTTPError) as raised:
        get(server, "/../../etc/passwd")
    assert raised.value.code == 404


def test_the_response_forbids_scripts_and_framing(server):
    _, _, headers = get(server, "/")
    policy = headers["Content-Security-Policy"]
    assert "default-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_the_popup_is_off_until_a_conversation_is_actually_open(zeus):
    """`ended_at IS NULL` is the test, not "recent".

    Both CheckIn and _handle_activation close their conversation in a
    `finally`, so an open row means one is running right now. A pop-up that
    fired on the most recent conversation instead would appear every time
    the page was opened, hours after anyone spoke.
    """
    from zeus.dashboard.render import live_session

    # The fixture's conversation was never ended -> live.
    assert live_session(snapshot_of(zeus))["active"] is True

    store = Store(zeus["db"], FakeClock(NOW))
    store.end_conversation(1)
    store.close()
    assert live_session(snapshot_of(zeus))["active"] is False


def test_the_session_endpoint_feeds_the_popup(server):
    status, body, headers = get(server, "/api/session")
    assert status == 200
    payload = json.loads(body)
    assert payload["active"] is True
    assert payload["messages"][-1]["content"] == "Ship the dashboard."
    # Small on purpose: fetched every two seconds for as long as the page is
    # open, so it must not carry the action log and every transcript.
    assert len(body) < 2000


def test_a_transcript_cannot_smuggle_markup_into_the_popup(tmp_path):
    """The pop-up is the one place text arrives as JSON rather than escaped
    HTML, so it is painted with textContent. This pins the server half."""
    store = Store(tmp_path / "zeus.db", FakeClock(NOW))
    conversation = store.start_conversation("wake")
    store.add_message(conversation, "user", "<img src=x onerror=alert(1)>")
    store.close()
    page = render_page(
        read_snapshot(tmp_path / "zeus.db", tmp_path / "j", LOS_ANGELES, NOW)
    )
    assert "<img src=x" not in page
    assert "&lt;img src=x" in page


def test_a_broken_snapshot_reports_itself_instead_of_hanging(zeus):
    """A dashboard that 500s silently tells the user nothing about the
    daemon it exists to report on."""
    instance = build_server(zeus["db"], zeus["journal_dir"], LOS_ANGELES, port=0)

    def explode():
        raise RuntimeError("boom")

    instance.snapshot_source = explode
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as raised:
            get(instance, "/")
        assert raised.value.code == 500
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=5)
