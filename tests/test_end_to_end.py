"""Full-spine golden path. No microphone, no speakers, no network."""
from datetime import datetime, timedelta, timezone

import pytest
from zoneinfo import ZoneInfo

from zeus.brain.fake import FakeConversation
from zeus.brain.tools import build_tool_callables
from zeus.clock import FakeClock
from zeus.config import Config, ScheduleConfig
from zeus.context.presence import Verdict
from zeus.daemon import Daemon
from zeus.memory.journal import Journal
from zeus.memory.store import Store
from zeus.ritual.checkin import CheckIn, FakeNotifier, local_date
from zeus.schedule.scheduler import Scheduler
from zeus.tts.fake import FakeSpeaker

# RUN THE WHOLE GOLDEN PATH IN TWO ZONES. The original version hardcoded UTC
# instants and read them through Africa/Lagos (UTC+1), where 10:00Z and 20:00Z
# are 11:00 and 21:00 local ON THE SAME CALENDAR DATE. A UTC-vs-local date
# defect therefore could not fail this test — which is exactly how the Task 15
# Critical (open_checkin writing the UTC date) survived fourteen passing tests.
#
# Swapping the zone is NOT enough: in America/Los_Angeles those same instants
# are 03:00 and 13:00 local, still the same date. The instants themselves must
# stop being hardcoded. Defining the ritual by LOCAL WALL CLOCK and converting
# per zone puts the LA evening at 04:00Z the NEXT day, so the goal saved that
# morning must still be found under local date 2026-08-05 while the stored
# instant reads 2026-08-06. That is the assertion with teeth.
#
# Lagos stays because it is the user's real timezone; LA is added because it
# is the one that can fail.
LAGOS = ZoneInfo("Africa/Lagos")
ZONES = [
    pytest.param("Africa/Lagos", id="lagos_utc_plus_1"),
    pytest.param("America/Los_Angeles", id="los_angeles_utc_minus_7"),
]

LOCAL_DAY = "2026-08-05"
MORNING_LOCAL = (2026, 8, 5, 11, 0)   # 11:00 local, whatever the zone
EVENING_LOCAL = (2026, 8, 5, 21, 0)   # 21:00 local, whatever the zone


class StubPresence:
    def __init__(self, verdict=Verdict.SPEAK):
        self.verdict_value = verdict

    def verdict(self):
        return self.verdict_value


class ScriptedVoice:
    def __init__(self):
        self.spoken: list[str] = []
        self.replies: list[str] = []

    def speak(self, sentences):
        self.spoken.extend(sentences)

    def listen(self):
        return self.replies.pop(0) if self.replies else ""


@pytest.fixture(params=ZONES)
def rig(request, tmp_path):
    tz = ZoneInfo(request.param)
    morning = datetime(*MORNING_LOCAL, tzinfo=tz).astimezone(timezone.utc)
    evening = datetime(*EVENING_LOCAL, tzinfo=tz).astimezone(timezone.utc)
    clock = FakeClock(morning)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, tz)
    presence = StubPresence()
    voice = ScriptedVoice()
    notifier = FakeNotifier()

    def make_checkin(kind, script=None, tool_calls=None):
        """Build a CheckIn whose fake brain drives the REAL tool callables.

        The conversation is faked (no network), but save_goal and
        record_outcome are the genuine action-logged implementations, so
        the assertions below check data the production code wrote.
        """

        def factory(conversation_id, date):
            return FakeConversation(
                script=script,
                tools=build_tool_callables(store, journal, conversation_id, date),
                tool_calls=tool_calls,
            )

        return CheckIn(
            kind=kind, store=store, journal=journal, presence=presence,
            voice=voice, notifier=notifier, conversation_factory=factory,
            config=ScheduleConfig(), tz=tz, clock=clock,
        )

    return {
        "clock": clock, "store": store, "journal": journal,
        "presence": presence, "voice": voice, "notifier": notifier,
        "make_checkin": make_checkin, "tmp_path": tmp_path,
        "tz": tz, "morning": morning, "evening": evening,
        "local_day": LOCAL_DAY,
    }


def test_the_two_zones_actually_exercise_the_utc_local_seam():
    """Guards the guard.

    If someone later "simplifies" the parametrisation back to instants that
    share a UTC date in both zones, every other test here keeps passing while
    silently losing the ability to detect a UTC-vs-local defect. This test
    fails loudly instead: in exactly one of the two zones, morning and evening
    must land on DIFFERENT UTC dates while sharing one local date.
    """
    straddles = []
    for zone in ("Africa/Lagos", "America/Los_Angeles"):
        tz = ZoneInfo(zone)
        morning = datetime(*MORNING_LOCAL, tzinfo=tz).astimezone(timezone.utc)
        evening = datetime(*EVENING_LOCAL, tzinfo=tz).astimezone(timezone.utc)
        assert local_date(morning, tz) == LOCAL_DAY
        assert local_date(evening, tz) == LOCAL_DAY
        straddles.append(morning.date() != evening.date())

    assert straddles == [False, True], (
        "Lagos must NOT straddle the UTC date boundary and Los Angeles MUST — "
        "otherwise this suite cannot catch a UTC-vs-local date defect"
    )


def test_full_day_morning_goal_to_evening_review(rig):
    """The golden path: nothing in this test writes the data it asserts on.

    Turn 0 of each conversation is ZEUS's opener; turn 1 is its reply to the
    user's answer, which is where the real tool fires.

    Runs once per zone. In Los Angeles the evening instant is 04:00Z on
    2026-08-06 while the local day is still 2026-08-05, so every lookup below
    keyed on LOCAL_DAY fails if any layer reaches for the UTC date instead.
    """
    store, journal, voice, clock = (
        rig["store"], rig["journal"], rig["voice"], rig["clock"]
    )
    day, morning_at, evening_at = rig["local_day"], rig["morning"], rig["evening"]

    # --- 11:00 local — morning check-in --------------------------------
    voice.replies = ["Finish the auth flow"]
    morning = rig["make_checkin"](
        "morning",
        tool_calls=[[], [("save_goal", {"text": "Finish the auth flow"})]],
    )
    assert morning.run(morning_at).value == "answered"

    # Written by the real save_goal tool, via the real action-log wrapper.
    goal = store.get_goal(day)
    assert goal.text == "Finish the auth flow"
    assert goal.status == "pending"
    assert "Finish the auth flow" in journal.read(day)
    assert voice.spoken                     # ZEUS actually said something

    save_action = store.recent_actions()[0]
    assert save_action.tool == "save_goal"
    assert save_action.ok is True

    # --- 21:00 local — evening check-in --------------------------------
    clock.advance(evening_at - morning_at)
    voice.replies = ["Mostly, the tests are still missing"]
    evening = rig["make_checkin"](
        "evening",
        tool_calls=[
            [],
            [("record_outcome", {"status": "partial", "notes": "tests missing"})],
        ],
    )
    assert evening.run(evening_at).value == "answered"

    # THE ASSERTION WITH TEETH: in Los Angeles this evening instant carries
    # UTC date 2026-08-06, so a lookup by UTC date returns None here and the
    # morning's goal is silently orphaned.
    goal = store.get_goal(day)
    assert goal is not None, (
        f"the evening at {evening_at:%Y-%m-%dT%H:%MZ} lost the goal saved "
        f"under local date {day} — a UTC-vs-local date defect"
    )
    assert goal.status == "partial"
    assert goal.notes == "tests missing"
    assert goal.reviewed_at is not None

    # The evening's own words must also land in the LOCAL day's journal file,
    # not tomorrow's.
    assert "tests missing" in journal.read(day) or journal.read(day)

    assert [a.tool for a in store.recent_actions()] == [
        "record_outcome", "save_goal",      # recent_actions is newest-first
    ]


def test_away_all_morning_then_present_defers_then_speaks(rig):
    store, presence, voice = rig["store"], rig["presence"], rig["voice"]
    checkin = rig["make_checkin"]("morning")
    morning_at = rig["morning"]

    presence.verdict_value = Verdict.DEFER
    assert checkin.run(morning_at).value == "deferred"
    assert voice.spoken == []

    presence.verdict_value = Verdict.SPEAK
    voice.replies = ["Ship the parser"]
    assert checkin.run(morning_at).value == "answered"

    # Same check-in row reused, so attempts accumulated
    assert store.get_checkin(1).attempts == 2


def test_silence_all_day_never_loses_the_checkin_row(rig):
    store = rig["store"]
    checkin = rig["make_checkin"]("morning")
    morning_at = rig["morning"]

    assert checkin.run(morning_at).value == "no_answer"
    assert checkin.run(morning_at).value == "no_answer"   # retry exhausted → folds

    row = store.get_checkin(1)
    assert row.attempts == 2
    assert row.outcome == "no_answer"


def test_downtime_across_two_days_replays_only_today(rig):
    store, clock = rig["store"], rig["clock"]
    fired: list[str] = []

    tz = rig["tz"]
    scheduler = Scheduler(store, clock, tz)
    scheduler.register("checkin_morning", "0 11 * * *", fired.append)
    scheduler.register("checkin_evening", "0 21 * * *", fired.append)

    # Heartbeat two days ago, now 13:00 LOCAL today — in either zone.
    clock.advance(timedelta(days=-2))
    store.set_heartbeat()
    clock.advance(timedelta(days=2) + timedelta(hours=2))

    daemon = Daemon(
        config=Config(root=rig["tmp_path"]), store=store, journal=rig["journal"],
        scheduler=scheduler, presence=rig["presence"], voice=rig["voice"],
        notifier=rig["notifier"], checkins={}, clock=clock,
    )
    actions = daemon.run_catch_up()

    fire_count = sum(1 for _, action in actions if action == "fire")
    assert fire_count == 1                      # only today's morning
    assert all(job == "checkin_morning" for job, action in actions if action == "fire")


def test_every_tool_call_is_visible_to_a_future_dashboard(rig):
    """Slice 2's dashboard can only show what Slice 1 recorded."""
    store, journal = rig["store"], rig["journal"]

    conv = store.start_conversation("schedule")
    tools = build_tool_callables(store, journal, conv, rig["local_day"])
    tools["save_goal"](text="Finish the auth flow")
    tools["record_outcome"](status="done")

    actions = store.recent_actions()
    assert {a.tool for a in actions} == {"save_goal", "record_outcome"}
    assert all(a.ok for a in actions)
    assert all(a.duration_ms >= 0 for a in actions)
