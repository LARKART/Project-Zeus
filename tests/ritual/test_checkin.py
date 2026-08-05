from datetime import datetime, timedelta, timezone

import pytest
from zoneinfo import ZoneInfo

from zeus.brain.fake import FakeConversation
from zeus.clock import FakeClock
from zeus.config import Config, ScheduleConfig
from zeus.context.presence import Signals, Verdict
from zeus.memory.journal import Journal
from zeus.memory.store import Store
from zeus.ritual.checkin import CheckIn, FakeNotifier, local_date
from zeus.ritual.retry import Outcome
from zeus.tts.fake import FakeSpeaker

LAGOS = ZoneInfo("Africa/Lagos")
NOW = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)  # 11:00 Lagos


class StubPresence:
    def __init__(self, verdict):
        self._verdict = verdict

    def verdict(self):
        return self._verdict


class StubVoice:
    """Stands in for VoiceIO: records what was spoken, replays what is heard."""

    def __init__(self, heard=None):
        self.spoken: list[str] = []
        self._heard = list(heard or [])

    def speak(self, sentences):
        self.spoken.extend(sentences)

    def listen(self):
        return self._heard.pop(0) if self._heard else ""


@pytest.fixture
def wiring(tmp_path):
    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    return clock, store, journal


def _checkin(kind, wiring, verdict, heard, script=None):
    clock, store, journal = wiring
    voice = StubVoice(heard)
    notifier = FakeNotifier()
    conversation = FakeConversation(script or {})
    return (
        CheckIn(
            kind=kind, store=store, journal=journal,
            presence=StubPresence(verdict), voice=voice, notifier=notifier,
            conversation_factory=lambda conv_id, local: conversation,
            config=ScheduleConfig(), tz=LAGOS, clock=clock,
        ),
        voice, notifier, store, conversation,
    )


def test_local_date_uses_the_local_zone():
    # 23:30 UTC is already the next day in Lagos (UTC+1)
    late = datetime(2026, 8, 5, 23, 30, tzinfo=timezone.utc)
    assert local_date(late, LAGOS) == "2026-08-06"


def test_speak_verdict_runs_the_conversation_and_records_the_answer(wiring):
    checkin, voice, _, store, conversation = _checkin(
        "morning", wiring, Verdict.SPEAK, heard=["Finish the auth flow"],
        script={"[morning check-in] Greet the user briefly and ask what the "
                "one thing is that has to happen today.": ["Morning.",
                                                           "What's the one thing?"]},
    )
    outcome = checkin.run(NOW)

    assert outcome is Outcome.ANSWERED
    assert voice.spoken[0] == "Morning."
    assert conversation.sent[1] == "Finish the auth flow"
    assert store.get_checkin(1).outcome == "answered"


def test_silence_produces_no_answer_and_a_retry(wiring):
    checkin, _, _, store, _ = _checkin("morning", wiring, Verdict.SPEAK, heard=[])
    outcome = checkin.run(NOW)

    assert outcome is Outcome.NO_ANSWER
    checkin_row = store.get_checkin(1)
    assert checkin_row.outcome == "no_answer"
    assert checkin_row.attempts == 1


def test_defer_verdict_never_speaks(wiring):
    checkin, voice, notifier, store, _ = _checkin(
        "morning", wiring, Verdict.DEFER, heard=["ignored"]
    )
    outcome = checkin.run(NOW)

    assert outcome is Outcome.DEFERRED
    assert voice.spoken == []
    assert notifier.sent == []


def test_notify_verdict_notifies_without_speaking(wiring):
    checkin, voice, notifier, _, _ = _checkin(
        "morning", wiring, Verdict.NOTIFY, heard=["ignored"]
    )
    outcome = checkin.run(NOW)

    assert outcome is Outcome.DEFERRED
    assert voice.spoken == []
    assert len(notifier.sent) == 1
    assert "ZEUS" in notifier.sent[0][0]


def test_evening_checkin_recalls_the_morning_goal(wiring):
    _, store, _ = wiring
    store.set_goal("2026-08-05", "Finish the auth flow")

    checkin, _, _, _, conversation = _checkin(
        "evening", wiring, Verdict.SPEAK, heard=["Mostly, tests are missing"]
    )
    checkin.run(NOW)

    assert "Finish the auth flow" in conversation.sent[0]


def test_evening_checkin_without_a_goal_uses_the_folded_opener(wiring):
    checkin, _, _, _, conversation = _checkin(
        "evening", wiring, Verdict.SPEAK, heard=["I worked on docs"]
    )
    checkin.run(NOW)

    assert "goal never captured" in conversation.sent[0]


def test_conversation_stops_at_three_exchanges(wiring):
    checkin, _, _, _, conversation = _checkin(
        "morning", wiring, Verdict.SPEAK,
        heard=["one", "two", "three", "four", "five"],
    )
    checkin.run(NOW)
    # opener + at most 3 user replies
    assert len(conversation.sent) <= 4


def test_attempts_accumulate_across_repeated_runs(wiring):
    checkin, _, _, store, _ = _checkin("morning", wiring, Verdict.DEFER, heard=[])
    checkin.run(NOW)
    checkin.run(NOW)
    assert store.get_checkin(1).attempts == 2


def test_journal_records_a_missed_checkin(wiring):
    _, _, journal = wiring
    checkin, _, _, _, _ = _checkin("evening", wiring, Verdict.SPEAK, heard=[])
    checkin.run(NOW)
    assert "no answer" in journal.read("2026-08-05").lower()
