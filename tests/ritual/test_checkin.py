from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from zoneinfo import ZoneInfo

from zeus.audio.mic import FRAME_SAMPLES
from zeus.brain.fake import FakeConversation
from zeus.clock import FakeClock
from zeus.config import Config, ScheduleConfig
from zeus.context.presence import Signals, Verdict
from zeus.memory.journal import Journal
from zeus.memory.store import Store
from zeus.ritual.checkin import CheckIn, FakeNotifier, VoiceIO, local_date
from zeus.ritual.retry import Outcome
from zeus.stt.fake import FakeTranscriber
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


# ---- VoiceIO: the half-duplex seam --------------------------------------
#
# Everything above exercises CheckIn through StubVoice, which never touches
# VoiceIO itself. That leaves the mute/unmute pairing unproven: if speak()'s
# finally were ever weakened to a plain trailing unmute(), a raising
# Speaker.say() would leave the wake detector muted forever with no code
# path left to unmute it — ZEUS goes permanently deaf until the daemon is
# restarted. These tests pin that seam directly.

SILENCE = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()
SPEECH = (np.ones(FRAME_SAMPLES, dtype=np.int16) * 8000).tobytes()


class RecordingActivator:
    """Records mute/unmute ordering. Not FakeActivator: that one ignores
    start()/stop() and has no mute()/unmute() at all."""

    def __init__(self):
        self.events = []

    def mute(self):
        self.events.append("mute")

    def unmute(self):
        self.events.append("unmute")


class BoomSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)
        raise RuntimeError("audio device vanished mid-sentence")

    def stop(self):
        pass


class _SilentMic:
    """Yields only silence frames, then ends."""

    def frames(self):
        for _ in range(50):
            yield SILENCE


class _LoudMic:
    """Yields loud frames without ever going quiet.

    Capped at 2000 frames — far above the 375 the default config's listen
    timeout should consume — so that a regression which fails to respect
    the timeout ends the generator and fails the byte-count assertion below
    instead of hanging the suite.
    """

    def frames(self):
        for _ in range(2000):
            yield SPEECH


def _silent_mic():
    return _SilentMic()


def _loud_mic():
    return _LoudMic()


def _voice(activator, speaker, transcriber=None, mic=None):
    from zeus.config import AudioConfig
    from zeus.audio.endpointer import Endpointer
    config = AudioConfig()
    return VoiceIO(
        activator, mic, Endpointer(config), transcriber, speaker, config
    )


def test_speak_brackets_the_utterance_in_mute_and_unmute():
    activator, speaker = RecordingActivator(), FakeSpeaker()
    _voice(activator, speaker).speak(["One.", "Two."])
    assert activator.events == ["mute", "unmute"]
    assert speaker.said == ["One.", "Two."]


def test_speak_unmutes_even_when_the_speaker_raises():
    """Without the finally, ZEUS stays muted forever and never wakes again."""
    activator, speaker = RecordingActivator(), BoomSpeaker()
    with pytest.raises(RuntimeError):
        _voice(activator, speaker).speak(["One.", "Two."])
    assert activator.events == ["mute", "unmute"]
    assert speaker.said == ["One."]          # aborted at the first sentence


def test_listen_returns_empty_when_nothing_was_heard():
    activator = RecordingActivator()
    mic = _silent_mic()            # yields only silence frames, then ends
    voice = _voice(activator, FakeSpeaker(), FakeTranscriber([]), mic)
    assert voice.listen() == ""


def test_listen_stops_at_the_configured_timeout():
    """30s listen window at 80ms/frame is 375 frames; a mic that never goes
    quiet must still return rather than holding the microphone open."""
    from zeus.config import AudioConfig

    config = AudioConfig()
    expected_frames = int(
        config.listen_timeout.total_seconds() * config.sample_rate / FRAME_SAMPLES
    )
    activator = RecordingActivator()
    transcriber = FakeTranscriber(["hello"])
    voice = _voice(activator, FakeSpeaker(), transcriber, _loud_mic())

    result = voice.listen()

    assert result == "hello"
    assert transcriber.calls == [expected_frames * FRAME_SAMPLES * 2]
