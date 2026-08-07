import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from zoneinfo import ZoneInfo

from zeus.audio.mic import FRAME_SAMPLES, MicStream
from zeus.brain.fake import FakeConversation
from zeus.clock import FakeClock
from zeus.config import Config, ScheduleConfig
from zeus.context.presence import Signals, Verdict
from zeus.memory.journal import Journal
from zeus.memory.store import Store
from zeus.ritual.checkin import CheckIn, FakeNotifier, VoiceIO, local_date
from zeus.ritual.retry import Outcome
from zeus.schedule.cron import hhmm_to_cron, next_occurrence
from zeus.stt.fake import FakeTranscriber
from zeus.tts.fake import FakeSpeaker

LAGOS = ZoneInfo("Africa/Lagos")
LOS_ANGELES = ZoneInfo("America/Los_Angeles")
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


def test_silence_is_answered_out_loud_once(wiring):
    """C-I4, spec §10: "Empty or unintelligible transcript -> 'I didn't catch
    that' ONCE, then end the turn cleanly."

    `if not reply: break` ended the turn in silence, which from the user's
    side is indistinguishable from a daemon that has crashed -- and §10's
    governing rule is "fail loudly, never pretend: a voice assistant that
    silently stops listening is worse than one that says it is broken".
    """
    from zeus.brain.prompts import NOT_CAUGHT_LINE

    checkin, voice, _, _, _ = _checkin("morning", wiring, Verdict.SPEAK, heard=[])
    checkin.run(NOW)

    assert voice.spoken.count(NOT_CAUGHT_LINE) == 1, (
        f"expected exactly one {NOT_CAUGHT_LINE!r}, spoke {voice.spoken}"
    )
    assert voice.spoken[-1] == NOT_CAUGHT_LINE


def test_silence_after_a_reply_is_still_answered_exactly_once(wiring):
    """ONCE PER TURN, not once per empty reply in the loop. The user
    answered the first question and then walked away."""
    from zeus.brain.prompts import NOT_CAUGHT_LINE

    checkin, voice, _, _, _ = _checkin(
        "morning", wiring, Verdict.SPEAK, heard=["Finish the auth flow"]
    )
    checkin.run(NOW)

    assert voice.spoken.count(NOT_CAUGHT_LINE) == 1
    assert voice.spoken[-1] == NOT_CAUGHT_LINE


def test_a_conversation_that_is_answered_throughout_never_says_it(wiring):
    """The other half: a turn that ends because it ran out of exchanges, not
    because it heard nothing, must not accuse the user of being inaudible."""
    from zeus.brain.prompts import NOT_CAUGHT_LINE

    checkin, voice, _, _, _ = _checkin(
        "morning", wiring, Verdict.SPEAK, heard=["one", "two", "three"]
    )
    checkin.run(NOW)

    assert NOT_CAUGHT_LINE not in voice.spoken


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


def test_a_retry_finds_the_same_row_when_local_and_utc_dates_differ(wiring):
    """Regression: open_checkin used to derive local_date from
    scheduled_for.date() -- the UTC calendar date -- while find_open_checkin
    searched on the local one. Africa/Lagos (UTC+1) can never expose this:
    local and UTC dates only diverge at a negative UTC offset. Los Angeles
    (UTC-7 in August) can, and scheduled_for below is built the way the real
    scheduler builds it -- cron.next_occurrence always returns a UTC-tagged
    datetime -- not hand-constructed to dodge the bug.
    """
    clock, store, journal = wiring
    anchor = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)  # 03:00 PDT
    scheduled_for = next_occurrence(hhmm_to_cron("21:00"), anchor, LOS_ANGELES)
    date = local_date(scheduled_for, LOS_ANGELES)
    # The whole point: the UTC calendar date and the local one disagree.
    assert scheduled_for.date().isoformat() != date

    checkin = CheckIn(
        kind="evening", store=store, journal=journal,
        presence=StubPresence(Verdict.DEFER), voice=StubVoice(),
        notifier=FakeNotifier(),
        conversation_factory=lambda conv_id, local: FakeConversation({}),
        config=ScheduleConfig(), tz=LOS_ANGELES, clock=clock,
    )

    checkin.run(scheduled_for)
    after_first = [
        r["id"] for r in
        store.connection.execute("SELECT id FROM checkins ORDER BY id").fetchall()
    ]
    assert len(after_first) == 1
    first_id = after_first[0]

    checkin.run(scheduled_for)
    after_second = [
        r["id"] for r in
        store.connection.execute("SELECT id FROM checkins ORDER BY id").fetchall()
    ]

    # A retry must reuse the same row, not open a new one -- else attempts
    # never advance, both retry budgets never exhaust, and the table
    # accumulates a duplicate row per attempt.
    assert after_second == [first_id], (
        f"expected the retry to reuse row {first_id}, found rows {after_second}"
    )
    assert store.get_checkin(first_id).attempts == 2


# ---- the §9.3 ladder, at the unit level (Task 19) ------------------------
#
# next_step() has always returned Decision(outcome, retry_after,
# fold_forward); run() read .outcome and dropped the rest, so no retry was
# ever recorded anywhere. These pin the persistence half. The scheduling
# half -- that a recorded retry actually FIRES -- is only observable with a
# real Scheduler and a moving clock, and lives in tests/test_daemon.py.


class BoomConversation:
    """A brain that is down. Models an Anthropic API hard failure."""

    def send(self, text):
        raise RuntimeError("the API is unreachable")


def _rig(tmp_path, verdict, conversation):
    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    config = ScheduleConfig()
    checkin = CheckIn(
        kind="morning", store=store, journal=journal,
        presence=StubPresence(verdict), voice=StubVoice(),
        notifier=FakeNotifier(),
        conversation_factory=lambda conv_id, local: conversation,
        config=config, tz=LAGOS, clock=clock,
    )
    return SimpleNamespace(checkin=checkin, store=store, config=config, clock=clock)


def _deferring_checkin(tmp_path):
    return _rig(tmp_path, Verdict.DEFER, FakeConversation({}))


def _raising_checkin(tmp_path):
    return _rig(tmp_path, Verdict.SPEAK, BoomConversation())


def test_a_deferred_checkin_schedules_its_own_retry(tmp_path):
    """The ladder, at the unit level: DEFER must leave a retry_at behind."""
    rig = _deferring_checkin(tmp_path)
    rig.checkin.run(NOW)
    row = rig.store.get_checkin(1)
    assert row.outcome == "deferred"
    assert row.retry_at == NOW + rig.config.defer_retry_after


def test_the_retry_is_measured_from_now_not_from_the_scheduled_time(tmp_path):
    """A retry that comes due at 11:20 must next come due at 11:40, not at
    11:20 again. Anchoring the interval on `scheduled_for` (the 11:00
    occurrence, which every retry re-runs with) would leave retry_at pinned
    at 11:20 forever and turn the daemon tick into a hot loop.
    """
    rig = _deferring_checkin(tmp_path)
    rig.checkin.run(NOW)
    rig.clock.advance(rig.config.defer_retry_after)

    rig.checkin.run(NOW)                       # the retry re-runs with 11:00

    assert rig.store.get_checkin(1).retry_at == (
        NOW + 2 * rig.config.defer_retry_after
    )


def test_an_exhausted_checkin_stops_scheduling_retries(tmp_path):
    """`max_defer_retries` counts RETRIES, not total attempts, so the ladder
    is one initial run plus that many retries -- see next_step()'s
    `attempts + 1 > config.max_defer_retries` and
    tests/ritual/test_retry.py::test_defer_retries_up_to_three_times, which
    pins a retry at attempts=0, 1 AND 2. Hence the `+ 1`.
    """
    rig = _deferring_checkin(tmp_path)
    for _ in range(rig.config.max_defer_retries + 1):
        rig.checkin.run(NOW)
        rig.clock.advance(rig.config.defer_retry_after)
    assert rig.store.get_checkin(1).attempts == rig.config.max_defer_retries + 1
    assert rig.store.get_checkin(1).retry_at is None


def test_an_answered_checkin_clears_its_pending_retry(tmp_path):
    """The user was away at 11:00 and answered at 11:20. Nothing must fire
    at 11:40 -- the pending retry has to be cleared, not merely left behind
    on a row whose outcome happens to have changed."""
    rig = _deferring_checkin(tmp_path)
    rig.checkin.run(NOW)
    assert rig.store.get_checkin(1).retry_at is not None

    rig.clock.advance(rig.config.defer_retry_after)
    answering = CheckIn(
        kind="morning", store=rig.store,
        journal=Journal(tmp_path / "journal", rig.clock, LAGOS),
        presence=StubPresence(Verdict.SPEAK),
        voice=StubVoice(heard=["Finish the auth flow"]),
        notifier=FakeNotifier(),
        conversation_factory=lambda conv_id, local: FakeConversation({}),
        config=rig.config, tz=LAGOS, clock=rig.clock,
    )
    answering.run(NOW)

    row = rig.store.get_checkin(1)
    assert row.outcome == "answered"
    assert row.retry_at is None
    assert rig.store.due_retries(NOW + timedelta(hours=6)) == []


def test_a_conversation_that_raises_still_counts_as_an_attempt(tmp_path):
    """Without this, a failing brain retries forever: the exception escaped
    run() before update_checkin, so attempts never incremented."""
    rig = _raising_checkin(tmp_path)
    rig.checkin.run(NOW)                        # must not raise
    assert rig.store.get_checkin(1).attempts == 1


def test_a_conversation_that_raises_is_recorded_as_no_answer(tmp_path):
    """A dead attempt walks the NO_ANSWER ladder (30m x 1), not the DEFER
    one: ZEUS did try to speak. The row must still carry a retry so a
    transient API failure gets a second chance."""
    rig = _raising_checkin(tmp_path)
    outcome = rig.checkin.run(NOW)
    row = rig.store.get_checkin(1)
    assert outcome is Outcome.NO_ANSWER
    assert row.outcome == "no_answer"
    assert row.retry_at == NOW + rig.config.no_answer_retry_after


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
    config = AudioConfig()
    return VoiceIO(activator, mic, transcriber, speaker, config)


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


def _await(predicate, message, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, message
        time.sleep(0.005)


def test_listen_does_not_inherit_audio_buffered_before_it_was_called():
    """listen() must not pick up ZEUS's own speech that was already in
    flight when it started, or the endpointer reads it as the start of the
    user's reply.

    This used to depend on listen() calling mic.drain(). It is structural
    now: mic.frames() opens a FRESH subscription whose queue starts empty,
    so audio broadcast before listen() began went to queues listen() does
    not own. That also covers HotkeyActivator, which has no mute/unmute at
    all and so could never have drained anything -- the emptiness no longer
    depends on any activator remembering to act.

    Uses a real MicStream so the assertion is about the actual queues, not
    a double's promise. Runs on a background thread with a bounded wait
    rather than a bare call, because a regression here doesn't raise -- it
    hangs (Subscription.frames() polls forever with nothing to stop it).
    """
    from zeus.config import AudioConfig

    mic = MicStream(AudioConfig())
    mic.subscribe()  # a long-lived consumer, as the wake detector is

    # Buffered before listen() is called -- the backlog of ZEUS's own voice.
    # It reaches the subscriptions that exist NOW, never listen()'s.
    for _ in range(5):
        mic._on_audio(SPEECH, FRAME_SAMPLES, None, None)

    transcriber = FakeTranscriber(["hello"])
    voice = _voice(RecordingActivator(), FakeSpeaker(), transcriber, mic)

    outcome = {}
    finished = threading.Event()

    def run():
        outcome["text"] = voice.listen()
        finished.set()

    threading.Thread(target=run, daemon=True).start()
    _await(lambda: len(mic._subscribers) == 2, "listen() never subscribed")

    # Produced only after listen() subscribed -- what the user actually says.
    for _ in range(3):
        mic._on_audio(SPEECH, FRAME_SAMPLES, None, None)
    mic.stop()  # lets frames() end once this second batch is consumed

    assert finished.wait(5), "listen() did not return within 5s"
    assert outcome["text"] == "hello"
    assert transcriber.calls == [3 * FRAME_SAMPLES * 2]


def test_listen_brackets_the_capture_in_mute_and_unmute():
    """RENAMED from test_listen_holds_the_detector_muted_for_the_whole_window,
    which claimed a property it did not test: `events == ["mute", "unmute"]`
    is equally satisfied by a listen() that calls mute() and unmute() back
    to back and then captures with the detector fully live. That mutant
    passed all 223 tests. What this test actually pins is the PAIRING -- both
    calls happen, in that order, exactly once. The window itself is pinned
    by test_a_live_wake_detector_cannot_fire_during_the_listen_window below,
    against a real WakeWordActivator and a model that does fire.
    """
    activator = RecordingActivator()
    voice = _voice(activator, FakeSpeaker(), FakeTranscriber([]), _silent_mic())

    assert voice.listen() == ""
    assert activator.events == ["mute", "unmute"]


def test_listen_unmutes_even_when_capture_raises():
    """Without the finally, a mid-capture failure leaves the detector muted
    forever and ZEUS never wakes again -- the same trap speak() already has
    a regression test for."""

    class BoomMic:
        def frames(self):
            raise RuntimeError("audio device vanished mid-capture")

    activator = RecordingActivator()
    voice = _voice(activator, FakeSpeaker(), FakeTranscriber([]), BoomMic())

    with pytest.raises(RuntimeError):
        voice.listen()
    assert activator.events == ["mute", "unmute"]


def test_a_live_wake_detector_cannot_fire_during_the_listen_window(monkeypatch):
    """C2 (round 4), BOTH halves, across the real wiring build_daemon() builds.

    ONE MicStream is shared by the wake detector (its own thread, running
    forever) and VoiceIO.listen() (the main thread, during a check-in).

    First half -- the fan-out. With a single shared queue, queue.get()
    removed the frame, so the detector took roughly half the user's answer
    and Whisper received non-contiguous 80 ms chunks, recorded as NO_ANSWER
    or garbage. Every frame produced during the listen window must reach the
    transcriber whole.

    Second half -- the mute (F2, round 5). The detector now HEARS all of the
    user's answer, so a "hey zeus" inside it would launch an ad-hoc
    conversation on top of the running check-in. This test used to run a
    _NeverFires model, which proves nothing about muting: a fully live
    detector fires nothing either. Mutating listen() to `mute(); unmute();`
    followed by an unguarded capture -- the ruling exactly inverted -- passed
    all 223 tests. The model below scores ABOVE threshold on every frame, so
    a detector that is live for any part of the window fires, and the
    assertions are that it did not: no events, and not one frame scored.
    """
    from zeus.audio.wakeword import WakeWordActivator
    from zeus.config import AudioConfig, WakeConfig

    class _AlwaysFires:
        """Every frame is a wake word. If the detector is listening at all
        during the window, this makes it impossible to miss."""

        def __init__(self):
            self.calls = 0

        def predict(self, samples):
            self.calls += 1
            return {"hey_jarvis": 0.99}

    model = _AlwaysFires()
    mic = MicStream(AudioConfig())
    activator = WakeWordActivator(mic, WakeConfig(), threshold=0.5)
    monkeypatch.setattr(activator, "_load_model", lambda: model)
    activator.start()

    fired = []
    threading.Thread(
        target=lambda: fired.extend(activator.events()), daemon=True
    ).start()
    _await(
        lambda: activator._subscription is not None,
        "the wake detector never subscribed",
    )

    transcriber = FakeTranscriber(["the whole answer"])
    voice = _voice(activator, FakeSpeaker(), transcriber, mic)

    outcome = {}
    finished = threading.Event()

    def run():
        outcome["text"] = voice.listen()
        finished.set()

    threading.Thread(target=run, daemon=True).start()
    _await(lambda: len(mic._subscribers) == 2, "listen() never subscribed")

    spoken_frames = 10
    for _ in range(spoken_frames):
        mic._on_audio(SPEECH, FRAME_SAMPLES, None, None)

    # Let the detector actually reach every frame while the window is still
    # open. Without this the frames could still be sitting in its queue when
    # listen() returns, and "no events" would be a statement about timing
    # rather than about muting.
    _await(
        lambda: activator._subscription._queue.empty(),
        "the wake detector never consumed the answer",
    )
    mic.stop()

    assert finished.wait(5), "listen() did not return within 5s"
    assert outcome["text"] == "the whole answer"
    assert transcriber.calls == [spoken_frames * FRAME_SAMPLES * 2], (
        "the wake detector consumed part of the user's answer"
    )
    assert fired == [], (
        f"the wake detector fired {len(fired)} time(s) inside the listen "
        "window -- a 'hey zeus' in the user's answer would start a second "
        "conversation on top of the running check-in"
    )
    assert model.calls == 0, (
        f"{model.calls} frames of the user's answer were scored by the wake "
        "model during the listen window; the detector must be muted for all "
        "of it, not merely at the start and end"
    )
