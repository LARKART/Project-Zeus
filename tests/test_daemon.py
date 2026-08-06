import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from zoneinfo import ZoneInfo

from zeus.audio.activator import FakeActivator
from zeus.audio.mic import FRAME_SAMPLES, MicStream
from zeus.brain.fake import FakeConversation
from zeus.clock import FakeClock
from zeus.config import AudioConfig, Config, ScheduleConfig
from zeus.context.presence import Verdict
from zeus.daemon import Daemon, DegradedPresence, audio_self_test, catch_up_actions
from zeus.memory.journal import Journal
from zeus.memory.store import Store
from zeus.ritual.checkin import CheckIn, FakeNotifier
from zeus.ritual.retry import Outcome
from zeus.schedule.scheduler import MissedRun, Scheduler

LAGOS = ZoneInfo("Africa/Lagos")
NOW = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)

SILENCE = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()
SPEECH = (np.ones(FRAME_SAMPLES, dtype=np.int16) * 5000).tobytes()


def _mic(frames):
    mic = MicStream(AudioConfig())
    for frame in frames:
        mic._on_audio(frame, FRAME_SAMPLES, None, None)
    mic.stop()
    return mic


class _RecordingMic:
    """A mic double that only records start()/stop() calls.

    Not a real MicStream: start() must never touch sounddevice, since
    building one requires a real audio device. Used to assert Daemon.stop()
    reaches the mic, without exercising MicStream.frames()'s own shutdown
    plumbing (that's covered separately by tests/audio/test_mic.py).
    """

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class _FailingMic:
    """A mic double whose self-test always fails: no frames, ever."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def frames(self):
        return iter(())


class _MicThatGoesQuietMidCapture:
    """Yields a couple of frames, then blocks forever.

    Models a dead-hardware variant of R1: the audio callback simply stops
    firing. A real MicStream in that state has nothing that would ever set
    its `_stopping` Event, so mic.frames() would hang the calling thread
    indefinitely -- there is no further frame and no StopIteration, so a
    generator-based `for` loop never gets control back. This reproduces
    that shape deterministically, without needing real hardware: the fresh
    Event below is never `set()`, by construction, so `.wait()` on it never
    returns.
    """

    def frames(self):
        yield SPEECH
        yield SPEECH
        threading.Event().wait()


class _StubPresence:
    def __init__(self, verdict: Verdict) -> None:
        self._verdict = verdict

    def verdict(self) -> Verdict:
        return self._verdict


class _StubVoice:
    """Records what was spoken. Mirrors tests/ritual/test_checkin.py's
    StubVoice, kept local rather than imported since test modules don't
    share fixtures across files by convention here."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, sentences) -> None:
        self.spoken.extend(sentences)

    def listen(self) -> str:
        return ""


def test_self_test_passes_on_real_audio():
    assert audio_self_test(_mic([SPEECH] * 13), seconds=1.0) is True


def test_self_test_fails_on_pure_silence():
    """Risk R1: a TCC-denied stream opens fine and returns silence forever."""
    assert audio_self_test(_mic([SILENCE] * 13), seconds=1.0) is False


def test_self_test_fails_when_no_frames_arrive():
    assert audio_self_test(_mic([]), seconds=1.0) is False


def test_self_test_times_out_when_audio_stops_mid_capture():
    """Important (review round 2): `seconds` bounds a frame COUNT, not
    wall-clock. test_self_test_fails_when_no_frames_arrive does not cover
    this -- its `_mic()` helper calls mic.stop() first, so `_stopping` is
    already set and mic.frames() returns immediately. That tests "already
    stopped", not "still running but gone silent": if the real audio
    callback stops firing mid-capture, mic.frames() blocks forever with no
    frame and no StopIteration ever coming, and nothing sets `_stopping`
    during a self-test. Without an internal deadline this would hang
    Daemon.start() forever.

    `seconds=0.5` needs 6 frames (see audio_self_test's `wanted`
    calculation); the double below only ever produces 2 before blocking,
    so the third `next()` call is guaranteed to hit the block, not race it.

    audio_self_test's own fix is what must return False here -- wrapping
    the call in a background thread with a bounded wait is a safety net
    for THIS TEST (never call something that might hang directly in a test
    body), not a substitute for the fix: if audio_self_test regressed back
    to hanging forever, this thread would never finish either, and the
    assertion below would fail cleanly instead of hanging the suite.
    """
    result: dict[str, bool] = {}
    finished = threading.Event()

    def run() -> None:
        result["value"] = audio_self_test(
            _MicThatGoesQuietMidCapture(), seconds=0.5
        )
        finished.set()

    threading.Thread(target=run, daemon=True).start()
    completed = finished.wait(timeout=5)

    assert completed, (
        "audio_self_test did not return within 5s -- the mid-capture "
        "deadline regressed"
    )
    assert result["value"] is False


@pytest.mark.parametrize(
    "missed,expected",
    [
        # Morning missed earlier today → still worth asking
        ([MissedRun("checkin_morning", NOW, True)], [("checkin_morning", "fire")]),
        # Morning missed on a previous day → asking now is noise
        ([MissedRun("checkin_morning", NOW, False)], [("checkin_morning", "skip")]),
        # Evening is never replayed, even on the same day
        ([MissedRun("checkin_evening", NOW, True)], [("checkin_evening", "skip")]),
        ([MissedRun("checkin_evening", NOW, False)], [("checkin_evening", "skip")]),
    ],
)
def test_catch_up_policy(missed, expected):
    assert catch_up_actions(missed) == expected


def test_catch_up_policy_preserves_order():
    missed = [
        MissedRun("checkin_morning", NOW - timedelta(days=1), False),
        MissedRun("checkin_evening", NOW - timedelta(days=1), False),
        MissedRun("checkin_morning", NOW, True),
    ]
    assert [action for _, action in catch_up_actions(missed)] == [
        "skip", "skip", "fire",
    ]


def test_unknown_job_is_never_fired_by_catch_up():
    assert catch_up_actions([MissedRun("watchman_scan", NOW, True)]) == [
        ("watchman_scan", "skip")
    ]


def test_a_skipped_catch_up_is_not_refired_by_the_next_tick(tmp_path):
    """Critical #1 (review round 2): Scheduler.catch_up() reads the
    heartbeat while run_pending() reads each job's last_run_at -- separate
    state. Without run_catch_up() consuming the occurrence via
    store.set_job_run(), the very first tick() after catch-up recomputes
    from the stale last_run_at and overrides whatever catch-up just
    decided -- here, un-skipping a "skip" decision and asking about
    yesterday's evening after all, in violation of spec §9.2.

    This only reproduces on a restart, when the job row already carries a
    last_run_at -- with a fresh store, run_pending's first call merely
    seeds the baseline and fires nothing. The brief's own `daemon` fixture
    never calls run_pending before run_catch_up, so it cannot express this
    state; built directly on real Store/Scheduler/FakeClock instead.

    Uses checkin_evening, which is never catch-up-eligible under any
    circumstance (CATCH_UP_ELIGIBLE = {"checkin_morning"}), so the correct
    decision here is unconditionally "skip".
    """
    clock = FakeClock(datetime(2026, 8, 4, 20, 0, tzinfo=timezone.utc))
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    scheduler = Scheduler(store, clock, LAGOS)
    calls: list[datetime] = []
    scheduler.register("checkin_evening", "0 21 * * *", calls.append)

    # Prior run: yesterday's 21:00 Lagos (20:00 UTC) occurrence already
    # fired, so last_run_at is populated -- only true after a restart.
    store.set_job_run("checkin_evening", clock.now_utc())

    # Heartbeat this morning, well before today's 21:00 Lagos occurrence.
    clock.advance(timedelta(hours=13))  # 2026-08-05 09:00 UTC
    store.set_heartbeat()

    # "Restart" tonight, after today's 21:00 Lagos (20:00 UTC) occurrence
    # has already passed.
    clock.advance(timedelta(hours=13))  # 2026-08-05 22:00 UTC

    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={}, clock=clock,
    )

    actions = instance.run_catch_up()
    assert actions == [("checkin_evening", "skip")]
    assert calls == []

    instance.tick()

    assert calls == [], (
        "the tick immediately after catch-up re-fired a run catch-up had "
        "just decided to skip"
    )


def test_a_fired_catch_up_is_not_refired_by_the_next_tick(tmp_path):
    """Mirror of the skip case above: a run catch-up fires must not be
    fired AGAIN by the tick that immediately follows. Without consuming
    the occurrence, run_pending recomputes from the stale last_run_at,
    finds today's occurrence still due, and fires it a second time --
    CheckIn would either reopen the same row and re-run _converse
    (incrementing attempts a second time for one real occurrence) or, if
    the first attempt already resolved to answered, open a second row and
    ask the question again.

    checkin_morning is the only catch-up-eligible job.
    """
    clock = FakeClock(datetime(2026, 8, 4, 10, 0, tzinfo=timezone.utc))
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    scheduler = Scheduler(store, clock, LAGOS)
    calls: list[datetime] = []

    class _RecordingCheckIn:
        def run(self, scheduled_for):
            calls.append(scheduled_for)

    checkin = _RecordingCheckIn()
    scheduler.register("checkin_morning", "0 11 * * *", checkin.run)

    # Prior run: yesterday's 11:00 Lagos (10:00 UTC) occurrence already
    # fired, so last_run_at is populated -- only true after a restart.
    store.set_job_run("checkin_morning", clock.now_utc())

    # Heartbeat early this morning, before today's 11:00 Lagos occurrence.
    clock.advance(timedelta(hours=22))  # 2026-08-05 08:00 UTC
    store.set_heartbeat()

    # "Restart" after today's 11:00 Lagos (10:00 UTC) occurrence has passed.
    clock.advance(timedelta(hours=4))  # 2026-08-05 12:00 UTC

    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={"checkin_morning": checkin}, clock=clock,
    )

    actions = instance.run_catch_up()
    fired_at = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)
    assert actions == [("checkin_morning", "fire")]
    assert calls == [fired_at]

    instance.tick()

    assert calls == [fired_at], (
        "the tick immediately after catch-up re-fired a run catch-up had "
        "already fired"
    )


@pytest.fixture
def daemon(tmp_path):
    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    scheduler = Scheduler(store, clock, LAGOS)
    fired: list[str] = []
    scheduler.register("checkin_morning", "0 11 * * *", lambda when: fired.append("m"))
    daemon = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={}, clock=clock,
    )
    return daemon, store, clock, fired


def test_tick_writes_a_heartbeat(daemon):
    instance, store, _, _ = daemon
    assert store.heartbeat() is None
    instance.tick()
    assert store.heartbeat() == NOW


def test_tick_advances_the_heartbeat(daemon):
    instance, store, clock, _ = daemon
    instance.tick()
    clock.advance(timedelta(minutes=5))
    instance.tick()
    assert store.heartbeat() == NOW + timedelta(minutes=5)


def test_degraded_flag_defaults_to_false(daemon):
    instance, _, _, _ = daemon
    assert instance.degraded is False


def test_run_catch_up_is_empty_without_a_heartbeat(daemon):
    instance, _, _, _ = daemon
    assert instance.run_catch_up() == []


def test_stop_stops_the_mic_not_just_the_activator(tmp_path):
    """Finding #1: WakeWordActivator.events() only checks `self._running`
    AFTER `mic.frames()` yields a frame:

        for frame in self._mic.frames():
            if not self._running:
                return

    With no incoming audio, `mic.frames()` never yields, so that check is
    never reached — activator.stop() alone cannot unblock a thread parked
    inside it. What actually unblocks it is MicStream.frames() observing
    its own `_stopping` Event, which only `MicStream.stop()` sets. So
    Daemon.stop() must call mic.stop() directly, not rely on the activator
    to propagate the shutdown.

    This is deliberately the narrower assertion, not an end-to-end proof
    that a hung WakeWordActivator.events() call actually returns: wiring a
    real WakeWordActivator needs a wake-word model that
    openwakeword.utils.download_models() fetches over the network, which
    tests must never require. Recording doubles for both activator and mic
    keep this test to what Daemon.stop() itself is responsible for: calling
    stop() on both, not just the activator. MicStream's own shutdown
    behaviour (frames() ending once _stopping is set) is covered directly
    by tests/audio/test_mic.py.
    """
    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    scheduler = Scheduler(store, clock, LAGOS)
    activator = FakeActivator(count=0)
    mic = _RecordingMic()
    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={}, clock=clock, activator=activator, mic=mic,
    )

    instance.stop()

    assert mic.stopped is True
    assert activator.stopped is True


# ---- DegradedPresence: C2 (review round 2) -------------------------------
#
# `degraded` used to be wired to nothing CheckIn could see: it gated only
# the wake-word thread, while CheckIn picks SPEAK vs NOTIFY purely from
# presence.verdict(), which knows nothing about microphone health. With a
# dead mic, a check-in would still speak into the void, listen(), hear
# nothing, and record NO_ANSWER -- precisely what audio_self_test (risk R1)
# exists to prevent.


def test_degraded_presence_translates_speak_to_notify():
    assert DegradedPresence(_StubPresence(Verdict.SPEAK)).verdict() is Verdict.NOTIFY


def test_degraded_presence_passes_defer_through():
    assert DegradedPresence(_StubPresence(Verdict.DEFER)).verdict() is Verdict.DEFER


def test_a_failed_self_test_makes_check_ins_notify_instead_of_speak(tmp_path):
    """Behavioural, not just the flag: test_degraded_flag_defaults_to_false
    already covers `degraded` itself and proves nothing about whether
    CheckIn actually respects it.

    Wires a REAL CheckIn (not a stub) sharing the SAME presence object with
    the Daemon, exactly as build_daemon() does -- both are handed the one
    `presence` built from config.context -- so this proves the wiring
    reaches the CheckIn, not merely that DegradedPresence's own logic is
    correct in isolation (that's pinned separately above). presence
    returns SPEAK; the mic double fails audio_self_test immediately (no
    frames, ever). After start(), the check-in must notify rather than
    speak.
    """
    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    scheduler = Scheduler(store, clock, LAGOS)
    presence = _StubPresence(Verdict.SPEAK)
    voice = _StubVoice()
    notifier = FakeNotifier()
    conversation = FakeConversation({})
    checkin = CheckIn(
        kind="morning", store=store, journal=journal,
        presence=presence, voice=voice, notifier=notifier,
        conversation_factory=lambda conv_id, local: conversation,
        config=ScheduleConfig(), tz=LAGOS, clock=clock,
    )
    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=presence, voice=voice,
        notifier=notifier, checkins={"checkin_morning": checkin},
        clock=clock, mic=_FailingMic(),
    )

    instance.start()
    assert instance.degraded is True
    # start() itself already sent one notification ("mic unavailable") --
    # notifier is the SAME object handed to both Daemon and CheckIn here,
    # matching build_daemon()'s real wiring. What this test is actually
    # about is the notification checkin.run() sends on TOP of that.
    sent_by_daemon_startup = len(notifier.sent)

    outcome = checkin.run(NOW)

    assert len(notifier.sent) == sent_by_daemon_startup + 1
    assert notifier.sent[-1] == ("ZEUS", "Morning check-in")
    assert voice.spoken == []
    assert outcome is Outcome.DEFERRED
