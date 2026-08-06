from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from zoneinfo import ZoneInfo

from zeus.audio.activator import FakeActivator
from zeus.audio.mic import FRAME_SAMPLES, MicStream
from zeus.clock import FakeClock
from zeus.config import AudioConfig, Config
from zeus.daemon import Daemon, audio_self_test, catch_up_actions
from zeus.memory.journal import Journal
from zeus.memory.store import Store
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


def test_self_test_passes_on_real_audio():
    assert audio_self_test(_mic([SPEECH] * 13), seconds=1.0) is True


def test_self_test_fails_on_pure_silence():
    """Risk R1: a TCC-denied stream opens fine and returns silence forever."""
    assert audio_self_test(_mic([SILENCE] * 13), seconds=1.0) is False


def test_self_test_fails_when_no_frames_arrive():
    assert audio_self_test(_mic([]), seconds=1.0) is False


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
