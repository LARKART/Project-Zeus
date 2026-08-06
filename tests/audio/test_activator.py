import threading
import time

import numpy as np
import pytest

from zeus.audio.activator import (
    ActivationEvent,
    FakeActivator,
    HotkeyActivator,
    build_activator,
)
from zeus.audio.mic import FRAME_SAMPLES, MicStream
from zeus.audio.wakeword import WakeWordActivator
from zeus.config import AudioConfig, WakeConfig

FRAME = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()


def test_fake_activator_yields_then_stops():
    activator = FakeActivator(count=2)
    activator.start()
    assert [e.source for e in activator.events()] == ["fake", "fake"]
    assert activator.started is True


def test_hotkey_activator_fires_on_sentinel_file(tmp_path):
    trigger = tmp_path / "trigger"
    activator = HotkeyActivator(trigger, poll_seconds=0)
    activator.start()
    trigger.touch()
    event = next(activator.events())
    assert event.source == "hotkey"
    assert not trigger.exists()  # consumed
    activator.stop()


class DummyModel:
    """Stands in for openwakeword.Model."""

    def __init__(self, scores):
        self._scores = list(scores)

    def predict(self, samples):
        return {"hey_jarvis": self._scores.pop(0) if self._scores else 0.0}


def _wake_activator(monkeypatch, mic, scores):
    activator = WakeWordActivator(mic, WakeConfig(), threshold=0.5)
    monkeypatch.setattr(activator, "_load_model", lambda: DummyModel(scores))
    return activator


def _live_detector(activator, timeout=5.0):
    """Run events() on its own thread and wait until it has really subscribed.

    Under the fan-out MicStream, "the detector is live" means its generator
    is RUNNING. Merely calling events() builds a generator object that has
    subscribed to nothing, and every subscriber has a private queue that
    starts empty — so frames pushed before the first advance reach no queue
    at all. The old single-shared-queue MicStream hid that distinction:
    frames landed in the stream's one queue whether anyone was reading or
    not, which let a test look like it was exercising mute/unmute while
    proving nothing. Waiting on `_subscription` is what makes the ordering
    real rather than assumed.

    Returns (collected_events, finished) — `finished` is set once the
    generator ends, which happens when mic.stop() is called.
    """
    collected: list[ActivationEvent] = []
    finished = threading.Event()

    def run() -> None:
        collected.extend(activator.events())
        finished.set()

    threading.Thread(target=run, daemon=True).start()
    deadline = time.monotonic() + timeout
    while activator._subscription is None:
        assert time.monotonic() < deadline, "the detector never subscribed"
        time.sleep(0.005)
    return collected, finished


def _feed(mic, count):
    for _ in range(count):
        mic._on_audio(FRAME, FRAME_SAMPLES, None, None)


def test_wake_word_fires_above_threshold(monkeypatch):
    mic = MicStream(AudioConfig())
    activator = _wake_activator(monkeypatch, mic, [0.1, 0.9, 0.1])
    activator.start()
    collected, finished = _live_detector(activator)

    _feed(mic, 3)
    mic.stop()  # sentinel terminates frames()

    assert finished.wait(5), "the detector never finished after mic.stop()"
    assert [e.source for e in collected] == ["wake"]


def test_wake_word_ignores_scores_below_threshold(monkeypatch):
    mic = MicStream(AudioConfig())
    activator = _wake_activator(monkeypatch, mic, [0.1, 0.2, 0.3])
    activator.start()
    collected, finished = _live_detector(activator)

    _feed(mic, 3)
    mic.stop()

    assert finished.wait(5)
    assert collected == []


def test_muting_suppresses_detection(monkeypatch):
    """Half-duplex rule, spec §7.3: ZEUS must not hear itself speaking."""
    mic = MicStream(AudioConfig())
    model = CountingModel(scores=[0.9, 0.9, 0.9])
    activator = WakeWordActivator(mic, WakeConfig(), threshold=0.5)
    monkeypatch.setattr(activator, "_load_model", lambda: model)
    activator.start()
    collected, finished = _live_detector(activator)

    activator.mute()
    _feed(mic, 3)
    mic.stop()

    assert finished.wait(5)
    assert collected == []
    # Not merely "no event": muted frames must never reach the model at all.
    assert model.calls == 0


def test_unmute_restores_detection(monkeypatch):
    """unmute() must not leave the detector permanently deaf.

    The wake frame is queued only after unmute(), so unmute()'s drain
    (which empties the queue while it holds nothing) cannot be confused
    with discarding it.
    """
    mic = MicStream(AudioConfig())
    activator = _wake_activator(monkeypatch, mic, [0.9])
    activator.start()
    collected, finished = _live_detector(activator)

    activator.mute()
    activator.unmute()
    _feed(mic, 1)  # real audio, after unmute
    mic.stop()

    assert finished.wait(5)
    assert [e.source for e in collected] == ["wake"]


def test_mute_is_idempotent(monkeypatch):
    mic = MicStream(AudioConfig())
    activator = _wake_activator(monkeypatch, mic, [0.9])
    activator.start()
    collected, finished = _live_detector(activator)

    activator.mute()
    activator.mute()          # must not raise
    activator.unmute()        # a single unmute must still restore detection
    _feed(mic, 1)
    mic.stop()

    assert finished.wait(5)
    assert [e.source for e in collected] == ["wake"]


def test_unmute_without_mute_is_safe(monkeypatch):
    mic = MicStream(AudioConfig())
    activator = _wake_activator(monkeypatch, mic, [0.9])
    activator.start()
    collected, finished = _live_detector(activator)

    activator.unmute()        # never muted; must not raise, must not deafen
    activator.unmute()        # and again
    _feed(mic, 1)
    mic.stop()

    assert finished.wait(5)
    assert [e.source for e in collected] == ["wake"]


class CountingModel:
    """Counts predict() calls; proves muted-and-discarded frames are never scored."""

    def __init__(self, scores=()):
        self.calls = 0
        self._scores = list(scores)

    def predict(self, samples):
        self.calls += 1
        score = self._scores.pop(0) if self._scores else 0.0
        return {"hey_jarvis": score}


def test_unmute_discards_audio_captured_while_speaking(monkeypatch):
    """Regression: unmute() must drain frames queued while ZEUS was speaking.

    events() is a generator; mid-conversation it is suspended at its yield
    while the daemon handles the activation, so nothing is consuming the
    detector's queue during that window. Without a drain in unmute(), all
    of ZEUS's own speech queued during mute gets scored the instant
    detection resumes — the exact feedback loop mute() exists to prevent.

    The suspension is modelled explicitly rather than assumed: the consumer
    below parks at the yield exactly as Daemon._activation_loop does while
    _handle_activation runs. Without that, a live detector would consume
    and discard each muted frame as it arrived and the drain would be a
    no-op, which is not the situation the drain exists for.

    Also pins the OTHER half of the ruling: the drain must reach only this
    detector's own queue. tests/audio/test_mic.py::
    test_drain_empties_only_the_callers_own_queue covers the Subscription
    side; here the assertion is that unmute() goes through the
    subscription, so a check-in listening on the main thread keeps its
    audio.
    """
    mic = MicStream(AudioConfig())
    model = CountingModel(scores=[0.9])   # first frame wakes, rest score 0.0
    activator = WakeWordActivator(mic, WakeConfig(), threshold=0.5)
    monkeypatch.setattr(activator, "_load_model", lambda: model)
    activator.start()

    collected: list[ActivationEvent] = []
    activated = threading.Event()
    resume = threading.Event()
    finished = threading.Event()

    def run():
        for event in activator.events():
            collected.append(event)
            activated.set()
            # Suspended at the yield, exactly as the daemon is while it
            # runs the ad-hoc conversation.
            resume.wait(5)
        finished.set()

    threading.Thread(target=run, daemon=True).start()
    deadline = time.monotonic() + 5.0
    while activator._subscription is None:
        assert time.monotonic() < deadline, "the detector never subscribed"
        time.sleep(0.005)

    _feed(mic, 1)  # the wake word
    assert activated.wait(5), "the detector never fired"
    assert model.calls == 1

    activator.mute()
    _feed(mic, 20)  # ZEUS's own voice, piling up behind the suspended yield
    activator.unmute()

    resume.set()
    mic.stop()
    assert finished.wait(5)

    assert [e.source for e in collected] == ["wake"]
    assert model.calls == 1, (
        f"unmute() left {model.calls - 1} frames of ZEUS's own speech to be "
        "scored on resume — the feedback loop mute() exists to prevent"
    )


def test_factory_builds_wake_word_activator():
    mic = MicStream(AudioConfig())
    assert isinstance(build_activator(WakeConfig(), mic), WakeWordActivator)


def test_factory_rejects_unknown_provider():
    mic = MicStream(AudioConfig())
    with pytest.raises(ValueError, match="unknown wake provider"):
        build_activator(WakeConfig(provider="porcupine"), mic)


def test_activation_event_is_hashable_and_comparable():
    assert ActivationEvent("wake") == ActivationEvent("wake")
