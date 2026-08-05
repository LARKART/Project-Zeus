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


def _wake_activator(monkeypatch, scores, frame_count):
    mic = MicStream(AudioConfig())
    for _ in range(frame_count):
        mic._on_audio(FRAME, FRAME_SAMPLES, None, None)
    mic.stop()  # sentinel terminates frames()

    activator = WakeWordActivator(mic, WakeConfig(), threshold=0.5)
    monkeypatch.setattr(activator, "_load_model", lambda: DummyModel(scores))
    return activator


def test_wake_word_fires_above_threshold(monkeypatch):
    activator = _wake_activator(monkeypatch, [0.1, 0.9, 0.1], 3)
    activator.start()
    assert [e.source for e in activator.events()] == ["wake"]


def test_wake_word_ignores_scores_below_threshold(monkeypatch):
    activator = _wake_activator(monkeypatch, [0.1, 0.2, 0.3], 3)
    activator.start()
    assert list(activator.events()) == []


def test_muting_suppresses_detection(monkeypatch):
    """Half-duplex rule, spec §7.3: ZEUS must not hear itself speaking."""
    activator = _wake_activator(monkeypatch, [0.9, 0.9, 0.9], 3)
    activator.start()
    activator.mute()
    assert list(activator.events()) == []


def test_unmute_restores_detection(monkeypatch):
    activator = _wake_activator(monkeypatch, [0.9], 1)
    activator.start()
    activator.mute()
    activator.unmute()
    assert [e.source for e in activator.events()] == ["wake"]


def test_factory_builds_wake_word_activator():
    mic = MicStream(AudioConfig())
    assert isinstance(build_activator(WakeConfig(), mic), WakeWordActivator)


def test_factory_rejects_unknown_provider():
    mic = MicStream(AudioConfig())
    with pytest.raises(ValueError, match="unknown wake provider"):
        build_activator(WakeConfig(provider="porcupine"), mic)


def test_activation_event_is_hashable_and_comparable():
    assert ActivationEvent("wake") == ActivationEvent("wake")
