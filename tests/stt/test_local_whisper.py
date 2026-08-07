import numpy as np
import pytest

from zeus.config import SttConfig
from zeus.stt import build_transcriber
from zeus.stt.local_whisper import LocalWhisper, pcm_to_float32


def test_pcm_to_float32_normalises_to_unit_range():
    pcm = np.array([0, 32767, -32768], dtype=np.int16).tobytes()
    floats = pcm_to_float32(pcm)
    assert floats.dtype == np.float32
    assert floats[0] == pytest.approx(0.0)
    assert floats[1] == pytest.approx(1.0, abs=1e-4)
    assert floats[2] == pytest.approx(-1.0, abs=1e-4)


def test_empty_audio_short_circuits_without_loading_a_model(tmp_path):
    stt = LocalWhisper("base.en", "int8", tmp_path)
    assert stt.transcribe(b"", 16000) == ""
    assert stt._model is None  # never loaded


def test_transcribe_joins_segments(monkeypatch, tmp_path):
    class Segment:
        def __init__(self, text):
            self.text = text

    class DummyModel:
        def transcribe(self, audio, **kwargs):
            return [Segment(" Finish the"), Segment(" auth flow ")], None

    stt = LocalWhisper("base.en", "int8", tmp_path)
    monkeypatch.setattr(stt, "_load", lambda: DummyModel())
    pcm = np.zeros(16000, dtype=np.int16).tobytes()
    assert stt.transcribe(pcm, 16000) == "Finish the auth flow"


def test_model_failure_returns_empty_not_an_exception(monkeypatch, tmp_path):
    class Exploding:
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("ctranslate2 exploded")

    stt = LocalWhisper("base.en", "int8", tmp_path)
    monkeypatch.setattr(stt, "_load", lambda: Exploding())
    pcm = np.zeros(16000, dtype=np.int16).tobytes()
    assert stt.transcribe(pcm, 16000) == ""


def test_concurrent_first_calls_load_the_model_exactly_once(monkeypatch, tmp_path):
    """A3: the lazy load was check-then-act across two threads.

    The daemon builds ONE transcriber and hands it to both the main thread
    (scheduled check-ins) and the wake thread (ad-hoc conversations), and
    `if self._model is None: self._model = self._load()` lets both pass the
    check. Measured before the fix: 4 concurrent first-calls produced 4
    WhisperModel loads, each writing the same files into models_dir.

    The barrier makes the race deterministic rather than hoping the threads
    interleave, and the sleep inside _load models the seconds a real
    WhisperModel takes to come up -- the window the second thread walks
    into.
    """
    import threading
    import time

    class Segment:
        text = " hello "

    class DummyModel:
        def transcribe(self, audio, **kwargs):
            return [Segment()], None

    loads = []
    ready = threading.Barrier(4)

    def slow_load():
        loads.append(1)
        time.sleep(0.05)
        return DummyModel()

    stt = LocalWhisper("base.en", "int8", tmp_path)
    monkeypatch.setattr(stt, "_load", slow_load)
    pcm = np.zeros(1600, dtype=np.int16).tobytes()

    heard = []

    def call():
        ready.wait(timeout=5)
        heard.append(stt.transcribe(pcm, 16000))

    threads = [threading.Thread(target=call) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert len(loads) == 1, (
        f"{len(loads)} concurrent model loads; each writes the same files "
        "into models_dir at the same time"
    )
    assert heard == ["hello"] * 4


def test_factory_rejects_unknown_provider(tmp_path):
    with pytest.raises(ValueError, match="unknown STT provider"):
        build_transcriber(SttConfig(provider="deepgram"), tmp_path)
