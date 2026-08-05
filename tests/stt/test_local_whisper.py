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


def test_factory_rejects_unknown_provider(tmp_path):
    with pytest.raises(ValueError, match="unknown STT provider"):
        build_transcriber(SttConfig(provider="deepgram"), tmp_path)
