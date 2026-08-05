"""Local faster-whisper transcription. Default provider for Slice 1 (spec D2).

Runs on CPU with int8 quantisation — the machine is an Intel Mac with no
GPU, so this trades a few seconds of latency for zero cost and audio that
never leaves the device.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def pcm_to_float32(pcm: bytes) -> np.ndarray:
    """Convert 16-bit signed PCM to the float32 [-1, 1] range Whisper wants."""
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


class LocalWhisper:
    def __init__(self, model: str, compute: str, models_dir: Path) -> None:
        self._model_name = model
        self._compute = compute
        self._models_dir = models_dir
        self._model = None

    def _load(self):
        from faster_whisper import WhisperModel

        self._models_dir.mkdir(parents=True, exist_ok=True)
        log.info("loading whisper model %s (%s)", self._model_name, self._compute)
        return WhisperModel(
            self._model_name,
            device="cpu",
            compute_type=self._compute,
            download_root=str(self._models_dir),
        )

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        if not pcm:
            return ""
        try:
            if self._model is None:
                self._model = self._load()
            segments, _ = self._model.transcribe(
                pcm_to_float32(pcm), language="en", beam_size=1
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
        except Exception:
            log.error("transcription failed", exc_info=True)
            return ""
