"""openWakeWord activation over the shared mic stream.

RISK R2: openWakeWord ships no 'zeus' model, so the default is
'hey_jarvis'. `WakeConfig.model` is a name or a path to an .onnx file, so a
custom model is a config edit rather than a code change.
"""
from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from zeus.audio.activator import ActivationEvent
from zeus.config import WakeConfig

log = logging.getLogger(__name__)


class WakeWordActivator:
    def __init__(self, mic, config: WakeConfig, threshold: float = 0.5) -> None:
        self._mic = mic
        self._config = config
        self._threshold = threshold
        self._model = None
        self._muted = False
        self._running = False

    def _load_model(self):
        import openwakeword
        from openwakeword.model import Model

        try:
            openwakeword.utils.download_models()
        except Exception:
            log.debug("wake model download skipped", exc_info=True)
        return Model(wakeword_models=[self._config.model])

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def mute(self) -> None:
        """Suppress detection while ZEUS is speaking (spec §7.3, half-duplex)."""
        self._muted = True

    def unmute(self) -> None:
        """Re-enable detection after ZEUS has finished speaking.

        Drains the mic queue first. While muted, events() is typically
        suspended at its yield mid-conversation rather than consuming
        frames, so the queue holds ZEUS's own voice. Without this drain the
        detector scores all of it on resume and re-triggers itself — the
        exact feedback loop muting exists to prevent. Measured: 51 queued
        frames of self-audio, all 51 scored after unmute.
        """
        self._muted = False
        self._mic.drain()

    def events(self) -> Iterator[ActivationEvent]:
        if self._model is None:
            self._model = self._load_model()
        for frame in self._mic.frames():
            if not self._running:
                return
            if self._muted:
                continue
            samples = np.frombuffer(frame, dtype=np.int16)
            try:
                scores = self._model.predict(samples)
            except Exception:
                log.error("wake-word inference failed", exc_info=True)
                continue
            if any(score >= self._threshold for score in scores.values()):
                yield ActivationEvent("wake")
