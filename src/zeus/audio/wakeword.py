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
        # No mic.drain() here: while muted, events() keeps pulling frames
        # off the mic queue via the `for frame in self._mic.frames()` loop
        # below and simply skips scoring them (`if self._muted: continue`),
        # so no backlog accumulates during a normal mute/unmute cycle where
        # events() is being iterated throughout. Draining on unmute would
        # instead discard whatever frame is sitting in the queue at the
        # moment unmute() runs, regardless of whether it arrived before
        # mute() or during it — including genuine audio that was queued
        # before events() ever started consuming.
        self._muted = False

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
