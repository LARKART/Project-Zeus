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
        # Set while events() is iterating; unmute() drains through it.
        self._subscription = None

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

        Drains THIS detector's own subscription, never the whole stream: a
        scheduled check-in capturing an answer on the main thread holds its
        own subscription, and a stream-wide drain would delete the user's
        in-flight reply.
        """
        self._muted = False
        if self._subscription is not None:
            self._subscription.drain()

    def events(self) -> Iterator[ActivationEvent]:
        if self._model is None:
            self._model = self._load_model()
        # subscribe() rather than mic.frames(): the detector is a long-lived
        # consumer that needs a stable handle for unmute() to drain. Closing
        # it on exit stops _on_audio filling a queue nobody reads.
        with self._mic.subscribe() as subscription:
            self._subscription = subscription
            try:
                yield from self._detect(subscription)
            finally:
                self._subscription = None

    def _detect(self, subscription) -> Iterator[ActivationEvent]:
        for frame in subscription.frames():
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
