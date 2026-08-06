"""openWakeWord activation over the shared mic stream.

RISK R2: openWakeWord ships no 'zeus' model, so the default is
'hey_jarvis'. `WakeConfig.model` is a name or a path to an .onnx file, so a
custom model is a config edit rather than a code change.
"""
from __future__ import annotations

import logging
import threading
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
        # Depth, not a flag: mute windows nest across two threads.
        self._mute_depth = 0
        self._mute_lock = threading.Lock()

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
        """Suppress detection while ZEUS is speaking (spec §7.3, half-duplex).

        A DEPTH COUNTER, not a boolean. The daemon shares one VoiceIO and
        one activator between the main thread (scheduled check-ins) and the
        wake thread (ad-hoc conversations), and they overlap in practice:
        the user says "hey zeus" at 08:59:55, listen() mutes and opens a
        window of up to 30s; the morning check-in fires at 09:00:00 on the
        main thread, speaks its opener, and its `finally: unmute()` clears
        the mute while the wake thread is STILL capturing. Reproduced: the
        detector scored 5 of 5 frames of the user's in-flight answer. With
        a counter, the inner unmute decrements to 1 and the detector stays
        muted until the outer window closes.
        """
        with self._mute_lock:
            self._mute_depth += 1
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

        Only the OUTERMOST unmute actually unmutes — see mute() for why.

        Clamped at zero as DEFENCE IN DEPTH, not because anything emits a
        stray unmute today: both call sites (VoiceIO.speak and
        VoiceIO.listen) run mute() immediately before a try/finally that
        unmutes, and no activator defines unmute without mute, so the
        `if unmute:` guard never fires unpaired. An earlier version of this
        docstring claimed VoiceIO "calls unmute() in a finally whether or
        not the paired mute() ran" — that is false, and worth correcting
        rather than deleting: without the clamp, ONE stray unmute would
        leave the depth at -1 and the next nested window would unwind a
        level early, unmuting while an outer window is still open.
        """
        with self._mute_lock:
            self._mute_depth = max(0, self._mute_depth - 1)
            if self._mute_depth > 0:
                return
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
