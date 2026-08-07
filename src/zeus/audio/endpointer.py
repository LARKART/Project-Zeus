"""Utterance boundary detection by energy and silence run-length.

Deliberately simple: no VAD model, no extra dependency. A neural VAD would
be more robust in noise, but this runs on every 80 ms frame on an Intel CPU
that is already paying for wake-word inference and Whisper.
"""
from __future__ import annotations

import math
from typing import Iterable, Iterator

from zeus.audio.mic import FRAME_SAMPLES

_INT16_MAX = 32768.0
_FRAME_SECONDS = FRAME_SAMPLES / 16000.0


def rms(frame: bytes) -> float:
    """Root-mean-square amplitude of an int16 frame, normalised to [0, 1]."""
    if not frame:
        return 0.0
    import numpy as np

    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(samples ** 2))) / _INT16_MAX)


class Endpointer:
    """Fires once speech has been heard and then a sustained silence run."""

    def __init__(self, config, threshold: float = 0.02) -> None:
        self._threshold = threshold
        self._silence_frames_needed = max(
            1, round(config.silence_timeout.total_seconds() / _FRAME_SECONDS)
        )
        self.saw_speech = False
        self._silence_run = 0

    def reset(self) -> None:
        self.saw_speech = False
        self._silence_run = 0

    def feed(self, frame: bytes) -> bool:
        """Returns True when the utterance is judged complete."""
        if rms(frame) >= self._threshold:
            self.saw_speech = True
            self._silence_run = 0
            return False
        if not self.saw_speech:
            return False        # leading silence does not end anything
        self._silence_run += 1
        return self._silence_run >= self._silence_frames_needed


def capture_utterance(
    frames: Iterable[bytes],
    endpointer: Endpointer,
    pre_roll: bytes,
    listen_timeout_frames: int,
) -> bytes:
    """Assemble pre-roll plus live audio until the endpointer or timeout fires.

    Returns b"" when no speech was heard at all, which the caller treats as
    a NO_ANSWER (spec §9.3).
    """
    endpointer.reset()
    collected: list[bytes] = [pre_roll] if pre_roll else []
    consumed = 0
    iterator: Iterator[bytes] = iter(frames)

    for frame in iterator:
        collected.append(frame)
        consumed += 1
        if endpointer.feed(frame):
            break
        if consumed >= listen_timeout_frames:
            break

    if not endpointer.saw_speech:
        return b""
    return b"".join(collected)
