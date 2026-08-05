"""Single-owner microphone stream with pre-roll ring buffer. See spec §5.2.

Exactly one CoreAudio input stream exists in the process. It fans out to
two consumers: the wake-word detector (continuous) and utterance capture
(on demand).

The ring buffer is the reason the first words after a wake word are not
lost. Without it, "Zeus, what's my battery" arrives as "battery", because
the detector is still deciding while the user keeps talking.
"""
from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from typing import Iterator

log = logging.getLogger(__name__)

FRAME_SAMPLES = 1280        # 80 ms at 16 kHz — openWakeWord's expected chunk
BYTES_PER_SAMPLE = 2        # int16
_QUEUE_MAX = 256
_SENTINEL = object()


class RingBuffer:
    """Fixed-length FIFO of raw audio frames."""

    def __init__(self, max_frames: int) -> None:
        self._frames: deque[bytes] = deque(maxlen=max_frames)
        self._lock = threading.Lock()

    def push(self, frame: bytes) -> None:
        with self._lock:
            self._frames.append(frame)

    def snapshot(self) -> bytes:
        with self._lock:
            return b"".join(self._frames)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()


class MicStream:
    def __init__(self, config) -> None:
        self._config = config
        frames_per_second = config.sample_rate / FRAME_SAMPLES
        self._ring = RingBuffer(int(config.ring_seconds * frames_per_second))
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._stream = None
        self._running = False
        self.dropped = 0

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        if self._running:
            raise RuntimeError("MicStream already running")
        import sounddevice as sd

        self._stream = sd.RawInputStream(
            samplerate=self._config.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=self._on_audio,
        )
        self._stream.start()
        self._running = True
        log.info("microphone stream started at %d Hz", self._config.sample_rate)

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                log.debug("error closing stream", exc_info=True)
            self._stream = None
        self._queue.put(_SENTINEL)

    def __enter__(self) -> "MicStream":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # -- audio path ----------------------------------------------------
    def _on_audio(self, indata, frames, time_info, status) -> None:
        """sounddevice callback. Must never raise — it runs on the audio thread."""
        if status:
            log.debug("audio status: %s", status)
        frame = bytes(indata)
        self._ring.push(frame)
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def frames(self) -> Iterator[bytes]:
        """Blocking iterator over live frames. Ends when stop() is called."""
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            yield item

    def pre_roll(self) -> bytes:
        """Audio captured just before now — prepended to a new utterance."""
        return self._ring.snapshot()

    def drain(self) -> None:
        """Discard queued frames, e.g. after ZEUS finishes speaking."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
