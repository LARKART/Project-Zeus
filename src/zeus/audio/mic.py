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
_POLL_SECONDS = 0.1


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
        self._stopping = threading.Event()
        # Guards the check-and-set of _running in start()/stop() so two
        # concurrent start() calls cannot both pass the check and each
        # construct a RawInputStream. Never taken in _on_audio() — that
        # runs on the real-time audio thread and must never contend with
        # a slow caller holding this lock.
        self._lifecycle_lock = threading.Lock()
        self.dropped = 0

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        with self._lifecycle_lock:
            if self._running:
                raise RuntimeError("MicStream already running")
            import sounddevice as sd

            # Cleared here, not in stop(): a restarted stream must not
            # inherit the previous run's shutdown signal.
            self._stopping.clear()
            self._stream = sd.RawInputStream(
                samplerate=self._config.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
                callback=self._on_audio,
            )
            self._stream.start()
            self._running = True
        # A restarted stream must not replay stale audio left over from
        # the previous run.
        self.drain()
        log.info("microphone stream started at %d Hz", self._config.sample_rate)

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._running = False
            # Set BEFORE closing the device so a consumer blocked in
            # frames() is released even if the close path raises. An
            # Event, not a queue sentinel: the queue is bounded, so
            # enqueuing a shutdown signal deadlocks precisely when
            # shutdown matters most — a full queue.
            self._stopping.set()
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    log.debug("error closing stream", exc_info=True)
                self._stream = None

    def __enter__(self) -> "MicStream":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # -- audio path ----------------------------------------------------
    def _on_audio(self, indata, frames, time_info, status) -> None:
        """sounddevice callback. Must never raise — it runs on the audio thread."""
        if self._stopping.is_set():
            # Shutdown is signalled: stop feeding the queue. frames() ends by
            # draining to empty, so a producer that keeps pushing after stop()
            # would starve it of that observation and hang the consumer. The
            # device close in stop() is best-effort and swallows exceptions,
            # so we cannot assume the callback has actually ceased.
            return
        if status:
            log.debug("audio status: %s", status)
        frame = bytes(indata)
        self._ring.push(frame)
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def frames(self) -> Iterator[bytes]:
        """Blocking iterator over live frames. Ends when stop() is called.

        Polls with a short timeout rather than blocking indefinitely so that
        stop() is observed promptly without needing anything pushed onto the
        queue. Deviation from the reviewed fix snippet, flagged: the check
        is done only once the queue is empty, not at the top of every loop
        iteration. Checking eagerly would drop any frames already queued at
        the moment stop() was called before frames() started consuming —
        that regresses test_frames_iterator_yields_pushed_audio, which
        (correctly) expects a drain-then-stop iterator, not abandon-on-stop.
        """
        while True:
            try:
                yield self._queue.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                if self._stopping.is_set():
                    return

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
