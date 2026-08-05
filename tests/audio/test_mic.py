import threading

import pytest

from zeus.audio.mic import FRAME_SAMPLES, MicStream, RingBuffer
from zeus.config import AudioConfig

FRAME = b"\x01\x02" * FRAME_SAMPLES  # one 80 ms chunk


def test_ring_buffer_keeps_only_the_last_n_frames():
    ring = RingBuffer(max_frames=2)
    ring.push(b"aa")
    ring.push(b"bb")
    ring.push(b"cc")
    assert ring.snapshot() == b"bbcc"


def test_ring_buffer_clear():
    ring = RingBuffer(max_frames=2)
    ring.push(b"aa")
    ring.clear()
    assert ring.snapshot() == b""


def test_ring_capacity_is_derived_from_config():
    # 3 seconds at 16 kHz in 1280-sample frames == 37 frames
    stream = MicStream(AudioConfig(sample_rate=16000, ring_seconds=3))
    assert stream._ring._frames.maxlen == 37


def test_callback_feeds_both_the_ring_and_the_queue():
    stream = MicStream(AudioConfig())
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)

    assert stream.pre_roll() == FRAME
    assert stream._queue.get_nowait() == FRAME


def test_pre_roll_survives_queue_consumption():
    """The whole point: pre-roll must still be there after frames are read."""
    stream = MicStream(AudioConfig())
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream._queue.get_nowait()
    assert stream.pre_roll() == FRAME


def test_frames_iterator_yields_pushed_audio():
    stream = MicStream(AudioConfig())
    stream._running = True
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream.stop()  # sentinel ends the iterator

    assert list(stream.frames()) == [FRAME, FRAME]


def test_dropped_frames_are_counted_not_raised():
    stream = MicStream(AudioConfig())
    stream._queue.maxsize = 1
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)  # queue full
    assert stream.dropped == 1


def test_start_is_rejected_twice():
    stream = MicStream(AudioConfig())
    stream._running = True
    with pytest.raises(RuntimeError, match="already running"):
        stream.start()


# -- Task 11 review fixes: regression tests ----------------------------


def test_stop_does_not_block_when_the_queue_is_full():
    """F1 (CRITICAL): stop() must not deadlock when the audio queue is full.

    A shutdown signal is out-of-band and must not be enqueued onto the
    same bounded, in-band channel that carries audio frames.
    """
    stream = MicStream(AudioConfig())
    for _ in range(stream._queue.maxsize):
        stream._queue.put_nowait(FRAME)
    assert stream._queue.full()

    finished = threading.Event()

    def run_stop():
        stream.stop()
        finished.set()

    t = threading.Thread(target=run_stop, daemon=True)
    t.start()
    completed = finished.wait(2)

    assert completed, "stop() did not return within 2s while the queue was full"


def test_frames_terminates_after_stop_with_an_empty_queue():
    """F1 corollary: the poll loop must observe shutdown without a queued item."""
    stream = MicStream(AudioConfig())
    stream._running = True
    collected = []
    finished = threading.Event()

    def consume():
        for item in stream.frames():
            collected.append(item)
        finished.set()

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    stream.stop()
    completed = finished.wait(2)

    assert completed, "frames() did not terminate within 2s after stop()"
    assert collected == []


def test_restart_does_not_inherit_the_previous_stop():
    """F2 (IMPORTANT): a restarted stream must not inherit the prior shutdown
    signal, and must not replay stale audio left over from the previous run.

    Device-free: simulates what start() does (clear the stop signal, then
    drain leftover frames) without opening a real sounddevice stream.
    """
    stream = MicStream(AudioConfig())
    stream._running = True
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream.stop()
    assert stream._stopping.is_set()

    # Simulate what start() does on restart, without a real device.
    stream._stopping.clear()
    stream.drain()
    stream._running = True

    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream.stop()

    assert list(stream.frames()) == [FRAME, FRAME]
