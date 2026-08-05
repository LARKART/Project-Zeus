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
