import sys
import threading
import time
import types

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


def test_callback_feeds_both_the_ring_and_every_subscriber():
    # A subscription first: the callback broadcasts to the queues that
    # exist when it fires, and a fresh subscription starts empty.
    stream = MicStream(AudioConfig())
    subscription = stream.subscribe()
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)

    assert stream.pre_roll() == FRAME
    assert subscription._queue.get_nowait() == FRAME


def test_pre_roll_survives_queue_consumption():
    """The whole point: pre-roll must still be there after frames are read."""
    stream = MicStream(AudioConfig())
    subscription = stream.subscribe()
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    subscription._queue.get_nowait()
    assert stream.pre_roll() == FRAME


def test_frames_iterator_yields_pushed_audio():
    stream = MicStream(AudioConfig())
    stream._running = True
    subscription = stream.subscribe()
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream.stop()  # sentinel ends the iterator

    assert list(subscription.frames()) == [FRAME, FRAME]


def test_dropped_frames_are_counted_not_raised():
    stream = MicStream(AudioConfig())
    subscription = stream.subscribe()
    subscription._queue.maxsize = 1
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)  # queue full
    # Counted per subscriber as well as stream-wide: one slow consumer
    # must be distinguishable from a stream that is dropping for everyone.
    assert subscription.dropped == 1
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
    subscription = stream.subscribe()
    for _ in range(subscription._queue.maxsize):
        subscription._queue.put_nowait(FRAME)
    assert subscription._queue.full()

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
    subscription = stream.subscribe()
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream.stop()
    assert stream._stopping.is_set()

    # Simulate what start() does on restart, without a real device. The
    # stream-wide drain is safe only here, before the device is running:
    # no capture can be in flight to lose.
    stream._stopping.clear()
    for existing in stream._subscribers:
        existing.drain()
    stream._running = True

    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream.stop()

    assert list(subscription.frames()) == [FRAME, FRAME]


def test_start_drains_stale_audio_before_it_starts_the_device(monkeypatch):
    """F4 (round 5): the stream-wide drain in start() must run BEFORE the
    device does.

    start() used to drain after self._stream.start(), under a comment
    reading "the device is not running yet, so no capture can be in flight
    to lose". That comment was false, and it is the only thing telling a
    future reader when a stream-wide drain is safe -- everywhere else one is
    the bug that deletes an in-flight answer. Harmless today only because
    nothing is subscribed at mic.start(); the ordering is what makes the
    rule true, so the ordering is what gets pinned.

    No hardware: sounddevice is imported inside start(), so a fake module
    stands in and records the queue depth at the moment the device starts.
    """
    observed = {}

    class _FakeDevice:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            observed["queued_at_device_start"] = subscription._queue.qsize()

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(RawInputStream=lambda **kw: _FakeDevice(**kw)),
    )

    stream = MicStream(AudioConfig())
    subscription = stream.subscribe()
    subscription._queue.put_nowait(FRAME)   # stale audio from the last run

    stream.start()
    stream.stop()

    assert observed["queued_at_device_start"] == 0, (
        "start() began capturing before it drained, so the drain now runs "
        "against a live device -- exactly the stream-wide drain the comment "
        "says is never safe while running"
    )


def test_frames_terminates_even_if_the_callback_keeps_firing():
    """F3: frames() must not be starved of termination by a producer that
    keeps pushing after stop() — e.g. because stop()'s device close raised
    and was swallowed, leaving the sounddevice callback still firing.

    Drain-then-stop means frames() only returns once it observes an empty
    queue; a producer that never lets the queue go empty would otherwise
    hang the consumer forever. _on_audio must become a no-op once shutdown
    is signalled so the queue can actually drain.
    """
    stream = MicStream(AudioConfig())
    stream._running = True
    stream.stop()
    assert stream._stopping.is_set()

    keep_producing = threading.Event()
    keep_producing.set()

    def produce():
        while keep_producing.is_set():
            stream._on_audio(FRAME, FRAME_SAMPLES, None, None)

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()

    finished = threading.Event()

    def consume():
        for _ in stream.frames():
            pass
        finished.set()

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()

    completed = finished.wait(2)
    keep_producing.clear()

    assert completed, "frames() did not terminate within 2s while the callback kept firing"


# -- round 4, C2: one stream, many consumers ---------------------------
#
# The daemon builds ONE MicStream and hands it to TWO consumers: the wake
# detector (iterating frames() forever on its own thread) and utterance
# capture (calling frames() on the main thread during a check-in). A single
# shared queue is a hand-off, not a fan-out -- queue.get() REMOVES the
# frame, so each frame reached exactly one of them and the detector ate
# roughly half of the user's spoken answer.

PRIMER = b"\x7f\x7f" * FRAME_SAMPLES


def _numbered(index: int) -> bytes:
    """A frame whose bytes identify it, so a split can be seen, not guessed."""
    return bytes([index, 0]) * FRAME_SAMPLES


def test_two_concurrent_consumers_each_receive_every_frame():
    """C2, reproduced: 20 frames pushed to two concurrent consumers ended up
    as 20 for one of them and 0 for the other, with ZERO frames reaching
    both.

    Written against the public frames() API so it reproduces the defect on
    the old single-queue MicStream as well as passing on the fan-out one.
    The priming loop exists because a consumer only registers when its
    generator is first advanced: keep feeding throwaway frames until BOTH
    consumers have read one, so the payload below cannot be pushed into a
    queue nobody is watching yet.
    """
    stream = MicStream(AudioConfig())
    stream._running = True
    seen: dict[str, list[bytes]] = {"a": [], "b": []}
    live = {"a": threading.Event(), "b": threading.Event()}

    def consume(name: str) -> None:
        for frame in stream.frames():
            if frame == PRIMER:
                live[name].set()
            else:
                seen[name].append(frame)

    threads = [
        threading.Thread(target=consume, args=(name,), daemon=True)
        for name in ("a", "b")
    ]
    for thread in threads:
        thread.start()

    deadline = time.monotonic() + 5.0
    while not (live["a"].is_set() and live["b"].is_set()):
        assert time.monotonic() < deadline, "both consumers never started reading"
        stream._on_audio(PRIMER, FRAME_SAMPLES, None, None)
        time.sleep(0.002)

    payload = [_numbered(i) for i in range(20)]
    for frame in payload:
        stream._on_audio(frame, FRAME_SAMPLES, None, None)
    stream.stop()

    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert seen["a"] == payload, "consumer A missed frames another consumer took"
    assert seen["b"] == payload, "consumer B missed frames another consumer took"


def test_each_subscription_has_its_own_queue():
    stream = MicStream(AudioConfig())
    stream._running = True
    detector = stream.subscribe()
    listener = stream.subscribe()

    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream.stop()

    assert list(detector.frames()) == [FRAME]
    assert list(listener.frames()) == [FRAME]


def test_drain_empties_only_the_callers_own_queue():
    """WakeWordActivator.unmute() drains. If that drain were stream-wide it
    would delete the in-flight answer of a check-in listening on the main
    thread -- the very bug the fan-out exists to prevent."""
    stream = MicStream(AudioConfig())
    stream._running = True
    detector = stream.subscribe()
    listener = stream.subscribe()

    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    detector.drain()
    stream.stop()

    assert list(detector.frames()) == []
    assert list(listener.frames()) == [FRAME]


def test_a_closed_subscription_stops_collecting_frames():
    """Without unsubscribe, every finished check-in would leave a queue
    behind that the audio callback keeps filling forever."""
    stream = MicStream(AudioConfig())
    stream._running = True
    with stream.subscribe() as subscription:
        pass

    assert stream._subscribers == []
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream.stop()
    assert list(subscription.frames()) == []


# -- A1 (CRITICAL): the device stops delivering ------------------------


def test_frames_ends_when_the_device_stops_delivering(monkeypatch):
    """A1: frames() must end on a silent device, not wait forever.

    Nothing here calls stop(): the stream is still "running", the callback
    has simply ceased firing -- what CoreAudio does on sleep/wake, an
    AirPods connect, a USB mic unplug or a coreaudiod restart. Before the
    idle bound, `_stopping` was the iterator's ONLY exit, so this shape
    parked the consumer forever, and every bound above it counts FRAMES,
    which cannot advance when no frame arrives.

    The timeout constant is monkeypatched so the test costs 0.3s instead of
    5s. The 5.0s production value is a judgement call documented in mic.py;
    what is pinned here is the mechanism, and that it is armed at all.
    """
    monkeypatch.setattr("zeus.audio.mic._IDLE_TIMEOUT_SECONDS", 0.3)

    stream = MicStream(AudioConfig())
    stream._running = True
    subscription = stream.subscribe()
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)

    collected = []
    finished = threading.Event()

    def consume():
        collected.extend(subscription.frames())
        finished.set()

    threading.Thread(target=consume, daemon=True).start()

    assert finished.wait(5), (
        "frames() never returned on a device that stopped delivering -- "
        "the daemon's tick loop runs on that thread, so the heartbeat "
        "stops, no further check-in fires, and the process stays ALIVE so "
        "launchd's KeepAlive never restarts it"
    )
    assert collected == [FRAME], "the frames delivered before the silence were lost"
    assert not stream._stopping.is_set(), (
        "the test proved the wrong thing: the stream was stopped, which is "
        "the exit path that already worked"
    )


def test_the_idle_bound_does_not_fire_while_frames_keep_arriving(monkeypatch):
    """The other half: a live device must never be judged dead.

    Frames arrive during silence too -- the endpointer needs them to detect
    the end of an utterance -- so an idle bound that did not reset on every
    delivered frame would cut a long answer off mid-sentence.
    """
    monkeypatch.setattr("zeus.audio.mic._IDLE_TIMEOUT_SECONDS", 0.3)

    stream = MicStream(AudioConfig())
    stream._running = True
    subscription = stream.subscribe()

    collected = []
    finished = threading.Event()

    def consume():
        collected.extend(subscription.frames())
        finished.set()

    threading.Thread(target=consume, daemon=True).start()

    # 0.75s of delivery -- well past the 0.3s bound, in 0.05s steps.
    for _ in range(15):
        stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
        time.sleep(0.05)
    assert not finished.is_set(), (
        "the idle bound fired while the device was still delivering audio"
    )

    stream.stop()
    assert finished.wait(5)
    assert len(collected) == 15


def test_frames_closes_its_one_off_subscription_on_exit():
    """mic.frames() is a convenience wrapper around subscribe(); the queue it
    opens must not outlive the iteration."""
    stream = MicStream(AudioConfig())
    stream._running = True
    stream.stop()
    assert list(stream.frames()) == []
    assert stream._subscribers == []
