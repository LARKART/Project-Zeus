import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from zoneinfo import ZoneInfo

from zeus.audio.activator import FakeActivator
from zeus.audio.mic import FRAME_SAMPLES, MicStream
from zeus.brain.fake import FakeConversation
from zeus.brain.prompts import NOT_CAUGHT_LINE
from zeus.brain.tools import build_tool_callables
from zeus.clock import FakeClock, from_utc_iso
from zeus.config import AudioConfig, Config, ScheduleConfig
from zeus.context.presence import Verdict
from zeus.daemon import (
    Daemon,
    DegradedPresence,
    SwitchablePresence,
    audio_self_test,
    build_daemon,
    catch_up_actions,
)
from zeus.memory.journal import Journal
from zeus.memory.store import Store
from zeus.ritual.checkin import CheckIn, FakeNotifier
from zeus.ritual.retry import Outcome
from zeus.schedule.scheduler import MissedRun, Scheduler

LAGOS = ZoneInfo("Africa/Lagos")
LOS_ANGELES = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)

SILENCE = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()
SPEECH = (np.ones(FRAME_SAMPLES, dtype=np.int16) * 5000).tobytes()


def _await(predicate, message, timeout=5.0):
    """Spin until `predicate` holds, or fail with `message`."""
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, message
        time.sleep(0.005)


def _mic(frames):
    """A real MicStream that delivers `frames` to whoever subscribes next.

    Frames can no longer be pre-loaded before a consumer exists: MicStream
    fans out to per-subscriber queues and a fresh subscription starts empty
    -- that emptiness is what stops the wake detector eating a check-in's
    answer (round 4, C2). So the frames are pushed from a background thread
    once a subscriber has appeared (audio_self_test subscribes when it
    advances mic.frames()), then the stream is stopped so frames()
    terminates. Daemon, not joined: if no subscriber ever appears the
    thread simply gives up after its own deadline rather than holding the
    suite.
    """
    mic = MicStream(AudioConfig())

    def feed():
        deadline = time.monotonic() + 5.0
        while not mic._subscribers:
            if time.monotonic() >= deadline:
                return
            time.sleep(0.005)
        for frame in frames:
            mic._on_audio(frame, FRAME_SAMPLES, None, None)
        mic.stop()

    threading.Thread(target=feed, daemon=True).start()
    return mic


class _RecordingMic:
    """A mic double that only records start()/stop() calls.

    Not a real MicStream: start() must never touch sounddevice, since
    building one requires a real audio device. Used to assert Daemon.stop()
    reaches the mic, without exercising MicStream.frames()'s own shutdown
    plumbing (that's covered separately by tests/audio/test_mic.py).
    """

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class _FailingMic:
    """A mic double whose self-test always fails: no frames, ever."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def frames(self):
        return iter(())


class _MicThatGoesQuietMidCapture:
    """Yields a couple of frames, then blocks forever.

    Models a dead-hardware variant of R1: the audio callback simply stops
    firing. A real MicStream in that state has nothing that would ever set
    its `_stopping` Event, so mic.frames() would hang the calling thread
    indefinitely -- there is no further frame and no StopIteration, so a
    generator-based `for` loop never gets control back. This reproduces
    that shape deterministically, without needing real hardware: the fresh
    Event below is never `set()`, by construction, so `.wait()` on it never
    returns.
    """

    def frames(self):
        yield SPEECH
        yield SPEECH
        threading.Event().wait()


class _SlowToStartMic:
    """Delivers nothing for a while, then real audio.

    Models the case M2 is about: a Bluetooth input switching into its HFP
    profile can take seconds before the first callback arrives. The
    self-test's deadline measures TOTAL capture, not time-to-first-frame,
    so too tight a budget makes a slow device indistinguishable from a dead
    one -- and the consequence is sticky, because `degraded` is never
    re-tested or cleared.
    """

    def __init__(self, delay: float) -> None:
        self._delay = delay

    def frames(self):
        time.sleep(self._delay)
        yield SPEECH


class _StubPresence:
    def __init__(self, verdict: Verdict) -> None:
        self._verdict = verdict

    def verdict(self) -> Verdict:
        return self._verdict


class _StubVoice:
    """Records what was spoken. Mirrors tests/ritual/test_checkin.py's
    StubVoice, kept local rather than imported since test modules don't
    share fixtures across files by convention here."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, sentences) -> None:
        self.spoken.extend(sentences)

    def listen(self) -> str:
        return ""


def test_self_test_passes_on_real_audio():
    assert audio_self_test(_mic([SPEECH] * 13), seconds=1.0) is True


def test_self_test_fails_on_pure_silence():
    """Risk R1: a TCC-denied stream opens fine and returns silence forever."""
    assert audio_self_test(_mic([SILENCE] * 13), seconds=1.0) is False


def test_self_test_fails_when_no_frames_arrive():
    assert audio_self_test(_mic([]), seconds=1.0) is False


def test_self_test_times_out_when_audio_stops_mid_capture():
    """Important (review round 2): `seconds` bounds a frame COUNT, not
    wall-clock. test_self_test_fails_when_no_frames_arrive does not cover
    this -- its `_mic()` helper calls mic.stop() first, so `_stopping` is
    already set and mic.frames() returns immediately. That tests "already
    stopped", not "still running but gone silent": if the real audio
    callback stops firing mid-capture, mic.frames() blocks forever with no
    frame and no StopIteration ever coming, and nothing sets `_stopping`
    during a self-test. Without an internal deadline this would hang
    Daemon.start() forever.

    `seconds=0.5` needs 6 frames (see audio_self_test's `wanted`
    calculation); the double below only ever produces 2 before blocking,
    so the third `next()` call is guaranteed to hit the block, not race it.

    audio_self_test's own fix is what must return False here -- wrapping
    the call in a background thread with a bounded wait is a safety net
    for THIS TEST (never call something that might hang directly in a test
    body), not a substitute for the fix: if audio_self_test regressed back
    to hanging forever, this thread would never finish either, and the
    assertion below would fail cleanly instead of hanging the suite.

    The safety net waits 10s because audio_self_test's own deadline is
    max(seconds * 5.0, 5.0) == 5.0s here (round 4, M2: the 5s budget is
    ruled, and a Bluetooth input's HFP profile switch is why). A net that
    is not strictly longer than the deadline it is guarding turns a passing
    fix into a failing test.
    """
    result: dict[str, bool] = {}
    finished = threading.Event()

    def run() -> None:
        result["value"] = audio_self_test(
            _MicThatGoesQuietMidCapture(), seconds=0.5
        )
        finished.set()

    threading.Thread(target=run, daemon=True).start()
    completed = finished.wait(timeout=10)

    assert completed, (
        "audio_self_test did not return within 10s -- the mid-capture "
        "deadline regressed"
    )
    assert result["value"] is False


def test_the_self_test_deadline_tolerates_a_slow_device():
    """M2 (round 4): the ruled deadline is max(seconds * 5.0, 5.0); the code
    had quietly tightened it to seconds * 2 + 1.0.

    seconds=0.1 wants a single frame and puts the two budgets far apart --
    1.2s under the tightened rule, 5.0s under the ruled one -- so a device
    that takes 1.5s to produce its first frame is judged DEAD by the
    tightened deadline and healthy by the ruled one. Picking that gap keeps
    the test's own cost to ~1.5s rather than the ~3.5s a `seconds=1.0`
    version would need.
    """
    result: dict[str, bool] = {}
    finished = threading.Event()

    def run() -> None:
        result["value"] = audio_self_test(_SlowToStartMic(1.5), seconds=0.1)
        finished.set()

    threading.Thread(target=run, daemon=True).start()
    assert finished.wait(timeout=10), "audio_self_test never returned"
    assert result["value"] is True, (
        "a device that took 1.5s to deliver its first frame was reported as "
        "a dead microphone -- the self-test deadline is too tight"
    )


def test_self_test_fails_when_a_real_mic_dies_part_way_through(monkeypatch):
    """A4, and the hole A1's own fix opened.

    Subscription.frames() now gives up after an idle timeout instead of
    parking forever, which fixes the leaked consumer thread -- but it also
    means `done` no longer means "the wanted frames arrived". It fires when
    frames() gave up too. A mic that delivered 2 of 6 frames and died would
    set `done`, sail past the deadline, find seen != 0 and energy != 0, and
    be reported HEALTHY: the exact R1 failure this function exists to
    catch, reintroduced through the back door.

    A REAL MicStream, since the whole point is the interaction with the
    real frames() bound; the idle timeout is shortened so it lands well
    inside the self-test deadline, which is precisely the ordering that
    makes `done` win the race.
    """
    monkeypatch.setattr("zeus.audio.mic._IDLE_TIMEOUT_SECONDS", 0.3)

    mic = MicStream(AudioConfig())

    def feed():
        deadline = time.monotonic() + 5.0
        while not mic._subscribers:
            if time.monotonic() >= deadline:
                return
            time.sleep(0.005)
        for _ in range(2):          # 2 frames, then the callback dies
            mic._on_audio(SPEECH, FRAME_SAMPLES, None, None)

    threading.Thread(target=feed, daemon=True).start()

    assert audio_self_test(mic, seconds=0.5) is False, (
        "a microphone that stopped delivering after 2 of 6 frames was "
        "reported as healthy"
    )
    # ...and the consumer thread is gone rather than left holding a live
    # subscription that _on_audio keeps filling forever (A4).
    deadline = time.monotonic() + 5.0
    while mic._subscribers and time.monotonic() < deadline:
        time.sleep(0.02)
    assert mic._subscribers == [], "the self-test leaked its subscription"

    for _ in range(50):
        mic._on_audio(SPEECH, FRAME_SAMPLES, None, None)
    assert mic.dropped == 0, (
        "an orphaned subscription is still swallowing frames and polluting "
        "MicStream.dropped"
    )


@pytest.mark.parametrize(
    "missed,expected",
    [
        # Morning missed earlier today → still worth asking
        ([MissedRun("checkin_morning", NOW, True)], [("checkin_morning", "fire")]),
        # Morning missed on a previous day → asking now is noise
        ([MissedRun("checkin_morning", NOW, False)], [("checkin_morning", "skip")]),
        # Evening is never replayed, even on the same day
        ([MissedRun("checkin_evening", NOW, True)], [("checkin_evening", "skip")]),
        ([MissedRun("checkin_evening", NOW, False)], [("checkin_evening", "skip")]),
    ],
)
def test_catch_up_policy(missed, expected):
    assert catch_up_actions(missed) == expected


def test_catch_up_policy_preserves_order():
    missed = [
        MissedRun("checkin_morning", NOW - timedelta(days=1), False),
        MissedRun("checkin_evening", NOW - timedelta(days=1), False),
        MissedRun("checkin_morning", NOW, True),
    ]
    assert [action for _, action in catch_up_actions(missed)] == [
        "skip", "skip", "fire",
    ]


def test_unknown_job_is_never_fired_by_catch_up():
    assert catch_up_actions([MissedRun("watchman_scan", NOW, True)]) == [
        ("watchman_scan", "skip")
    ]


def test_a_skipped_catch_up_is_not_refired_by_the_next_tick(tmp_path):
    """Critical #1 (review round 2): Scheduler.catch_up() reads the
    heartbeat while run_pending() reads each job's last_run_at -- separate
    state. Without run_catch_up() consuming the occurrence via
    store.set_job_run(), the very first tick() after catch-up recomputes
    from the stale last_run_at and overrides whatever catch-up just
    decided -- here, un-skipping a "skip" decision and asking about
    yesterday's evening after all, in violation of spec §9.2.

    This only reproduces on a restart, when the job row already carries a
    last_run_at -- with a fresh store, run_pending's first call merely
    seeds the baseline and fires nothing. The brief's own `daemon` fixture
    never calls run_pending before run_catch_up, so it cannot express this
    state; built directly on real Store/Scheduler/FakeClock instead.

    Uses checkin_evening, which is never catch-up-eligible under any
    circumstance (CATCH_UP_ELIGIBLE = {"checkin_morning"}), so the correct
    decision here is unconditionally "skip".
    """
    clock = FakeClock(datetime(2026, 8, 4, 20, 0, tzinfo=timezone.utc))
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    scheduler = Scheduler(store, clock, LAGOS)
    calls: list[datetime] = []
    scheduler.register("checkin_evening", "0 21 * * *", calls.append)

    # Prior run: yesterday's 21:00 Lagos (20:00 UTC) occurrence already
    # fired, so last_run_at is populated -- only true after a restart.
    store.set_job_run("checkin_evening", clock.now_utc())

    # Heartbeat this morning, well before today's 21:00 Lagos occurrence.
    clock.advance(timedelta(hours=13))  # 2026-08-05 09:00 UTC
    store.set_heartbeat()

    # "Restart" tonight, after today's 21:00 Lagos (20:00 UTC) occurrence
    # has already passed.
    clock.advance(timedelta(hours=13))  # 2026-08-05 22:00 UTC

    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={}, clock=clock,
    )

    actions = instance.run_catch_up()
    assert actions == [("checkin_evening", "skip")]
    assert calls == []

    instance.tick()

    assert calls == [], (
        "the tick immediately after catch-up re-fired a run catch-up had "
        "just decided to skip"
    )


def test_a_fired_catch_up_is_not_refired_by_the_next_tick(tmp_path):
    """Mirror of the skip case above: a run catch-up fires must not be
    fired AGAIN by the tick that immediately follows. Without consuming
    the occurrence, run_pending recomputes from the stale last_run_at,
    finds today's occurrence still due, and fires it a second time --
    CheckIn would either reopen the same row and re-run _converse
    (incrementing attempts a second time for one real occurrence) or, if
    the first attempt already resolved to answered, open a second row and
    ask the question again.

    checkin_morning is the only catch-up-eligible job.
    """
    clock = FakeClock(datetime(2026, 8, 4, 10, 0, tzinfo=timezone.utc))
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    scheduler = Scheduler(store, clock, LAGOS)
    calls: list[datetime] = []

    class _RecordingCheckIn:
        def run(self, scheduled_for):
            calls.append(scheduled_for)

    checkin = _RecordingCheckIn()
    scheduler.register("checkin_morning", "0 11 * * *", checkin.run)

    # Prior run: yesterday's 11:00 Lagos (10:00 UTC) occurrence already
    # fired, so last_run_at is populated -- only true after a restart.
    store.set_job_run("checkin_morning", clock.now_utc())

    # Heartbeat early this morning, before today's 11:00 Lagos occurrence.
    clock.advance(timedelta(hours=22))  # 2026-08-05 08:00 UTC
    store.set_heartbeat()

    # "Restart" after today's 11:00 Lagos (10:00 UTC) occurrence has passed.
    clock.advance(timedelta(hours=4))  # 2026-08-05 12:00 UTC

    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={"checkin_morning": checkin}, clock=clock,
    )

    actions = instance.run_catch_up()
    fired_at = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)
    assert actions == [("checkin_morning", "fire")]
    assert calls == [fired_at]

    instance.tick()

    assert calls == [fired_at], (
        "the tick immediately after catch-up re-fired a run catch-up had "
        "already fired"
    )


# ---- C-I4: an activation that heard nothing must not end in silence ------


def test_an_activation_that_heard_nothing_says_so(tmp_path):
    """Spec §10, the wake-word half. The user said "hey zeus" -- they ARE
    talking to it -- and `if not heard: return` walked away without a word.
    No conversation is started, so this costs no API call and still works
    when the brain is the thing that is down."""
    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    scheduler = Scheduler(store, clock, LAGOS)
    voice = _StubVoice()                     # listen() always returns ""
    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=voice, notifier=None,
        checkins={}, clock=clock,
    )

    instance._handle_activation()

    assert voice.spoken == [NOT_CAUGHT_LINE]
    assert store.messages(1) == [], "a conversation was started for no input"


# ---- C-I1: catch-up must record what it skipped --------------------------


def _skipping_daemon(tmp_path, tz, kind, cron, missed_at, restart_at):
    """A daemon restarting after downtime that swallowed one occurrence."""
    clock = FakeClock(missed_at - timedelta(hours=1))
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, tz)
    scheduler = Scheduler(store, clock, tz)

    class _RecordingCheckIn:
        def __init__(self):
            self.runs = []

        def run(self, scheduled_for):
            self.runs.append(scheduled_for)

    checkin = _RecordingCheckIn()
    name = f"checkin_{kind}"
    scheduler.register(name, cron, checkin.run)
    store.set_heartbeat()                    # last seen an hour before

    clock.advance(restart_at - clock.now_utc())
    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={name: checkin}, clock=clock, tz=tz,
    )
    return SimpleNamespace(
        daemon=instance, store=store, journal=journal, checkin=checkin,
    )


def test_a_skipped_evening_catch_up_is_recorded_as_skipped(tmp_path):
    """Spec §9.2: "Evening check-in missed -> record outcome=skipped".

    The skip branch logged a line and called set_job_run, and that was all
    -- nothing reached the `checkins` table, so a week of downtime left NO
    trace that seven evenings had been skipped. checkins.outcome has
    carried 'skipped' in its CHECK constraint from the first migration for
    exactly this, and D10's justification for building the action log in
    Slice 1 is that the Slice 2 dashboard can only show history that was
    recorded from the beginning.
    """
    rig = _skipping_daemon(
        tmp_path, LAGOS, "evening", "0 21 * * *",
        missed_at=datetime(2026, 8, 5, 20, 0, tzinfo=timezone.utc),
        restart_at=datetime(2026, 8, 5, 22, 0, tzinfo=timezone.utc),
    )

    assert rig.daemon.run_catch_up() == [("checkin_evening", "skip")]

    rows = rig.store.connection.execute(
        "SELECT kind, local_date, outcome, attempts, retry_at FROM checkins"
    ).fetchall()
    assert len(rows) == 1, f"expected one skipped check-in row, got {len(rows)}"
    assert rows[0]["kind"] == "evening"
    assert rows[0]["outcome"] == "skipped"
    assert rows[0]["local_date"] == "2026-08-05"
    # It never fired, so counting an attempt would be a lie in the one
    # column the §9.3 ladder reads; and a settled check-in must not still
    # be pending a retry.
    assert rows[0]["attempts"] == 0
    assert rows[0]["retry_at"] is None
    assert "skipped" in rig.journal.read("2026-08-05").lower()


def test_a_stale_morning_catch_up_is_recorded_as_skipped(tmp_path):
    """Spec §9.2: "Morning check-in missed, day has rolled over -> do not
    fire; record outcome=skipped"."""
    rig = _skipping_daemon(
        tmp_path, LAGOS, "morning", "0 11 * * *",
        missed_at=datetime(2026, 8, 4, 10, 0, tzinfo=timezone.utc),
        restart_at=datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc),
    )

    actions = rig.daemon.run_catch_up()

    assert ("checkin_morning", "skip") in actions
    skipped = rig.store.connection.execute(
        "SELECT local_date FROM checkins WHERE outcome = 'skipped'"
    ).fetchall()
    assert [r["local_date"] for r in skipped] == ["2026-08-04"], (
        "the skip was recorded against the wrong day"
    )


def test_the_skipped_row_uses_the_local_date_not_the_utc_one(tmp_path):
    """The seam that has broken this codebase twice already: the scheduler
    always produces UTC-tagged datetimes, so deriving the date from
    scheduled_for.date() writes the UTC calendar date. Los Angeles is where
    those diverge -- 21:00 PDT on the 5th is 04:00 UTC on the 6th -- and a
    row keyed on the wrong date is invisible to every later lookup."""
    rig = _skipping_daemon(
        tmp_path, LOS_ANGELES, "evening", "0 21 * * *",
        missed_at=datetime(2026, 8, 6, 4, 0, tzinfo=timezone.utc),
        restart_at=datetime(2026, 8, 6, 6, 0, tzinfo=timezone.utc),
    )

    rig.daemon.run_catch_up()

    row = rig.store.connection.execute(
        "SELECT local_date, scheduled_for FROM checkins"
    ).fetchone()
    assert row["local_date"] == "2026-08-05", (
        "the skipped evening was filed under the UTC date, not the local one"
    )
    assert from_utc_iso(row["scheduled_for"]) == datetime(
        2026, 8, 6, 4, 0, tzinfo=timezone.utc
    )


def test_a_fired_catch_up_is_not_also_recorded_as_skipped(tmp_path):
    """The morning ZEUS actually asks about must be left to CheckIn.run to
    record. A 'skipped' row written beside it would mark the day settled
    and hide the real outcome."""
    rig = _skipping_daemon(
        tmp_path, LAGOS, "morning", "0 11 * * *",
        missed_at=datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc),
        restart_at=datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc),
    )

    assert rig.daemon.run_catch_up() == [("checkin_morning", "fire")]

    assert len(rig.checkin.runs) == 1
    assert rig.store.connection.execute(
        "SELECT count(*) AS n FROM checkins WHERE outcome = 'skipped'"
    ).fetchone()["n"] == 0


def test_a_skip_settles_the_open_row_instead_of_opening_a_second(tmp_path):
    """A morning deferred before the daemon died, then rolled over: THAT
    row has to be settled, not a second one opened beside it. Two rows for
    one occurrence is how attempts stopped accumulating the last time."""
    rig = _skipping_daemon(
        tmp_path, LAGOS, "morning", "0 11 * * *",
        missed_at=datetime(2026, 8, 4, 10, 0, tzinfo=timezone.utc),
        restart_at=datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc),
    )
    existing = rig.store.open_checkin(
        "morning", datetime(2026, 8, 4, 10, 0, tzinfo=timezone.utc), "2026-08-04"
    )
    rig.store.update_checkin(existing, outcome="deferred", attempts=2)

    rig.daemon.run_catch_up()

    rows = rig.store.connection.execute(
        "SELECT id, outcome, attempts FROM checkins ORDER BY id"
    ).fetchall()
    assert [r["id"] for r in rows] == [existing], (
        f"the skip opened a second row instead of settling the open one: "
        f"{[dict(r) for r in rows]}"
    )
    assert rows[0]["outcome"] == "skipped"
    assert rows[0]["attempts"] == 2, "the attempts already made were discarded"


def test_recording_a_skip_never_breaks_catch_up(tmp_path, monkeypatch):
    """Isolation, matching the fire branch: run_catch_up must still consume
    the occurrence and return, or start() dies before the daemon has ticked
    once -- and under KeepAlive that is a respawn loop."""
    rig = _skipping_daemon(
        tmp_path, LAGOS, "evening", "0 21 * * *",
        missed_at=datetime(2026, 8, 5, 20, 0, tzinfo=timezone.utc),
        restart_at=datetime(2026, 8, 5, 22, 0, tzinfo=timezone.utc),
    )

    def boom(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(rig.store, "open_checkin", boom)

    assert rig.daemon.run_catch_up() == [("checkin_evening", "skip")]
    consumed = {job.name: job.last_run_at for job in rig.store.jobs()}
    assert consumed["checkin_evening"] is not None, (
        "the occurrence was not consumed, so the next tick re-decides it"
    )


@pytest.fixture
def daemon(tmp_path):
    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    scheduler = Scheduler(store, clock, LAGOS)
    fired: list[str] = []
    scheduler.register("checkin_morning", "0 11 * * *", lambda when: fired.append("m"))
    daemon = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={}, clock=clock,
    )
    return daemon, store, clock, fired


def test_tick_writes_a_heartbeat(daemon):
    instance, store, _, _ = daemon
    assert store.heartbeat() is None
    instance.tick()
    assert store.heartbeat() == NOW


def test_tick_advances_the_heartbeat(daemon):
    instance, store, clock, _ = daemon
    instance.tick()
    clock.advance(timedelta(minutes=5))
    instance.tick()
    assert store.heartbeat() == NOW + timedelta(minutes=5)


def test_degraded_flag_defaults_to_false(daemon):
    instance, _, _, _ = daemon
    assert instance.degraded is False


def test_run_catch_up_is_empty_without_a_heartbeat(daemon):
    instance, _, _, _ = daemon
    assert instance.run_catch_up() == []


def test_stop_stops_the_mic_not_just_the_activator(tmp_path):
    """Finding #1: WakeWordActivator.events() only checks `self._running`
    AFTER `mic.frames()` yields a frame:

        for frame in self._mic.frames():
            if not self._running:
                return

    With no incoming audio, `mic.frames()` never yields, so that check is
    never reached — activator.stop() alone cannot unblock a thread parked
    inside it. What actually unblocks it is MicStream.frames() observing
    its own `_stopping` Event, which only `MicStream.stop()` sets. So
    Daemon.stop() must call mic.stop() directly, not rely on the activator
    to propagate the shutdown.

    This is deliberately the narrower assertion, not an end-to-end proof
    that a hung WakeWordActivator.events() call actually returns: wiring a
    real WakeWordActivator needs a wake-word model that
    openwakeword.utils.download_models() fetches over the network, which
    tests must never require. Recording doubles for both activator and mic
    keep this test to what Daemon.stop() itself is responsible for: calling
    stop() on both, not just the activator. MicStream's own shutdown
    behaviour (frames() ending once _stopping is set) is covered directly
    by tests/audio/test_mic.py.
    """
    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    scheduler = Scheduler(store, clock, LAGOS)
    activator = FakeActivator(count=0)
    mic = _RecordingMic()
    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={}, clock=clock, activator=activator, mic=mic,
    )

    instance.stop()

    assert mic.stopped is True
    assert activator.stopped is True


# ---- DegradedPresence: C2 (review round 2) -------------------------------
#
# `degraded` used to be wired to nothing CheckIn could see: it gated only
# the wake-word thread, while CheckIn picks SPEAK vs NOTIFY purely from
# presence.verdict(), which knows nothing about microphone health. With a
# dead mic, a check-in would still speak into the void, listen(), hear
# nothing, and record NO_ANSWER -- precisely what audio_self_test (risk R1)
# exists to prevent.


def test_degraded_presence_translates_speak_to_notify():
    assert DegradedPresence(_StubPresence(Verdict.SPEAK)).verdict() is Verdict.NOTIFY


def test_degraded_presence_passes_defer_through():
    assert DegradedPresence(_StubPresence(Verdict.DEFER)).verdict() is Verdict.DEFER


def test_degrading_reaches_check_ins_built_before_the_self_test(tmp_path):
    """The property the old _degrade_presence() attribute-reach-in was
    standing in for, pinned directly rather than inferred from
    Daemon.start() + shared-object wiring (that end-to-end proof is still
    covered separately below).

    A CheckIn is built with a SwitchablePresence BEFORE any self-test has
    run -- exactly the order build_daemon() uses, since CheckIns are built
    well before Daemon.start() ever calls audio_self_test(). degrade() is
    then called on that same object, playing the role of "the daemon's
    copy" -- it IS the daemon's copy, since build_daemon() hands the one
    SwitchablePresence instance to both the Daemon and every CheckIn. The
    CheckIn's own verdict must flip from SPEAK to NOTIFY as a result, with
    no rebuilding and no reaching into CheckIn's attributes from outside.
    """
    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    presence = SwitchablePresence(_StubPresence(Verdict.SPEAK))
    checkin = CheckIn(
        kind="morning", store=store, journal=journal,
        presence=presence, voice=_StubVoice(), notifier=FakeNotifier(),
        conversation_factory=lambda conv_id, local: FakeConversation({}),
        config=ScheduleConfig(), tz=LAGOS, clock=clock,
    )
    assert checkin._presence.verdict() is Verdict.SPEAK

    presence.degrade()

    assert checkin._presence.verdict() is Verdict.NOTIFY


def test_a_failed_self_test_makes_check_ins_notify_instead_of_speak(tmp_path):
    """Behavioural, not just the flag: test_degraded_flag_defaults_to_false
    already covers `degraded` itself and proves nothing about whether
    CheckIn actually respects it.

    Wires a REAL CheckIn (not a stub) sharing the SAME SwitchablePresence
    object with the Daemon, exactly as build_daemon() does -- both are
    handed the one `presence` built from config.context -- so this proves
    the wiring reaches the CheckIn end-to-end through Daemon.start(), not
    merely that DegradedPresence's own logic is correct in isolation
    (pinned separately above) or that a bare degrade() call propagates
    (pinned directly in test_degrading_reaches_check_ins_built_before_the_
    self_test above). presence returns SPEAK; the mic double fails
    audio_self_test immediately (no frames, ever). After start(), the
    check-in must notify rather than speak.
    """
    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    scheduler = Scheduler(store, clock, LAGOS)
    presence = SwitchablePresence(_StubPresence(Verdict.SPEAK))
    voice = _StubVoice()
    notifier = FakeNotifier()
    conversation = FakeConversation({})
    checkin = CheckIn(
        kind="morning", store=store, journal=journal,
        presence=presence, voice=voice, notifier=notifier,
        conversation_factory=lambda conv_id, local: conversation,
        config=ScheduleConfig(), tz=LAGOS, clock=clock,
    )
    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=presence, voice=voice,
        notifier=notifier, checkins={"checkin_morning": checkin},
        clock=clock, mic=_FailingMic(),
    )

    instance.start()
    assert instance.degraded is True
    # start() itself already sent one notification ("mic unavailable") --
    # notifier is the SAME object handed to both Daemon and CheckIn here,
    # matching build_daemon()'s real wiring. What this test is actually
    # about is the notification checkin.run() sends on TOP of that.
    sent_by_daemon_startup = len(notifier.sent)

    outcome = checkin.run(NOW)

    assert len(notifier.sent) == sent_by_daemon_startup + 1
    assert notifier.sent[-1] == ("ZEUS", "Morning check-in")
    assert voice.spoken == []
    assert outcome is Outcome.DEFERRED


def test_each_catch_up_decision_consumes_its_own_occurrence(tmp_path):
    """M1 (round 4): catch_up_actions returns one entry per missed RUN, so a
    job with several missed occurrences produces several entries. Resolving
    each entry's timestamp through a {job: latest_occurrence} dict gave
    every one of them that job's LATEST occurrence -- so the skip of
    yesterday's morning logged and consumed the timestamp of the occurrence
    about to be FIRED two iterations later.

    Two days of downtime, one daily job: catch_up() returns yesterday's
    morning (a different local day, so "skip") and today's ("fire"). Each
    decision must be recorded against the occurrence it is actually about.
    The durable end state is the same either way -- `missed` is ascending
    and the last write wins -- so the assertion is on the sequence, which
    is the only place the defect is visible.
    """
    clock = FakeClock(datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc))
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    scheduler = Scheduler(store, clock, LAGOS)
    fired: list[datetime] = []

    class _RecordingCheckIn:
        def run(self, scheduled_for):
            fired.append(scheduled_for)

    checkin = _RecordingCheckIn()
    scheduler.register("checkin_morning", "0 11 * * *", checkin.run)
    store.set_heartbeat()

    consumed: list[tuple[str, datetime]] = []
    real_set_job_run = store.set_job_run

    def recording_set_job_run(name, last_run_at):
        consumed.append((name, last_run_at))
        real_set_job_run(name, last_run_at)

    store.set_job_run = recording_set_job_run

    # Two days down: 11:00 Lagos == 10:00 UTC on the 4th and the 5th.
    clock.advance(timedelta(days=2))  # 2026-08-05 12:00 UTC
    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={"checkin_morning": checkin}, clock=clock,
    )

    yesterday = datetime(2026, 8, 4, 10, 0, tzinfo=timezone.utc)
    today = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)

    actions = instance.run_catch_up()

    assert actions == [("checkin_morning", "skip"), ("checkin_morning", "fire")]
    assert fired == [today]
    assert consumed == [("checkin_morning", yesterday), ("checkin_morning", today)], (
        "a catch-up decision was recorded against an occurrence other than "
        "the one it was made about"
    )


# ---- the §9.3 retry ladder, end to end (Task 19) -------------------------
#
# THE REGRESSION TESTS FOR THE WHOLE FINDING.
#
# next_step() has always computed Decision(outcome, retry_after,
# fold_forward). CheckIn.run read .outcome and dropped the rest; Scheduler
# had only a cron register(); nothing anywhere scheduled a run at
# `now + retry_after`. So a user away from the desk at 11:00 was asked once
# and ZEUS gave up for the day.
#
# It hid because the two tests that LOOK like retry tests
# (test_attempts_accumulate_across_repeated_runs,
# test_silence_all_day_never_loses_the_checkin_row) call checkin.run(...)
# twice at the SAME instant with no scheduler and no clock advance. They pin
# the state machine and the row reuse -- never the scheduling. Every layer
# was tested; the wire between them was not. So these drive the REAL
# Scheduler and a REAL CheckIn across a simulated morning with a moving
# clock. Calling run() by hand proves nothing about this bug.


class _TimestampingCheckIn:
    """Delegates to a real CheckIn, recording the WALL-CLOCK time of each run.

    The `scheduled_for` a retry re-runs with is deliberately the original
    11:00 occurrence, so it cannot distinguish the rungs of the ladder --
    what the ladder is about is when each run actually HAPPENS.
    """

    def __init__(self, inner, clock, fired):
        self._inner = inner
        self._clock = clock
        self._fired = fired

    def run(self, scheduled_for):
        self._fired.append(self._clock.now_utc())
        return self._inner.run(scheduled_for)


def _away(tmp_path, tz, kind, cron, start):
    """Real Store, Scheduler and CheckIn; the user is away from the desk.

    Presence is pinned to DEFER for the whole run, which is exactly the
    situation §9.3's ladder was designed for. `start` is an hour before the
    scheduled occurrence, so the scheduler's first run_pending() seeds its
    baseline without firing -- as it does on a real fresh start.

    tz REACHES THE Daemon, not just the Store/Journal/Scheduler/CheckIn.
    Omitted, Daemon.__init__ falls back to resolve_timezone("system"), which
    follows /etc/localtime -- so the rig claimed to be running in Los
    Angeles while the daemon inside it ran in whatever zone the developer's
    Mac is set to, and any daemon-level rule that reads self._tz was being
    exercised against the wrong zone (and differently on every machine).
    """
    clock = FakeClock(start)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, tz)
    scheduler = Scheduler(store, clock, tz)
    config = ScheduleConfig()
    fired: list[datetime] = []

    checkin = _TimestampingCheckIn(
        CheckIn(
            kind=kind, store=store, journal=journal,
            presence=_StubPresence(Verdict.DEFER), voice=_StubVoice(),
            notifier=FakeNotifier(),
            conversation_factory=lambda conv_id, local: FakeConversation({}),
            config=config, tz=tz, clock=clock,
        ),
        clock, fired,
    )
    name = f"checkin_{kind}"
    scheduler.register(name, cron, checkin.run)

    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={name: checkin}, clock=clock, tz=tz,
    )
    return SimpleNamespace(
        daemon=instance, store=store, clock=clock, fired=fired,
        config=config, tz=tz,
    )


def _away_all_morning(tmp_path, tz):
    return _away(
        tmp_path, tz, "morning", "0 11 * * *",
        datetime(2026, 8, 5, 10, 0, tzinfo=tz),
    )


def _local_times(moments, tz):
    return [m.astimezone(tz).strftime("%H:%M") for m in moments]


def test_the_daemon_runs_a_retry_when_it_comes_due(tmp_path):
    """The ladder actually fires, on the clock, without anyone calling run().

    The expected rungs are 11:00 (the cron occurrence) plus
    `max_defer_retries` retries 20 minutes apart. max_defer_retries counts
    RETRIES, not total attempts -- see next_step()'s
    `attempts + 1 > config.max_defer_retries` and test_retry.py's
    test_defer_retries_up_to_three_times, which pins a retry at attempts=0,
    1 AND 2 -- so with the shipped defaults (20m x 3) that is FOUR runs:
    11:00, 11:20, 11:40, 12:00.

    Before the fix this list was ["11:00"] and attempts was 1.
    """
    rig = _away_all_morning(tmp_path, tz=LOS_ANGELES)

    for _ in range(13 * 12):                    # 10:00 -> 23:00 local, every 5 min
        rig.daemon.tick()
        rig.clock.advance(timedelta(minutes=5))

    expected = ["11:00", "11:20", "11:40", "12:00"]
    assert _local_times(rig.fired, LOS_ANGELES) == expected, (
        "the §9.3 retry ladder did not fire: a user away from the desk at "
        "11:00 is asked once and ZEUS gives up for the day"
    )
    assert rig.store.get_checkin(1).attempts == rig.config.max_defer_retries + 1


def test_the_ladder_stops_once_the_retries_are_exhausted(tmp_path):
    """The other half: a ladder that never terminates is as broken as one
    that never starts. Once exhausted the morning check-in folds forward
    into the evening (§9.3) -- it must not keep asking every 20 minutes
    until midnight, and it must never poll on every tick.
    """
    rig = _away_all_morning(tmp_path, tz=LOS_ANGELES)

    for _ in range(13 * 12):
        rig.daemon.tick()
        rig.clock.advance(timedelta(minutes=5))

    row = rig.store.get_checkin(1)
    assert row.retry_at is None
    assert row.outcome == "deferred"            # still open, so the evening folds
    assert rig.store.due_retries(rig.clock.now_utc()) == []
    assert len(rig.fired) == rig.config.max_defer_retries + 1


def test_every_rung_of_the_ladder_reuses_the_original_checkin(tmp_path):
    """A retry belongs to the occurrence the ritual started with.

    Re-running with `now` instead of the original `scheduled_for` would
    compute a fresh local_date at a day boundary and open a second row, so
    attempts would restart at 1 and the ladder would never exhaust. Los
    Angeles is used throughout because its UTC offset is negative -- the
    seam where local and UTC dates diverge, which Africa/Lagos can never
    expose.
    """
    rig = _away_all_morning(tmp_path, tz=LOS_ANGELES)

    for _ in range(13 * 12):
        rig.daemon.tick()
        rig.clock.advance(timedelta(minutes=5))

    rows = rig.store.connection.execute(
        "SELECT id, local_date, scheduled_for FROM checkins ORDER BY id"
    ).fetchall()
    assert [r["id"] for r in rows] == [1], (
        f"the retries opened {len(rows)} check-in rows instead of reusing one"
    )
    assert rows[0]["local_date"] == "2026-08-05"
    assert from_utc_iso(rows[0]["scheduled_for"]).astimezone(
        LOS_ANGELES
    ).strftime("%H:%M") == "11:00"


def test_a_retry_across_midnight_still_belongs_to_the_original_day(tmp_path):
    """A retry must re-run with the ORIGINAL scheduled_for, never with `now`.

    This is the one arrangement where the two differ, and it is why
    test_every_rung_of_the_ladder_reuses_the_original_checkin cannot stand
    in for it: at 11:00 the occurrence and `now` fall on the same local
    date, so passing either reuses the same row and the defect is invisible.

    A 23:50 check-in deferred at 23:50 retries at 00:10 -- the next local
    day. Re-running with `now` computes local_date "2026-08-06", finds no
    open check-in for it, and opens a SECOND row whose attempts restart at
    1. Worse, the first row's retry_at is then never rewritten, so it stays
    due on every subsequent tick and the ladder becomes an unbounded loop.
    """
    rig = _away(
        tmp_path, LOS_ANGELES, "evening", "50 23 * * *",
        datetime(2026, 8, 5, 23, 0, tzinfo=LOS_ANGELES),
    )

    for _ in range(2 * 12):                      # 23:00 -> 01:00 local
        rig.daemon.tick()
        rig.clock.advance(timedelta(minutes=5))

    assert _local_times(rig.fired, LOS_ANGELES) == [
        "23:50", "00:10", "00:30", "00:50",
    ]
    rows = rig.store.connection.execute(
        "SELECT id, local_date, attempts FROM checkins ORDER BY id"
    ).fetchall()
    assert [r["id"] for r in rows] == [1], (
        f"the retry opened a second check-in row across the day boundary: "
        f"{[dict(r) for r in rows]}"
    )
    assert rows[0]["local_date"] == "2026-08-05"
    assert rows[0]["attempts"] == rig.config.max_defer_retries + 1


def test_a_retry_survives_a_daemon_restart(tmp_path):
    """The retry is durable, not in-memory: a daemon that dies at 11:05 and
    comes back at 11:25 must still honour the 11:20 rung. An in-process
    timer would lose it, and the LaunchAgent's KeepAlive makes restarts a
    routine event, not an edge case.
    """
    rig = _away_all_morning(tmp_path, tz=LOS_ANGELES)

    for _ in range(13):                          # 10:00 -> 11:00, fires once
        rig.daemon.tick()
        rig.clock.advance(timedelta(minutes=5))
    assert _local_times(rig.fired, LOS_ANGELES) == ["11:00"]
    pending = rig.store.get_checkin(1).retry_at
    rig.store.close()

    # A fresh process: new Store, new Scheduler, new CheckIn, same database.
    clock = FakeClock(datetime(2026, 8, 5, 11, 25, tzinfo=LOS_ANGELES))
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LOS_ANGELES)
    scheduler = Scheduler(store, clock, LOS_ANGELES)
    fired: list[datetime] = []
    checkin = _TimestampingCheckIn(
        CheckIn(
            kind="morning", store=store, journal=journal,
            presence=_StubPresence(Verdict.DEFER), voice=_StubVoice(),
            notifier=FakeNotifier(),
            conversation_factory=lambda conv_id, local: FakeConversation({}),
            config=ScheduleConfig(), tz=LOS_ANGELES, clock=clock,
        ),
        clock, fired,
    )
    scheduler.register("checkin_morning", "0 11 * * *", checkin.run)
    restarted = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={"checkin_morning": checkin}, clock=clock,
    )

    assert pending.astimezone(LOS_ANGELES).strftime("%H:%M") == "11:20"
    restarted.tick()

    assert _local_times(fired, LOS_ANGELES) == ["11:25"], (
        "the 11:20 retry was lost across the restart"
    )
    assert store.get_checkin(1).attempts == 2
    store.close()


def test_a_retry_that_raises_does_not_stop_the_tick(tmp_path):
    """One failing retry must not abort the sweep or crash run_forever --
    the same isolation Scheduler.run_pending already gives the cron path."""
    clock = FakeClock(datetime(2026, 8, 5, 11, 30, tzinfo=LOS_ANGELES))
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LOS_ANGELES)
    scheduler = Scheduler(store, clock, LOS_ANGELES)

    morning = store.open_checkin(
        "morning", clock.now_utc() - timedelta(minutes=30), "2026-08-05"
    )
    evening = store.open_checkin(
        "evening", clock.now_utc() - timedelta(minutes=30), "2026-08-05"
    )
    store.update_checkin(morning, outcome="deferred", attempts=1,
                         retry_at=clock.now_utc() - timedelta(minutes=10))
    store.update_checkin(evening, outcome="deferred", attempts=1,
                         retry_at=clock.now_utc() - timedelta(minutes=5))

    ran: list[str] = []

    class _Boom:
        def run(self, scheduled_for):
            ran.append("morning")
            raise RuntimeError("the API is unreachable")

    class _Fine:
        def run(self, scheduled_for):
            ran.append("evening")

    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={"checkin_morning": _Boom(), "checkin_evening": _Fine()},
        clock=clock,
    )

    instance.tick()                              # must not raise

    assert ran == ["morning", "evening"]
    assert store.heartbeat() == clock.now_utc()  # the tick still completed


def test_build_daemon_wraps_presence_so_a_failed_self_test_can_degrade_it(
    monkeypatch, tmp_path
):
    """M3 (round 4): both other wiring tests construct the SwitchablePresence
    themselves, so nothing pinned that build_daemon() wraps Presence at all
    -- and Daemon.start() calls self._presence.degrade() unconditionally on
    a failed self-test, which a bare Presence does not have.

    Runs the real factory. No microphone, no speaker and no network are
    touched: MicStream/WakeWordActivator/LocalWhisper/MacSay all defer their
    device and model work to first use, and anthropic.Anthropic() only reads
    the key. The key is set here rather than read from the environment so
    the test neither depends on nor consumes a real one.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-sent")

    instance = build_daemon(Config(root=tmp_path))

    assert isinstance(instance._presence, SwitchablePresence)
    # The same object reaches every CheckIn -- that shared identity is what
    # makes one degrade() call downgrade all of them.
    for name in ("checkin_morning", "checkin_evening"):
        assert instance._checkins[name]._presence is instance._presence
    # And one mic, fanned out to both consumers (round 4, C2): the detector
    # and VoiceIO must share the single CoreAudio stream, not open two.
    assert instance._voice._mic is instance._mic
    assert instance._activator._mic is instance._mic
    instance._store.close()


# ---- X1: a stale retry must never replay a check-in whose moment has passed
#
# THE REGRESSION TESTS FOR THE SECOND-ORDER DEFECT TASK 19 INTRODUCED.
#
# due_retries() filters only on `retry_at IS NOT NULL AND retry_at <= now`.
# Nothing bounded how LATE a retry could be, so a Mac that slept through the
# 11:20 rung fired it the next morning -- re-running with the ORIGINAL
# scheduled_for, which is correct for a 20-minute retry and catastrophic for
# a 22-hour one: `date` is yesterday, so today's answer is written onto
# yesterday's goal row, destroying a completed review, and today records
# nothing. Spec §9.2: "never replay a check-in whose moment has genuinely
# passed"; §9.3's ladder tops out 60 minutes after the occurrence.
#
# The bound is on the retry's AGE, not on its local calendar date. A local-day
# rule would drop the perfectly legitimate rungs of a late-evening ladder --
# a 23:50 check-in deferring to 00:10 is twenty minutes late, not a day --
# which test_a_retry_across_midnight_still_belongs_to_the_original_day pins,
# and it would make the decision depend on the daemon's timezone, the seam
# that has produced six defects in this codebase already.


def _goal_saving_checkin(store, journal, clock, tz, voice, kind="morning"):
    """A REAL CheckIn whose fake brain drives the REAL save_goal tool.

    The point of the test below is data loss, so the write has to be the
    production one: build_tool_callables closes over the local date the
    CheckIn was constructed with, which is exactly what a stale retry gets
    wrong.
    """

    def factory(conversation_id, date):
        return FakeConversation(
            tools=build_tool_callables(store, journal, conversation_id, date),
            tool_calls=[[("save_goal", {"text": "call the dentist"})]],
        )

    return CheckIn(
        kind=kind, store=store, journal=journal,
        presence=_StubPresence(Verdict.SPEAK), voice=voice,
        notifier=FakeNotifier(), conversation_factory=factory,
        config=ScheduleConfig(), tz=tz, clock=clock,
    )


def test_a_stale_retry_never_writes_todays_answer_onto_yesterdays_row(tmp_path):
    """The data-loss trigger, reproduced end to end.

    Morning check-in DEFERs at 11:00 on 2026-08-05 and schedules its 11:20
    rung. The lid closes; the machine is asleep until 09:00 the next day.
    Before the fix, that retry fired with scheduled_for=2026-08-05 11:00, so
    save_goal's upsert rewrote the 2026-08-05 goal row -- resetting the
    status and NULLing the notes of a review that had already been completed
    -- while 2026-08-06 recorded nothing at all.
    """
    tz = LOS_ANGELES
    clock = FakeClock(datetime(2026, 8, 5, 11, 0, tzinfo=tz))
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, tz)
    scheduler = Scheduler(store, clock, tz)
    voice = _StubVoice()

    goal_id = store.set_goal("2026-08-05", "water the plants")
    store.update_goal(goal_id, "done", "watered them before lunch")
    checkin_id = store.open_checkin("morning", clock.now_utc(), "2026-08-05")
    store.update_checkin(
        checkin_id, outcome="deferred", attempts=1,
        retry_at=clock.now_utc() + timedelta(minutes=20),
    )

    clock.advance(timedelta(hours=22))           # 09:00 the NEXT local day
    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={
            "checkin_morning": _goal_saving_checkin(
                store, journal, clock, tz, voice
            )
        },
        clock=clock, tz=tz,
    )

    instance.tick()

    survivor = store.get_goal("2026-08-05")
    assert survivor.text == "water the plants", (
        "a stale retry overwrote yesterday's goal with today's answer"
    )
    assert survivor.status == "done", "the completed review was reset to pending"
    assert survivor.notes == "watered them before lunch", "the notes were erased"
    assert store.get_goal("2026-08-06") is None, (
        "today's answer was filed under yesterday's date, so today has no goal"
    )
    assert voice.spoken == [], "the stale check-in was replayed out loud"


def test_a_stale_retry_is_settled_so_it_cannot_come_back(tmp_path):
    """§9.2's "record outcome=skipped", applied to the retry path.

    Dropping the retry is only half of it: leaving retry_at set would make
    the same stale row come due again on the very next tick, once a minute,
    for the rest of the process's life.
    """
    tz = LOS_ANGELES
    clock = FakeClock(datetime(2026, 8, 5, 21, 0, tzinfo=tz))
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, tz)
    scheduler = Scheduler(store, clock, tz)

    checkin_id = store.open_checkin("evening", clock.now_utc(), "2026-08-05")
    store.update_checkin(
        checkin_id, outcome="deferred", attempts=2,
        retry_at=clock.now_utc() + timedelta(minutes=20),
    )

    ran: list[datetime] = []

    class _RecordingCheckIn:
        def run(self, scheduled_for):
            ran.append(scheduled_for)

    clock.advance(timedelta(hours=11, minutes=5))   # 08:05 the next morning
    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={"checkin_evening": _RecordingCheckIn()}, clock=clock, tz=tz,
    )

    instance.tick()

    assert ran == [], (
        "the evening check-in was replayed the next morning — spec §9.2: "
        "'do not ask about yesterday today'"
    )
    row = store.get_checkin(checkin_id)
    assert row.outcome == "skipped"
    assert row.retry_at is None
    assert store.due_retries(clock.now_utc()) == [], (
        "the stale retry is still due and will fire again on the next tick"
    )
    # attempts is untouched: a retry that never fired is not an attempt.
    assert row.attempts == 2
    instance.tick()                                  # and stays settled
    assert ran == []


def test_a_retry_that_is_merely_late_still_fires(tmp_path):
    """Guards the guard: the staleness bound must not eat live rungs.

    A daemon restarted twenty-five minutes after the occurrence must still
    honour the 11:20 rung (test_a_retry_survives_a_daemon_restart pins that
    end to end); this pins the bound itself, at the granularity a mutation
    to the constant would show up in.
    """
    tz = LOS_ANGELES
    clock = FakeClock(datetime(2026, 8, 5, 11, 0, tzinfo=tz))
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, tz)
    scheduler = Scheduler(store, clock, tz)

    checkin_id = store.open_checkin("morning", clock.now_utc(), "2026-08-05")
    store.update_checkin(
        checkin_id, outcome="deferred", attempts=1,
        retry_at=clock.now_utc() + timedelta(minutes=20),
    )

    ran: list[datetime] = []

    class _RecordingCheckIn:
        def run(self, scheduled_for):
            ran.append(scheduled_for)

    clock.advance(timedelta(minutes=25))
    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={"checkin_morning": _RecordingCheckIn()}, clock=clock, tz=tz,
    )

    instance.tick()

    assert ran == [datetime(2026, 8, 5, 11, 0, tzinfo=tz)]


# ---- X2: a gap in the audio must not end "hey jarvis" for the life of the
#          process
#
# THE REGRESSION TEST FOR THE SECOND-ORDER DEFECT THE A1 IDLE BOUND
# INTRODUCED.
#
# Subscription.frames() used to return only on stop(); A1 made it also
# return after _IDLE_TIMEOUT_SECONDS of silence. The wake detector is the
# only long-lived consumer, and its chain is unconditional: frames() returns
# -> _detect returns -> events() closes its subscription and ends ->
# _activation_loop's single `for` falls out -> the thread returns, and
# nothing restarted it. Sleep/wake, AirPods taking over the default input, a
# USB mic unplugged, a coreaudiod restart -- any of them over five seconds
# permanently ended ad-hoc activation, with _running still True, nothing
# logged, and `doctor` unable to see it. Spec §10 inverted.
#
# Real MicStream, real WakeWordActivator, real Daemon thread. Only the
# openWakeWord model is a double, because loading the real one downloads
# over the network.


class _ScriptedModel:
    """Stands in for openwakeword.Model; `score` is flipped by the test."""

    def __init__(self) -> None:
        self.score = 0.0
        self.seen = 0

    def predict(self, samples):
        self.seen += 1
        return {"hey_jarvis": self.score}


class _SignallingVoice:
    """A voice that records what was spoken and signals that it was used."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.activated = threading.Event()

    def speak(self, sentences) -> None:
        self.spoken.extend(sentences)
        self.activated.set()

    def listen(self) -> str:
        return ""


def test_wake_word_activation_survives_a_gap_in_the_audio(
    tmp_path, monkeypatch, caplog
):
    """Frames, a gap past the idle bound, then frames again — still activates.

    The gap is a real one: the idle bound is shortened to 0.2s so the test
    costs a fraction of a second rather than five, but the mechanism is
    untouched — Subscription.frames() genuinely gives up, _detect genuinely
    returns, events() genuinely closes its subscription. Before the fix the
    activation thread was gone by then and the 0.9 scores below reached
    nobody.
    """
    import logging

    from zeus.audio.wakeword import WakeWordActivator
    from zeus.config import WakeConfig

    monkeypatch.setattr("zeus.audio.mic._IDLE_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr("zeus.daemon._ACTIVATION_RESTART_SECONDS", 0.01)

    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    scheduler = Scheduler(store, clock, LAGOS)
    mic = MicStream(AudioConfig())
    model = _ScriptedModel()
    activator = WakeWordActivator(mic, WakeConfig(), threshold=0.5)
    monkeypatch.setattr(activator, "_load_model", lambda: model)
    voice = _SignallingVoice()

    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=voice, notifier=None,
        checkins={}, clock=clock, activator=activator, mic=None, tz=LAGOS,
    )

    try:
        with caplog.at_level(logging.WARNING, logger="zeus.daemon"):
            instance.start()
            _await(lambda: activator._subscription is not None,
                   "the detector never subscribed")

            for _ in range(5):                       # audio, below threshold
                mic._on_audio(SILENCE, FRAME_SAMPLES, None, None)
            _await(lambda: model.seen >= 5, "the first frames were not scored")

            time.sleep(0.35)                         # the gap: past the bound
            _await(lambda: activator._subscription is not None,
                   "activation was never restarted after the audio gap")

            # The device comes back and the model scores the wake word.
            model.score = 0.9
            deadline = time.monotonic() + 5.0
            while not voice.activated.is_set() and time.monotonic() < deadline:
                mic._on_audio(SPEECH, FRAME_SAMPLES, None, None)
                time.sleep(0.02)

        assert voice.activated.is_set(), (
            "a >5s gap in the audio ended wake-word activation for the life "
            "of the process: 'hey jarvis' never works again"
        )
        assert voice.spoken == [NOT_CAUGHT_LINE]
        assert any(
            "restarting" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        ), (
            "activation restarted silently — §10: fail loudly, never pretend"
        )
    finally:
        instance.stop()
        mic.stop()


def test_the_activation_loop_ends_when_the_daemon_stops(tmp_path, monkeypatch):
    """The other half: a restart loop that cannot be stopped is a spin.

    stop() must end the thread rather than have it re-enter events() forever
    — and it must do so without waiting out the restart backoff, since
    launchd escalates SIGTERM to SIGKILL.
    """
    from zeus.audio.wakeword import WakeWordActivator
    from zeus.config import WakeConfig

    monkeypatch.setattr("zeus.audio.mic._IDLE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr("zeus.daemon._ACTIVATION_RESTART_SECONDS", 30.0)

    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    scheduler = Scheduler(store, clock, LAGOS)
    mic = MicStream(AudioConfig())
    activator = WakeWordActivator(mic, WakeConfig(), threshold=0.5)
    monkeypatch.setattr(activator, "_load_model", lambda: _ScriptedModel())

    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={}, clock=clock, activator=activator, mic=None, tz=LAGOS,
    )
    instance.start()
    _await(lambda: activator._subscription is not None,
           "the detector never subscribed")
    time.sleep(0.15)                         # let it idle out and back off

    instance.stop()
    mic.stop()

    # The thread itself, not threading.active_count(): the process-wide
    # count is polluted by every other thread the suite has running, and it
    # let an uninterruptible time.sleep() backoff survive a mutation run.
    instance._activation_thread.join(timeout=5.0)
    assert not instance._activation_thread.is_alive(), (
        "the activation thread outlived stop() — the 30s restart backoff was "
        "waited out instead of interrupted, and launchd escalates to SIGKILL"
    )


# ---- X3: degraded mode must notify once a check-in, not four times -------


def test_a_notified_checkin_is_not_notified_again_every_twenty_minutes(tmp_path):
    """The user-visible half of X3, driven through the real ladder.

    A failed mic self-test turns every SPEAK into NOTIFY (DegradedPresence),
    and nothing anywhere can mark a macOS notification "answered" — so while
    NOTIFY fell through to the DEFER branch it always ran to exhaustion:
    four notifications per check-in, eight a day, for as long as the
    microphone stayed broken. §9.3's table gives NOTIFY no retry at all.

    The scheduler, the CheckIn and the ladder are all real; only presence is
    pinned, which is exactly what DegradedPresence does.
    """
    tz = LOS_ANGELES
    clock = FakeClock(datetime(2026, 8, 5, 10, 0, tzinfo=tz))
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, tz)
    scheduler = Scheduler(store, clock, tz)
    notifier = FakeNotifier()

    checkin = CheckIn(
        kind="morning", store=store, journal=journal,
        presence=DegradedPresence(_StubPresence(Verdict.SPEAK)),
        voice=_StubVoice(), notifier=notifier,
        conversation_factory=lambda conv_id, local: FakeConversation({}),
        config=ScheduleConfig(), tz=tz, clock=clock,
    )
    scheduler.register("checkin_morning", "0 11 * * *", checkin.run)
    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={"checkin_morning": checkin}, clock=clock, tz=tz,
    )

    for _ in range(13 * 12):                     # 10:00 -> 23:00 local
        instance.tick()
        clock.advance(timedelta(minutes=5))

    assert len(notifier.sent) == 1, (
        f"a degraded-mode check-in sent {len(notifier.sent)} notifications; "
        f"§9.3 gives NOTIFY no retry, so it must send exactly one"
    )
    row = store.get_checkin(1)
    assert row.retry_at is None
    assert row.outcome == "deferred"             # unanswered, not abandoned


def test_a_raising_activation_source_is_restarted_too(tmp_path, monkeypatch, caplog):
    """The other way events() can end: by raising.

    openWakeWord failing to load its model raises straight out of events(),
    which killed the thread just as silently as the idle bound did. It must
    restart on the same loop — and say what actually happened, not blame the
    microphone.
    """
    import logging

    monkeypatch.setattr("zeus.daemon._ACTIVATION_RESTART_SECONDS", 0.01)

    class _BrokenActivator:
        def __init__(self):
            self.attempts = 0

        def start(self):
            pass

        def stop(self):
            pass

        def events(self):
            self.attempts += 1
            raise RuntimeError("openwakeword could not load its model")

    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    scheduler = Scheduler(store, clock, LAGOS)
    activator = _BrokenActivator()
    instance = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={}, clock=clock, activator=activator, mic=None, tz=LAGOS,
    )

    try:
        with caplog.at_level(logging.WARNING, logger="zeus.daemon"):
            instance.start()
            _await(lambda: activator.attempts >= 3,
                   "a raising activation source was not restarted")
    finally:
        instance.stop()

    instance._activation_thread.join(timeout=5.0)
    assert not instance._activation_thread.is_alive()
    assert any(
        "the activation source raised" in record.getMessage()
        for record in caplog.records
    ), "the restart log blamed the microphone for a model-load failure"
