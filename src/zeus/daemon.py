"""The zeusd daemon: wiring, supervision, and the main loop. See spec §4, §9.2."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Any

from zeus.audio.endpointer import rms
from zeus.audio.mic import FRAME_SAMPLES, MicStream
from zeus.brain.prompts import NOT_CAUGHT_LINE
from zeus.clock import Clock, SystemClock, resolve_timezone
from zeus.config import Config, load_config
from zeus.context.presence import Verdict
# local_date at module scope, unlike the rest of zeus.ritual.checkin (which
# build_daemon imports lazily): _record_skipped needs it on the catch-up
# path, and there must be exactly ONE definition of "the local calendar
# date a check-in belongs to" — re-deriving it here is how the UTC/local
# split that broke every retry got in last time.
from zeus.ritual.checkin import local_date
from zeus.schedule.cron import hhmm_to_cron
from zeus.ui.overlay import LISTENING
from zeus.schedule.scheduler import MissedRun, Scheduler

log = logging.getLogger(__name__)

# Only the morning check-in is ever replayed, and only on the same local
# day. A goal question at 15:00 is useful; the same question at 09:00 the
# next morning is noise. See spec §9.2.
CATCH_UP_ELIGIBLE = {"checkin_morning"}

# HOW LATE A §9.3 RETRY MAY STILL FIRE. The ladder's own span is the
# yardstick -- §9.3 tops out at max_defer_retries x defer_retry_after (60
# minutes with the shipped defaults) -- plus slack, because each rung is
# scheduled from `now` rather than from the occurrence, so tick granularity
# and a restart both push the last rung a little further out. Beyond that
# the occurrence's moment has genuinely passed (§9.2) and replaying it is
# not a late check-in, it is next-day data loss.
_RETRY_SLACK = timedelta(minutes=30)
# A hard ceiling regardless of configuration. max_defer_retries is a config
# value, so a large one would otherwise licence a ladder that spans into the
# next local day -- and a morning retry that crosses midnight re-runs with
# yesterday's scheduled_for, which is precisely the write that destroys a
# completed goal row. §9.2 outranks a long ladder.
_RETRY_MAX_AGE = timedelta(hours=6)

# How long the activation loop waits before re-entering events() after the
# microphone stopped delivering. Subscription.frames()' own idle bound
# already rate-limits each cycle to _IDLE_TIMEOUT_SECONDS; this only stops a
# source that fails INSTANTLY (a raising activator, say) from spinning.
_ACTIVATION_RESTART_SECONDS = 5.0

# ...and how far that wait is allowed to grow. The wait DOUBLES per restart up
# to this ceiling, because a fault that does not clear is the common case, not
# the rare one: a USB microphone unplugged and left out restarts every
# _IDLE_TIMEOUT_SECONDS + _ACTIVATION_RESTART_SECONDS = 10s forever, and each
# restart writes a WARNING. That is 8,640 lines a day, ~0.79 GB a year, into a
# zeusd.log that nothing rotates. Backing off to five minutes cuts it ~30x
# while leaving the FIRST failures exactly as loud and as prompt as before —
# which is the half that matters, since a transient gap (sleep/wake, AirPods
# taking the input) recovers on the first or second try and never reaches the
# ceiling at all.
_ACTIVATION_RESTART_MAX_SECONDS = 300.0

# The longest the run loop may go without noticing request_stop(). launchd's
# default grace period between SIGTERM and SIGKILL is 20 seconds, and the
# tick loop can otherwise be a full minute from its next wake.
_SHUTDOWN_POLL_SECONDS = 1.0


def audio_self_test(mic: MicStream, seconds: float = 1.0) -> bool:
    """Capture briefly and assert the microphone is actually producing audio.

    Risk R1: when macOS denies microphone access to a LaunchAgent-spawned
    process, the stream opens successfully and returns pure silence forever.
    Without this check ZEUS looks healthy while being completely deaf.

    `seconds` bounds a FRAME COUNT (roughly `seconds` of audio at 16kHz), not
    wall-clock time. MicStream.frames() polls its own queue internally and
    only returns control once its `_stopping` Event is set -- nothing sets
    that during a self-test. So if the audio callback stops firing
    mid-capture -- a plausible dead-hardware variant of R1, distinct from
    "never produced audio at all" -- the `for` loop below would never see
    another frame and would block forever: a generator only hands control
    back to its caller at a `yield`, and with the callback dead there is no
    further one coming, so no code here would ever run again to notice.
    Consuming on a background thread is what makes a real deadline possible
    even though that thread itself may never return: Event.wait(timeout) is
    backed by a monotonic clock, so it can never be defeated by a backward
    wall-clock jump (an NTP correction, a DST transition) the way a
    datetime.now()-based elapsed-time check could be.

    Subscription.frames() now carries its own idle bound (A1), which
    changes this function in two ways. The thread no longer runs forever on
    a dead device -- it ends within _IDLE_TIMEOUT_SECONDS of the last frame
    and closes its subscription, so the leak this used to spring on timeout
    is bounded and self-clearing. But `done` stopped meaning "the wanted
    frames arrived": it now also fires when frames() gave up. Hence the
    explicit `seen < wanted` check below. Without it, a mic that delivered
    2 of 12 frames and died would set `done`, sail past the deadline, find
    seen != 0 and energy != 0, and be reported HEALTHY -- the exact R1
    failure this function exists to catch, reintroduced through the back
    door by its own fix.
    """
    wanted = max(1, int(seconds * 16000 / FRAME_SAMPLES))
    energy = 0.0
    seen = 0
    done = threading.Event()

    def consume() -> None:
        nonlocal energy, seen
        for frame in mic.frames():
            energy = max(energy, rms(frame))
            seen += 1
            if seen >= wanted:
                break
        done.set()

    threading.Thread(target=consume, daemon=True).start()
    # Budget generously at 5s: this measures TOTAL capture, not
    # time-to-first-frame, and a Bluetooth input switching into its HFP
    # profile can take seconds before the first callback arrives. Too tight
    # a budget makes a slow device indistinguishable from a dead one -- and
    # the consequence is sticky, because `degraded` is never re-tested or
    # cleared, so ZEUS stays notification-only until the process restarts.
    deadline = max(seconds * 5.0, 5.0)
    if not done.wait(timeout=deadline):
        log.error(
            "audio self-test: timed out after %.1fs waiting for %d frames "
            "(got %d so far) — the microphone may have stopped producing "
            "audio mid-capture", deadline, wanted, seen,
        )
        return False
    if seen == 0:
        log.error("audio self-test: no frames received from the microphone")
        return False
    if seen < wanted:
        log.error(
            "audio self-test: the microphone delivered %d of %d frames and "
            "then stopped — the input device is failing mid-capture",
            seen, wanted,
        )
        return False
    if energy <= 0.0:
        log.error(
            "audio self-test: %d frames captured but all silent — "
            "microphone permission is probably denied", seen,
        )
        return False
    return True


def catch_up_actions(missed: list[MissedRun]) -> list[tuple[str, str]]:
    """Apply the spec §9.2 replay policy to runs missed during downtime."""
    actions: list[tuple[str, str]] = []
    for run in missed:
        eligible = run.job in CATCH_UP_ELIGIBLE and run.same_local_day
        actions.append((run.job, "fire" if eligible else "skip"))
    return actions


class DegradedPresence:
    """Presence adapter used when the mic self-test failed.

    Speaking is pointless when the microphone is dead — ZEUS would talk into
    a void, hear nothing, and record NO_ANSWER, which is exactly the outcome
    audio_self_test exists to prevent (risk R1). So SPEAK becomes NOTIFY.
    DEFER passes through untouched: being locked or idle still means defer,
    regardless of microphone health.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def verdict(self) -> Verdict:
        verdict = self._inner.verdict()
        return Verdict.NOTIFY if verdict is Verdict.SPEAK else verdict


class SwitchablePresence:
    """One level of indirection so the daemon can downgrade after startup.

    The self-test runs after the CheckIns are built, and CheckIn stores its
    presence at construction. Handing every CheckIn this wrapper means a
    single degrade() call reaches all of them — no rebuilding them, and no
    reaching into their private attributes from outside.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def verdict(self) -> Verdict:
        return self._inner.verdict()

    def degrade(self) -> None:
        self._inner = DegradedPresence(self._inner)


class Daemon:
    def __init__(
        self, config: Config, store, journal, scheduler: Scheduler,
        presence, voice, notifier, checkins: dict[str, Any], clock: Clock,
        activator=None, mic=None, tz=None,
    ) -> None:
        self._config = config
        # The zone catch-up records local_date in — the same convention
        # goals.date and checkins.local_date use everywhere else. Derived
        # from config when not supplied so no caller is forced to pass it,
        # but build_daemon passes the one it already resolved rather than
        # resolving /etc/localtime a second time.
        self._tz = tz or resolve_timezone(config.schedule.timezone)
        self._store = store
        self._journal = journal
        self._scheduler = scheduler
        self._presence = presence
        self._voice = voice
        self._notifier = notifier
        self._checkins = checkins
        self._clock = clock
        self._activator = activator
        self._mic = mic
        self._running = False
        # Set when shutdown is requested. An Event, not just the _running
        # bool, because the activation thread has to WAIT on it: a plain
        # flag would have to be polled, and the restart backoff would then
        # be waited out in full before a stop() was noticed.
        self._shutdown = threading.Event()
        self._activation_thread: threading.Thread | None = None
        self.degraded = False

    def run_catch_up(self) -> list[tuple[str, str]]:
        missed = self._scheduler.catch_up()
        actions = catch_up_actions(missed)
        # zip, not a {job: latest} lookup. catch_up_actions returns ONE
        # entry per missed RUN, so a job with several missed occurrences
        # (two days of downtime gives morning@d1, evening@d1, morning@d2)
        # gets several entries -- and a by-job dict resolved every one of
        # them to that job's LATEST occurrence, so the morning@d1 entry
        # logged and consumed the timestamp of the occurrence about to be
        # fired two iterations later. The final set_job_run lands on the
        # same value either way (missed is ascending, last write wins), so
        # the durable state is unchanged; what was wrong was the timestamp
        # each individual decision was reported and recorded against.
        for run, (job, action) in zip(missed, actions):
            when = run.scheduled_for
            if action == "fire" and job in self._checkins:
                log.info("catch-up: firing %s missed at %s", job, when)
                try:
                    self._checkins[job].run(when)
                except Exception:
                    # Mirrors the isolation Scheduler.run_pending already
                    # gives the regular tick path (Task 6): one missed
                    # check-in that fails to fire (e.g. the Anthropic API is
                    # unreachable) must not abort catch-up for the runs
                    # after it, and must never crash start()/run_forever()
                    # before the daemon has ticked even once. See Finding #4.
                    log.exception("catch-up run for %r failed", job)
            else:
                log.info("catch-up: skipping %s missed at %s", job, when)
                self._record_skipped(job, when)
            # Consume the occurrence either way — fired OR skipped.
            # Scheduler.catch_up() reads the heartbeat while run_pending()
            # reads each job's last_run_at; they are separate state, so
            # without this the very next tick() recomputes from a stale
            # last_run_at and overrides the decision just made -- an
            # eligible "fire" gets re-run a second time (CheckIn either
            # reopens the same row and re-converses, or opens a second row
            # and asks again), and a "skip" gets un-skipped and asked about
            # anyway. Only reproducible after a restart, when the job row
            # already carries a last_run_at; a fresh store's first
            # run_pending merely seeds the baseline and fires nothing,
            # which is why this was invisible to every test until now.
            # See review round 2, Critical #1.
            self._store.set_job_run(job, when)
        return actions

    def _record_skipped(self, job: str, when: datetime) -> None:
        """Write the `skipped` row spec §9.2 asks for.

        "Morning check-in missed, day has rolled over -> do not fire; record
        outcome=skipped" and "Evening check-in missed -> record
        outcome=skipped". The skip branch used to log a line and call
        set_job_run, and that was all: nothing reached the `checkins` table,
        so a week of downtime left NO trace that seven mornings and seven
        evenings had been skipped. checkins.outcome has carried 'skipped'
        in its CHECK constraint since the first migration for exactly this,
        and D10's whole justification for building the action log in Slice
        1 is that the Slice 2 dashboard can only ever show what was
        recorded from the beginning.

        find_open_checkin, not raw SQL, and reusing an open row rather than
        always inserting: a morning that was deferred before the daemon
        died and then rolled over must have THAT row settled, not a second
        row opened beside it. 'answered' and 'skipped' are terminal to
        find_open_checkin, so an already-settled check-in is never
        overwritten.

        attempts is left where it was -- a skipped check-in never fired, so
        counting an attempt would be a lie in the one column the retry
        ladder reads. retry_at is cleared: a settled check-in must not
        still be pending a retry.
        """
        kind = job.removeprefix("checkin_")
        if kind not in ("morning", "evening"):
            return
        try:
            date = local_date(when, self._tz)
            existing = self._store.find_open_checkin(kind, date)
            if existing is None:
                checkin_id = self._store.open_checkin(kind, when, date)
                attempts = 0
            else:
                checkin_id = existing.id
                attempts = existing.attempts
            self._store.update_checkin(
                checkin_id, outcome="skipped", attempts=attempts, retry_at=None
            )
            self._journal.append(
                f"{kind.title()} check-in: skipped (missed while ZEUS was "
                f"not running)"
            )
        except Exception:
            # Same isolation as the fire branch above: recording a skip
            # must never abort catch-up or crash start() before the daemon
            # has ticked once.
            log.exception("could not record %r as skipped", job)

    def _run_due_retries(self, now: datetime) -> None:
        """Fire check-ins whose retry_at has arrived (spec §9.3).

        Re-runs with the ORIGINAL scheduled_for, so local_date and the
        check-in row stay the ones the ritual started with — a retry at 11:20
        belongs to the 11:00 occurrence, not to a new one. Passing `now`
        instead would compute a fresh local date at a day boundary, open a
        second row, restart `attempts` at 1, and never exhaust.

        The retry lives in the database rather than an in-process timer, so a
        daemon restarted between 11:00 and 11:20 still honours it — and with
        KeepAlive, restarts are routine.

        THAT DURABILITY IS ALSO WHY STALENESS HAS TO BE CHECKED HERE.
        due_retries() filters only on `retry_at <= now`, and a Mac that
        sleeps with the lid closed keeps both the row and the process. So a
        morning check-in that deferred at 11:00 fired its 11:20 rung at 09:00
        the next day, re-running with the ORIGINAL scheduled_for — right for
        a twenty-minute retry, catastrophic for a twenty-two-hour one, since
        `date` is then yesterday and save_goal's upsert rewrites yesterday's
        goal row (status and notes reset) while today records nothing.
        Reproduced in test_daemon.py.

        The bound is on the retry's AGE, not on its local calendar date. The
        catch-up path's same_local_day rule cannot be reused verbatim here:
        an evening check-in at 23:50 defers to 00:10, which is a live rung
        twenty minutes old that merely happens to fall on the next date, and
        the day rule would settle it as skipped and never review the day.
        Age also keeps the decision out of the UTC/local seam entirely,
        which is where six defects in this codebase have come from.
        """
        deadline = self._retry_deadline()
        for due in self._store.due_retries(now):
            if now - due.scheduled_for > deadline:
                self._settle_stale_retry(due, now)
                continue
            checkin = self._checkins.get(f"checkin_{due.kind}")
            if checkin is None:
                continue
            log.info("retry: firing %s for the occurrence at %s",
                     due.kind, due.scheduled_for)
            try:
                checkin.run(due.scheduled_for)
            except Exception:
                # One failing retry must not stop the others or the tick.
                log.error("retry for %s failed", due.kind, exc_info=True)

    def _retry_deadline(self) -> timedelta:
        """How long after its occurrence a retry may still fire (§9.2, §9.3).

        Derived from the configured ladder rather than hardcoded, so a user
        who lengthens defer_retry_after does not silently lose their last
        rungs — and capped, so one who lengthens it absurdly cannot licence a
        replay on the following day.
        """
        return min(self._configured_ladder() + _RETRY_SLACK, _RETRY_MAX_AGE)

    def _configured_ladder(self) -> timedelta:
        """The span the configured §9.3 ladder asks for, before the cap."""
        schedule = self._config.schedule
        return max(
            schedule.defer_retry_after * schedule.max_defer_retries,
            schedule.no_answer_retry_after * schedule.max_no_answer_retries,
        )

    def _settle_stale_retry(self, due, now: datetime) -> None:
        """Record a retry that outlived its ladder as skipped, and clear it.

        Settling is not optional housekeeping: retry_at is what due_retries()
        selects on, so a stale row that is merely skipped over comes back on
        the very next tick — once a minute, forever. update_checkin targets
        DueRetry.id, so it lands on exactly the row that was due rather than
        on whatever find_open_checkin would resolve for that (kind, date).

        attempts is carried across unchanged, for the reason _record_skipped
        gives: a retry that never fired is not an attempt, and attempts is
        the one column the §9.3 ladder reads.
        """
        # Name the CAP when the cap is what bit, not §9.2 generally. A user
        # who configured a ladder longer than _RETRY_MAX_AGE has their last
        # rungs dropped by ZEUS's policy, not by their own configuration, and
        # a message blaming "its moment has passed" reads as if their settings
        # were honoured. It is the difference between a log they can act on
        # and one that quietly contradicts their config.toml.
        capped = self._configured_ladder() + _RETRY_SLACK > _RETRY_MAX_AGE
        because = (
            f"that is past the {_RETRY_MAX_AGE} ceiling ZEUS puts on a retry, "
            f"which is shorter than your configured ladder — see "
            f"defer_retry_after/max_defer_retries in config.toml"
            if capped else
            "so its moment has passed (spec §9.2)"
        )
        log.warning(
            "retry: dropping the %s retry for the occurrence at %s — it came "
            "due %s after that occurrence, %s. Recording it as skipped.",
            due.kind, due.scheduled_for, now - due.scheduled_for, because,
        )
        try:
            attempts = self._store.get_checkin(due.id).attempts
            self._store.update_checkin(
                due.id, outcome="skipped", attempts=attempts, retry_at=None
            )
            self._journal.append(
                f"{due.kind.title()} check-in: skipped (its retry came due "
                f"long after the check-in's moment had passed)"
            )
        except Exception:
            # Same isolation as every other per-item failure on this path.
            # Loud, and the row keeps its retry_at, so the next tick tries
            # to settle it again rather than pretending it is gone.
            log.exception("could not settle the stale %s retry", due.kind)

    def tick(self) -> None:
        now = self._clock.now_utc()
        # Before run_pending, so a due retry is not delayed by a cron job's
        # work. The daemon already wakes at least once a minute
        # (seconds_until_next caps at _MAX_SLEEP), so a 20-minute retry is
        # observed within a minute of coming due — no new sleep machinery.
        self._run_due_retries(now)
        self._scheduler.run_pending(now)
        self._store.set_heartbeat()

    def _activation_loop(self) -> None:
        """Consume activation events, RE-ENTERING events() if it ever ends.

        The outer `while` is the whole point. events() is not an endless
        source any more: A1 gave Subscription.frames() a wall-clock idle
        bound, and when it fires the chain is unconditional — frames()
        returns, _detect returns, events() closes its subscription and ends.
        With a single `for` this thread simply returned, and nothing ever
        restarted it: start() launches it exactly once. The triggers are the
        ordinary ones mic.py lists — sleep/wake, AirPods taking over the
        default input, a USB mic unplugged, a coreaudiod restart — so any
        one of them over five seconds permanently ended "hey jarvis", with
        _running still True, nothing logged, and `doctor` unable to see it.
        Spec §10 exactly inverted.

        WARNING, not INFO: a microphone that stopped delivering is a real
        fault even though ZEUS recovers from it, and this line is the only
        evidence of it that ever reaches zeusd.log.

        The backoff is a floor on the cycle time, not the main rate limit —
        the idle bound already caps each cycle at _IDLE_TIMEOUT_SECONDS.
        What it protects against is an activation source that fails
        instantly and repeatedly. Waiting on _shutdown rather than sleeping
        means stop() cuts the wait short instead of being made to sit
        through it, which matters because launchd escalates to SIGKILL.
        """
        if self._activator is None:
            return
        restarts = 0
        while self._running and not self._shutdown.is_set():
            # Named rather than assumed: events() can also end by RAISING —
            # openWakeWord failing to load its model, say — and a log line
            # that blamed the microphone for that would be one more false
            # statement in a codebase that has already been bitten by five.
            reason = "the microphone stopped delivering audio"
            try:
                for event in self._activator.events():
                    if not self._running:
                        return
                    # A delivered event proves the source recovered, so the
                    # backoff starts over. Without this, a Mac that sleeps
                    # once a day would ratchet its way to the five-minute
                    # ceiling over a week and then take five minutes to
                    # bring "hey jarvis" back after an ordinary lid-open.
                    restarts = 0
                    log.info("activated via %s", event.source)
                    try:
                        self._handle_activation()
                    except Exception:
                        log.error("ad-hoc conversation failed", exc_info=True)
            except Exception:
                # A raising activator must not end activation either — that
                # is the same silent death by a different door.
                reason = "the activation source raised"
                log.error("the activation source failed", exc_info=True)
            if not self._running or self._shutdown.is_set():
                return
            restarts += 1
            wait = min(
                _ACTIVATION_RESTART_SECONDS * 2 ** (restarts - 1),
                _ACTIVATION_RESTART_MAX_SECONDS,
            )
            log.warning(
                "%s, so wake-word activation ended; restarting it in %.0fs "
                "(restart #%d). If this keeps repeating, run 'zeus doctor' — "
                "the input device or the wake-word model is failing.",
                reason, wait, restarts,
            )
            if self._shutdown.wait(wait):
                return

    def _handle_activation(self) -> None:
        """Ad-hoc conversation triggered by the wake word."""
        if self._voice is None:
            return
        # RAISED FIRST, before listen() does anything. The panel is the only
        # signal that ZEUS heard its name -- if it appeared after the capture
        # instead, the user would speak into a screen showing nothing and
        # have no way to tell whether the wake word had registered.
        overlay = getattr(self._voice, "_overlay", None)
        if overlay is not None:
            overlay.show(LISTENING)
        try:
            self._handle_turn()
        finally:
            if overlay is not None:
                overlay.hide()

    def _handle_turn(self) -> None:
        heard = self._voice.listen()
        if not heard:
            # The wake word fired, so the user IS talking to ZEUS — going
            # silent here is the worst possible answer. Spec §10: say it
            # once, then end the turn cleanly. No conversation is started,
            # so this costs no API call and works even when the brain is
            # the thing that is down.
            self._voice.speak([NOT_CAUGHT_LINE])
            return
        conversation_id = self._store.start_conversation("wake")
        try:
            conversation = self._checkins["_adhoc_factory"](conversation_id)
            self._voice.speak(conversation.send(heard))
        finally:
            self._store.end_conversation(conversation_id)

    def start(self) -> None:
        # Cleared FIRST, before _running is raised, so a restarted daemon
        # does not inherit the previous run's shutdown signal — and so a
        # request_stop() arriving during the slow part of start() below
        # cannot be erased by a later clear.
        self._shutdown.clear()
        self._running = True
        if self._mic is not None:
            self._mic.start()
            if not audio_self_test(self._mic):
                self.degraded = True
                # self._presence is a SwitchablePresence built once, up
                # front, and handed to both this Daemon and every CheckIn
                # (see build_daemon). degrade() flips it in place, so every
                # holder of that one shared object sees the change -- no
                # rebuilding CheckIns, no reaching into their attributes.
                self._presence.degrade()
                log.error(
                    "ZEUS is running in DEGRADED mode: no working microphone. "
                    "Check-ins will notify instead of speaking. "
                    "Run 'zeus doctor' for details."
                )
                if self._notifier is not None:
                    self._notifier.notify(
                        "ZEUS — microphone unavailable",
                        "Running in notification-only mode. Run 'zeus doctor'.",
                    )
        if self._activator is not None and not self.degraded:
            self._activator.start()
            # Kept as an attribute so shutdown is OBSERVABLE. Without a
            # handle, "the activation thread ended" can only be inferred
            # from threading.active_count(), which counts every other
            # thread in the process too -- an oracle that let an
            # uninterruptible backoff survive a mutation run.
            self._activation_thread = threading.Thread(
                target=self._activation_loop, daemon=True
            )
            self._activation_thread.start()
        self.run_catch_up()

    def request_stop(self) -> None:
        """Ask the daemon to shut down. SAFE TO CALL FROM A SIGNAL HANDLER.

        It flips a bool and sets an Event. That is all, and the restraint is
        the point: stop() itself is NOT safe from a handler. stop() ->
        MicStream.stop() takes _lifecycle_lock, a plain non-reentrant Lock
        that MicStream.start() holds across the sounddevice import and the
        PortAudio open — seconds, on a Bluetooth input. A signal handler
        runs on the main thread, which is the thread inside start(), so a
        SIGTERM in that window self-deadlocked the process until launchd
        escalated to SIGKILL. The comment in cli.py asserting the opposite
        was the fifth false comment in this codebase.

        run_forever's loop observes this within _SHUTDOWN_POLL_SECONDS and
        its `finally` does the real teardown, on a thread that holds no
        lock. The bound on shutdown is therefore whatever work is genuinely
        in flight — a check-in conversation waiting on the Anthropic API,
        say — rather than a deadlock that can never clear.
        """
        self._running = False
        self._shutdown.set()

    def stop(self) -> None:
        self.request_stop()
        if self._activator is not None:
            self._activator.stop()
        if self._mic is not None:
            self._mic.stop()

    def run_forever(self) -> None:
        self.start()
        try:
            while self._running:
                self.tick()
                self._sleep_until_next()
        finally:
            self.stop()

    def _sleep_until_next(self) -> None:
        """Sleep to the next occurrence, in slices, so a stop is noticed.

        SLICED, not one long sleep. seconds_until_next() returns up to
        _MAX_SLEEP (60s), and PEP 475 makes an interrupted time.sleep()
        resume for its full remaining time — so a SIGTERM arriving just
        after a tick would not be acted on for another minute, and launchd
        sends SIGKILL long before that. Slicing costs a wakeup a second on
        a process that is already running wake-word inference twelve times
        a second.

        Still self._clock.sleep, not _shutdown.wait: the Clock is the one
        seam the whole scheduler is tested through, and a FakeClock advances
        virtual time here exactly as a SystemClock burns real time.
        """
        remaining = self._scheduler.seconds_until_next(self._clock.now_utc())
        while remaining > 0 and not self._shutdown.is_set():
            interval = min(remaining, _SHUTDOWN_POLL_SECONDS)
            self._clock.sleep(interval)
            remaining -= interval


def build_daemon(config: Config | None = None, overlay=None) -> Daemon:
    """Construct a daemon from real components. See spec §5 for the wiring."""
    import anthropic

    from zeus.audio.activator import build_activator
    from zeus.brain.conversation import Conversation
    from zeus.brain.prompts import SYSTEM_PROMPT
    from zeus.brain.tools import build_tools
    from zeus.memory.journal import Journal
    from zeus.memory.store import Store
    from zeus.mcp import Confirmer, MCPRegistry, load_server_configs
    from zeus.ritual.checkin import CheckIn, MacNotifier, VoiceIO
    from zeus.stt import build_transcriber
    from zeus.tts import build_speaker

    config = config or load_config()
    config.log_path.parent.mkdir(parents=True, exist_ok=True)

    clock = SystemClock()
    tz = resolve_timezone(config.schedule.timezone)
    store = Store(config.db_path, clock)
    journal = Journal(config.journal_dir, clock, tz)

    mic = MicStream(config.audio)
    activator = build_activator(config.wake, mic)
    voice = VoiceIO(
        activator, mic,
        build_transcriber(config.stt, config.models_dir),
        build_speaker(config.tts), config.audio,
        overlay=overlay,
    )
    notifier = MacNotifier()
    client = anthropic.Anthropic()

    # MCP: everything ZEUS can reach beyond its own database. Started BEFORE
    # the factories below because they close over the tool list, and started
    # here rather than lazily because tool discovery costs an `npx` download
    # on first run -- paying that in the middle of a conversation would leave
    # the user listening to silence.
    #
    # The confirmer is bound to the SAME VoiceIO the conversation uses, which
    # is what makes §12's gate real: the question is asked through the
    # speaker and answered through the microphone, in the same turn, rather
    # than through some out-of-band prompt the user never sees.
    mcp = MCPRegistry(confirmer=Confirmer(voice), store=store)
    if config.mcp.enabled:
        try:
            mcp.start(load_server_configs(config.mcp.servers))
        except Exception:
            # A tool surface that fails to build must not cost the ritual.
            log.error("mcp: registry failed to start; continuing with the "
                      "built-in tools only", exc_info=True)
    mcp_tools = mcp.beta_tools()
    if mcp_tools:
        log.info("mcp: %d tool(s) available to the brain", len(mcp_tools))

    def conversation_factory(conversation_id: int, date: str):
        return Conversation(
            client=client, config=config.brain, store=store, journal=journal,
            conversation_id=conversation_id, system=SYSTEM_PROMPT,
            tools=build_tools(store, journal, conversation_id, date)
            + mcp_tools,
        )

    def adhoc_factory(conversation_id: int):
        date = local_date(clock.now_utc(), tz)
        return Conversation(
            client=client, config=config.brain, store=store, journal=journal,
            conversation_id=conversation_id, system=SYSTEM_PROMPT,
            tools=build_tools(store, journal, conversation_id, date)
            + mcp_tools,
            effort=config.brain.effort_adhoc,
        )

    from zeus.context.presence import Presence

    # Wrapped so a failed self-test can downgrade every CheckIn at once. The
    # CheckIns below are built before start() runs the self-test, and each
    # stores its presence at construction — this indirection is what lets
    # start() reach them afterwards.
    presence = SwitchablePresence(Presence(config.context))
    scheduler = Scheduler(store, clock, tz)

    checkins: dict[str, Any] = {"_adhoc_factory": adhoc_factory}
    for kind, hhmm in (
        ("morning", config.schedule.morning),
        ("evening", config.schedule.evening),
    ):
        name = f"checkin_{kind}"
        checkins[name] = CheckIn(
            kind=kind, store=store, journal=journal, presence=presence,
            voice=voice, notifier=notifier,
            conversation_factory=conversation_factory,
            config=config.schedule, tz=tz, clock=clock,
        )
        scheduler.register(
            name, hhmm_to_cron(hhmm),
            (lambda n: lambda when: checkins[n].run(when))(name),
        )

    return Daemon(
        config=config, store=store, journal=journal, scheduler=scheduler,
        presence=presence, voice=voice, notifier=notifier, checkins=checkins,
        clock=clock, activator=activator, mic=mic, tz=tz,
    )
