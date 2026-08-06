"""The zeusd daemon: wiring, supervision, and the main loop. See spec §4, §9.2."""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any

from zeus.audio.endpointer import Endpointer, rms
from zeus.audio.mic import FRAME_SAMPLES, MicStream
from zeus.clock import Clock, SystemClock, resolve_timezone
from zeus.config import Config, load_config
from zeus.context.presence import Verdict
from zeus.schedule.cron import hhmm_to_cron
from zeus.schedule.scheduler import MissedRun, Scheduler

log = logging.getLogger(__name__)

# Only the morning check-in is ever replayed, and only on the same local
# day. A goal question at 15:00 is useful; the same question at 09:00 the
# next morning is noise. See spec §9.2.
CATCH_UP_ELIGIBLE = {"checkin_morning"}


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
        activator=None, mic=None,
    ) -> None:
        self._config = config
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

    def tick(self) -> None:
        now = self._clock.now_utc()
        self._scheduler.run_pending(now)
        self._store.set_heartbeat()

    def _activation_loop(self) -> None:
        if self._activator is None:
            return
        for event in self._activator.events():
            if not self._running:
                return
            log.info("activated via %s", event.source)
            try:
                self._handle_activation()
            except Exception:
                log.error("ad-hoc conversation failed", exc_info=True)

    def _handle_activation(self) -> None:
        """Ad-hoc conversation triggered by the wake word."""
        if self._voice is None:
            return
        heard = self._voice.listen()
        if not heard:
            return
        conversation_id = self._store.start_conversation("wake")
        try:
            conversation = self._checkins["_adhoc_factory"](conversation_id)
            self._voice.speak(conversation.send(heard))
        finally:
            self._store.end_conversation(conversation_id)

    def start(self) -> None:
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
            threading.Thread(target=self._activation_loop, daemon=True).start()
        self.run_catch_up()

    def stop(self) -> None:
        self._running = False
        if self._activator is not None:
            self._activator.stop()
        if self._mic is not None:
            self._mic.stop()

    def run_forever(self) -> None:
        self.start()
        try:
            while self._running:
                self.tick()
                self._clock.sleep(
                    self._scheduler.seconds_until_next(self._clock.now_utc())
                )
        finally:
            self.stop()


def build_daemon(config: Config | None = None) -> Daemon:
    """Construct a daemon from real components. See spec §5 for the wiring."""
    import anthropic

    from zeus.audio.activator import build_activator
    from zeus.brain.conversation import Conversation
    from zeus.brain.prompts import SYSTEM_PROMPT
    from zeus.brain.tools import build_tools
    from zeus.memory.journal import Journal
    from zeus.memory.store import Store
    from zeus.ritual.checkin import CheckIn, MacNotifier, VoiceIO, local_date
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
        activator, mic, Endpointer(config.audio),
        build_transcriber(config.stt, config.models_dir),
        build_speaker(config.tts), config.audio,
    )
    notifier = MacNotifier()
    client = anthropic.Anthropic()

    def conversation_factory(conversation_id: int, date: str):
        return Conversation(
            client=client, config=config.brain, store=store, journal=journal,
            conversation_id=conversation_id, system=SYSTEM_PROMPT,
            tools=build_tools(store, journal, conversation_id, date),
        )

    def adhoc_factory(conversation_id: int):
        date = local_date(clock.now_utc(), tz)
        return Conversation(
            client=client, config=config.brain, store=store, journal=journal,
            conversation_id=conversation_id, system=SYSTEM_PROMPT,
            tools=build_tools(store, journal, conversation_id, date),
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
        clock=clock, activator=activator, mic=mic,
    )
