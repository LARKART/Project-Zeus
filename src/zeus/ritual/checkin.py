"""Check-in orchestration. See spec §7.2, §8, §9.3.

Wires the context gate, the retry state machine, the brain, and storage
into one run of the daily ritual.
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from typing import Callable, Iterable, Protocol
from zoneinfo import ZoneInfo

from zeus.audio.endpointer import capture_utterance
from zeus.audio.mic import FRAME_SAMPLES
from zeus.brain.prompts import EVENING_OPENER, FOLDED_OPENER, MORNING_OPENER
from zeus.clock import Clock
from zeus.config import ScheduleConfig
from zeus.context.presence import Verdict
from zeus.memory.journal import Journal
from zeus.memory.store import Store
from zeus.ritual.retry import Outcome, next_step

log = logging.getLogger(__name__)

MAX_EXCHANGES = 3


def local_date(moment: datetime, tz: ZoneInfo) -> str:
    """The calendar date in the user's zone — the key goals are stored under."""
    return moment.astimezone(tz).strftime("%Y-%m-%d")


# ---- notifications ----------------------------------------------------
class Notifier(Protocol):
    def notify(self, title: str, body: str) -> None: ...


class MacNotifier:
    def notify(self, title: str, body: str) -> None:
        script = f'display notification "{body}" with title "{title}"'
        try:
            subprocess.run(["osascript", "-e", script], timeout=5, check=False)
        except Exception:
            log.debug("notification failed", exc_info=True)


class FakeNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def notify(self, title: str, body: str) -> None:
        self.sent.append((title, body))


# ---- voice ------------------------------------------------------------
class VoiceIO:
    """Speak-then-listen, honouring the half-duplex rule (spec §7.3)."""

    def __init__(self, activator, mic, endpointer, transcriber, speaker, audio_config):
        self._activator = activator
        self._mic = mic
        self._endpointer = endpointer
        self._transcriber = transcriber
        self._speaker = speaker
        self._config = audio_config

    def speak(self, sentences: Iterable[str]) -> None:
        mute = getattr(self._activator, "mute", None)
        unmute = getattr(self._activator, "unmute", None)
        if mute:
            mute()
        try:
            for sentence in sentences:
                self._speaker.say(sentence)
        finally:
            if unmute:
                unmute()

    def listen(self) -> str:
        """Capture one utterance with the wake detector held muted.

        No drain() is needed any more: mic.frames() opens a FRESH
        subscription whose queue starts empty, so it cannot inherit the
        audio of ZEUS's own speech the way the old single shared queue
        could. That also fixes it for HotkeyActivator, which has no
        mute/unmute at all — the emptiness is structural now, not something
        an activator has to remember to do.

        The mute is the other half of the fan-out fix. Once every consumer
        gets its own copy of every frame, the detector no longer steals the
        user's answer — but it now HEARS all of it, and a "hey zeus" spoken
        mid-answer would launch an ad-hoc conversation on top of the running
        check-in. Muting for the whole listen window closes that. speak()
        mutes for its own duration; this covers the rest of the turn.
        """
        mute = getattr(self._activator, "mute", None)
        unmute = getattr(self._activator, "unmute", None)
        if mute:
            mute()
        try:
            frames_per_second = self._config.sample_rate / FRAME_SAMPLES
            timeout_frames = int(
                self._config.listen_timeout.total_seconds() * frames_per_second
            )
            audio = capture_utterance(
                self._mic.frames(), self._endpointer,
                pre_roll=b"", listen_timeout_frames=timeout_frames,
            )
        finally:
            if unmute:
                unmute()
        if not audio:
            return ""
        return self._transcriber.transcribe(audio, self._config.sample_rate)


# ---- the ritual -------------------------------------------------------
class CheckIn:
    def __init__(
        self, kind: str, store: Store, journal: Journal, presence, voice,
        notifier: Notifier, conversation_factory: Callable[[int, str], object],
        config: ScheduleConfig, tz: ZoneInfo, clock: Clock,
    ) -> None:
        self._kind = kind
        self._store = store
        self._journal = journal
        self._presence = presence
        self._voice = voice
        self._notifier = notifier
        self._conversation_factory = conversation_factory
        self._config = config
        self._tz = tz
        self._clock = clock

    def _opener(self, date: str) -> str:
        if self._kind == "morning":
            return MORNING_OPENER
        goal = self._store.get_goal(date)
        return EVENING_OPENER(goal.text) if goal else FOLDED_OPENER

    def _find_or_open(self, scheduled_for: datetime) -> int:
        """Reuse today's open check-in row so attempts accumulate across retries.

        Goes through Store.find_open_checkin rather than raw SQL. That method
        matches on the stored local_date, which matters twice: scheduled_for is
        stored as UTC (so a UTC-date match would miss an evening check-in in a
        western timezone), and the match is for THIS date only — an unresolved
        check-in left over from a previous day must not be reused, or today's
        first attempt would inherit yesterday's attempt count and could exhaust
        its retries before it has run once.

        The local date is computed once, here, and passed to both
        find_open_checkin (the lookup) and open_checkin (the write). An
        earlier version let open_checkin derive it from scheduled_for.date(),
        which is the UTC date, not the local one — the scheduler always
        produces UTC-tagged datetimes (cron.next_occurrence ends in
        .astimezone(timezone.utc)), so the write and the lookup silently
        disagreed for every check-in at a negative UTC offset, and every
        retry opened a fresh row instead of reusing the open one.
        """
        date = local_date(scheduled_for, self._tz)
        existing = self._store.find_open_checkin(self._kind, date)
        if existing is not None:
            return existing.id
        # Same `date` for the write as for the lookup — that identity is the
        # whole point; deriving it twice is how they drifted apart before.
        return self._store.open_checkin(self._kind, scheduled_for, date)

    def run(self, scheduled_for: datetime) -> Outcome:
        date = local_date(scheduled_for, self._tz)
        checkin_id = self._find_or_open(scheduled_for)
        previous = self._store.get_checkin(checkin_id).attempts
        verdict = self._presence.verdict()

        answered: bool | None = None

        if verdict is Verdict.NOTIFY:
            self._notifier.notify(
                "ZEUS", "Morning check-in" if self._kind == "morning"
                else "Evening check-in"
            )
        elif verdict is Verdict.SPEAK:
            try:
                answered = self._converse(checkin_id, date)
            except Exception:
                # An attempt that died is still an attempt. Before retries
                # were wired this merely lost a row update; with them, a
                # persistently failing brain (the Anthropic API unreachable,
                # say) would retry until the day ended, because the
                # exception escaped run() before update_checkin and so
                # `attempts` never incremented past whatever it already was.
                log.error("check-in conversation failed", exc_info=True)
                answered = False

        decision = next_step(
            self._kind, verdict, answered, previous, self._config
        )
        # Persisting retry_at is what makes the §9.3 ladder real. next_step
        # has always computed retry_after; until now run() read .outcome and
        # dropped the rest, so nothing anywhere scheduled a second attempt
        # and a user away from the desk at 11:00 was asked exactly once.
        #
        # Measured from NOW, not from scheduled_for: every rung re-runs with
        # the original occurrence, so anchoring on scheduled_for would pin
        # retry_at at 11:20 forever and turn the daemon tick into a hot loop.
        retry_at = (
            self._clock.now_utc() + decision.retry_after
            if decision.retry_after is not None else None
        )
        self._store.update_checkin(
            checkin_id,
            outcome=decision.outcome.value,
            attempts=previous + 1,
            fired_at=self._clock.now_utc() if verdict is Verdict.SPEAK else None,
            retry_at=retry_at,
        )
        if decision.outcome is Outcome.NO_ANSWER:
            self._journal.append(f"{self._kind.title()} check-in: no answer")
        elif decision.outcome is Outcome.SKIPPED:
            self._journal.append(f"{self._kind.title()} check-in: skipped")
        return decision.outcome

    def _converse(self, checkin_id: int, date: str) -> bool:
        conversation_id = self._store.start_conversation("schedule")
        conversation = self._conversation_factory(conversation_id, date)
        try:
            self._voice.speak(conversation.send(self._opener(date)))
            heard_anything = False
            for _ in range(MAX_EXCHANGES):
                reply = self._voice.listen()
                if not reply:
                    break
                heard_anything = True
                self._voice.speak(conversation.send(reply))
            return heard_anything
        finally:
            self._store.end_conversation(conversation_id)
