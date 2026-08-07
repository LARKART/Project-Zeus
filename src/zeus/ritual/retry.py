"""Check-in retry state machine. See spec §9.3 — the authoritative source.

Two distinct retry paths with different causes, cadences, and limits:

  DEFER      user is away or the screen is locked   20 min × 3
  NO_ANSWER  ZEUS spoke into silence                30 min × 1

TWO, not three. NOTIFY is deliberately absent from that list, and from
§9.3's table — see next_step().

On exhaustion a morning check-in folds forward into the evening one; an
evening check-in is recorded as skipped, because there is nothing to fold
into.

Pure logic, no I/O — this is the most defect-prone rule set in the slice.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

from zeus.config import ScheduleConfig
from zeus.context.presence import Verdict


class Outcome(Enum):
    ANSWERED = "answered"
    NO_ANSWER = "no_answer"
    DEFERRED = "deferred"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    retry_after: timedelta | None
    fold_forward: bool


def _exhausted(kind: str, outcome: Outcome) -> Decision:
    """Morning folds into the evening check-in; evening has nowhere to go."""
    if kind == "morning":
        return Decision(outcome, None, True)
    return Decision(Outcome.SKIPPED, None, False)


def next_step(
    kind: str,
    verdict: Verdict,
    answered: bool | None,
    attempts: int,
    config: ScheduleConfig,
) -> Decision:
    """Decide what happens after one check-in attempt.

    `answered` is None when ZEUS never spoke (DEFER, or a NOTIFY that has
    not yet been acknowledged), True when the user replied, False when the
    listen window elapsed in silence.
    """
    if answered:
        return Decision(Outcome.ANSWERED, None, False)

    # NOTIFY WALKS THE DEFER LADDER, BUT NOTIFIES ONLY ONCE. The two halves
    # are separated on purpose, and this comment is the record of why —
    # getting it wrong has now produced a defect in each direction.
    #
    # Too many notifications: NOTIFY used to fall through to the DEFER
    # branch, which was harmless while run() still discarded
    # Decision.retry_after. Once Task 19 wired the ladder it became FOUR
    # notifications twenty minutes apart, because nothing anywhere can mark
    # a macOS notification "answered" so the ladder always ran to
    # exhaustion. DegradedPresence turns every SPEAK into NOTIFY, so in
    # degraded mode that is not an edge case but the guaranteed path: eight
    # a day until the microphone is fixed.
    #
    # Too few check-ins: the fix for that made NOTIFY terminal, which killed
    # the ladder outright. Measured — screen locked at 11:00 (DEFER, rung
    # booked for 11:20), a call or Focus session starting by 11:20 (NOTIFY),
    # and the morning was never asked again. A twenty-second call cost the
    # whole day's goal. §9.3's table gives NOTIFY no retry CAUSE, but it
    # does not contemplate the verdict changing mid-ladder, and the human
    # ruled on the reading: the ladder is the DEFER ladder, and NOTIFY
    # neither extends nor ends it.
    #
    # So the cadence is DEFER's, unchanged, and the ONCE is enforced where
    # the notification is actually sent — CheckIn.run() consults the row's
    # `notified` flag. Scheduling and side effect, kept apart, because that
    # is what let one defect masquerade as the other.
    #
    # `deferred`, not `skipped`: the check-in is unanswered, not abandoned,
    # and keeping the row non-terminal is what lets an unanswered morning
    # fold into the evening opener. `answered` is not consulted for NOTIFY
    # because ZEUS never spoke, so "spoke and heard nothing" cannot apply.
    if verdict is Verdict.NOTIFY or (verdict is Verdict.DEFER and answered is None):
        if attempts + 1 > config.max_defer_retries:
            return _exhausted(kind, Outcome.DEFERRED)
        return Decision(Outcome.DEFERRED, config.defer_retry_after, False)

    # ZEUS spoke and heard nothing back.
    if attempts + 1 > config.max_no_answer_retries:
        return _exhausted(kind, Outcome.NO_ANSWER)
    return Decision(Outcome.NO_ANSWER, config.no_answer_retry_after, False)
