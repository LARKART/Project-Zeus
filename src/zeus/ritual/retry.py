"""Check-in retry state machine. See spec §9.3 — the authoritative source.

Two distinct retry paths with different causes, cadences, and limits:

  DEFER      user is away or the screen is locked   20 min × 3
  NO_ANSWER  ZEUS spoke into silence                30 min × 1

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

    if verdict in (Verdict.DEFER, Verdict.NOTIFY) and answered is None:
        if attempts + 1 > config.max_defer_retries:
            return _exhausted(kind, Outcome.DEFERRED)
        return Decision(Outcome.DEFERRED, config.defer_retry_after, False)

    # ZEUS spoke and heard nothing back.
    if attempts + 1 > config.max_no_answer_retries:
        return _exhausted(kind, Outcome.NO_ANSWER)
    return Decision(Outcome.NO_ANSWER, config.no_answer_retry_after, False)
