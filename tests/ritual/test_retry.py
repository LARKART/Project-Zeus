from datetime import timedelta

import pytest

from zeus.config import ScheduleConfig
from zeus.context.presence import Verdict
from zeus.ritual.retry import Decision, Outcome, next_step

CONFIG = ScheduleConfig()  # 20m/3 defer, 30m/1 no-answer


def test_speaking_and_getting_an_answer_ends_the_sequence():
    result = next_step("morning", Verdict.SPEAK, answered=True, attempts=0, config=CONFIG)
    assert result == Decision(Outcome.ANSWERED, None, False)


def test_defer_schedules_a_twenty_minute_retry():
    result = next_step("morning", Verdict.DEFER, answered=None, attempts=0, config=CONFIG)
    assert result == Decision(Outcome.DEFERRED, timedelta(minutes=20), False)


@pytest.mark.parametrize("attempts", [0, 1, 2])
def test_defer_retries_up_to_three_times(attempts):
    result = next_step("morning", Verdict.DEFER, answered=None, attempts=attempts, config=CONFIG)
    assert result.retry_after == timedelta(minutes=20)


def test_defer_exhaustion_folds_a_morning_checkin_forward():
    result = next_step("morning", Verdict.DEFER, answered=None, attempts=3, config=CONFIG)
    assert result == Decision(Outcome.DEFERRED, None, True)


def test_defer_exhaustion_skips_an_evening_checkin():
    result = next_step("evening", Verdict.DEFER, answered=None, attempts=3, config=CONFIG)
    assert result == Decision(Outcome.SKIPPED, None, False)


def test_no_answer_schedules_a_thirty_minute_retry():
    result = next_step("morning", Verdict.SPEAK, answered=False, attempts=0, config=CONFIG)
    assert result == Decision(Outcome.NO_ANSWER, timedelta(minutes=30), False)


def test_no_answer_retries_only_once_then_folds():
    result = next_step("morning", Verdict.SPEAK, answered=False, attempts=1, config=CONFIG)
    assert result == Decision(Outcome.NO_ANSWER, None, True)


def test_no_answer_exhaustion_skips_an_evening_checkin():
    result = next_step("evening", Verdict.SPEAK, answered=False, attempts=1, config=CONFIG)
    assert result == Decision(Outcome.SKIPPED, None, False)


def test_notify_is_treated_as_deferred_until_acknowledged():
    # Deferred, but NOT retried -- this used to expect a 20-minute retry.
    # See the X3 block at the foot of this file for why that was wrong.
    result = next_step("morning", Verdict.NOTIFY, answered=None, attempts=0, config=CONFIG)
    assert result == Decision(Outcome.DEFERRED, None, False)


def test_notify_that_gets_answered_ends_the_sequence():
    result = next_step("evening", Verdict.NOTIFY, answered=True, attempts=2, config=CONFIG)
    assert result == Decision(Outcome.ANSWERED, None, False)


def test_outcome_values_match_the_database_constraint():
    assert {o.value for o in Outcome} == {"answered", "no_answer", "deferred", "skipped"}


def test_custom_config_is_honoured():
    config = ScheduleConfig(
        defer_retry_after=timedelta(minutes=5),
        max_defer_retries=1,
        no_answer_retry_after=timedelta(minutes=10),
        max_no_answer_retries=2,
    )
    assert next_step("morning", Verdict.DEFER, None, 0, config).retry_after == timedelta(minutes=5)
    assert next_step("morning", Verdict.DEFER, None, 1, config).retry_after is None
    assert next_step("morning", Verdict.SPEAK, False, 1, config).retry_after == timedelta(minutes=10)


# ---- X3: NOTIFY is not one of §9.3's retry paths ------------------------
#
# NOTIFY used to fall through to the DEFER branch, which was harmless while
# Task 19 was still dropping Decision.retry_after on the floor -- one
# notification per check-in. Once the ladder was wired, the same fall-through
# became four notifications twenty minutes apart, and nothing can ever mark a
# macOS notification "answered", so it always ran to exhaustion.
#
# In degraded mode (a failed mic self-test) DegradedPresence turns every
# SPEAK into NOTIFY, so this is not an edge case there but the guaranteed
# path: eight notifications a day, every day, until the mic is fixed.
#
# §9.3's table names exactly two retry causes, DEFER and NO_ANSWER. NOTIFY is
# not among them -- and the notification itself is the whole delivery: §8
# says it "speaks on click or wake word", so the user already holds a way
# back in that does not need ZEUS to ask again.


def test_notify_does_not_schedule_a_retry():
    result = next_step("morning", Verdict.NOTIFY, answered=None, attempts=0, config=CONFIG)
    assert result.retry_after is None, (
        "NOTIFY walked the DEFER ladder: four notifications per check-in, "
        "eight a day in degraded mode, where every check-in is a NOTIFY"
    )


def test_notify_leaves_the_checkin_open_rather_than_settling_it():
    """`deferred` is right; `skipped` and `answered` are both lies.

    The check-in has not been answered, and it has not been abandoned either
    -- the notification is sitting there waiting to be clicked. Keeping the
    row non-terminal is also what lets an unanswered morning fold into the
    evening opener, which is the same place an exhausted DEFER ladder ends
    up.
    """
    result = next_step("morning", Verdict.NOTIFY, answered=None, attempts=0, config=CONFIG)
    assert result.outcome is Outcome.DEFERRED


@pytest.mark.parametrize("attempts", [0, 1, 2, 3, 9])
def test_notify_never_schedules_a_retry_at_any_attempt_count(attempts):
    """No rung of the ladder, not just not the first.

    A NOTIFY landing partway through a DEFER ladder (the screen unlocks into
    a Focus session) must end the retries rather than continue them: the
    cause of THIS attempt is NOTIFY, and §9.3 gives NOTIFY no retry.
    """
    result = next_step(
        "evening", Verdict.NOTIFY, answered=None, attempts=attempts, config=CONFIG
    )
    assert result.retry_after is None


def test_defer_still_walks_the_ladder():
    """Guards the guard: the fix must not disarm the DEFER path with it."""
    assert next_step(
        "morning", Verdict.DEFER, answered=None, attempts=0, config=CONFIG
    ).retry_after == timedelta(minutes=20)
