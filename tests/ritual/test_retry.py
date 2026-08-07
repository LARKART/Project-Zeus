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
    # Deferred, and retried on DEFER's cadence. The notification is capped at
    # one per check-in in CheckIn.run(), not here -- see the §9.3 block at
    # the foot of this file, which records both ways this has been wrong.
    result = next_step("morning", Verdict.NOTIFY, answered=None, attempts=0, config=CONFIG)
    assert result == Decision(Outcome.DEFERRED, timedelta(minutes=20), False)


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


# ---- §9.3: NOTIFY walks the DEFER ladder, and notifies once -------------
#
# This rule has been wrong in both directions, so both are pinned here.
#
# TOO MANY NOTIFICATIONS. NOTIFY fell through to the DEFER branch, harmless
# while Task 19 still dropped Decision.retry_after on the floor. Once the
# ladder was wired it meant four notifications twenty minutes apart, because
# nothing can ever mark a macOS notification "answered" so it always ran to
# exhaustion -- and DegradedPresence turns every SPEAK into NOTIFY, making
# that the guaranteed path after a failed mic self-test: eight a day.
#
# TOO FEW CHECK-INS. The fix for that made NOTIFY terminal, which killed the
# ladder: a call starting at 11:20, after an 11:00 defer, cost the whole
# day's goal.
#
# The ruling: the cadence is DEFER's, unchanged, and the ONCE is enforced in
# CheckIn.run() via the row's `notified` flag -- scheduling and side effect
# kept apart, since conflating them is what let each defect look like the
# fix for the other. §8 is why once is enough: the notification "speaks on
# click or wake word", so the first one is still sitting there.


@pytest.mark.parametrize("attempts", [0, 1, 2])
def test_notify_keeps_the_defer_ladder_running(attempts):
    """A passing call must not cost the day's goal.

    Screen locked at 11:00 -> DEFER, rung booked for 11:20. A Zoom call or
    Focus session starts by 11:20 -> NOTIFY. If that ends the ladder, the
    morning is never asked again and the evening opens with FOLDED_OPENER.
    """
    result = next_step(
        "morning", Verdict.NOTIFY, answered=None, attempts=attempts, config=CONFIG
    )
    assert result.retry_after == timedelta(minutes=20), (
        "a NOTIFY rung ended the DEFER ladder: one passing call at the wrong "
        "moment and the day gets no goal at all"
    )


def test_notify_exhausts_on_the_same_rung_defer_does():
    """It shares the ladder, so it shares the end of it -- not a longer one.

    attempts + 1 > max_defer_retries is the boundary; at max_defer_retries
    the ladder is spent and a morning folds forward into the evening.
    """
    spent = next_step(
        "morning", Verdict.NOTIFY, answered=None,
        attempts=CONFIG.max_defer_retries, config=CONFIG,
    )
    assert spent.retry_after is None
    assert spent.fold_forward is True


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


def test_defer_still_walks_the_ladder():
    """Guards the guard: the rule must not disarm the DEFER path with it."""
    assert next_step(
        "morning", Verdict.DEFER, answered=None, attempts=0, config=CONFIG
    ).retry_after == timedelta(minutes=20)
