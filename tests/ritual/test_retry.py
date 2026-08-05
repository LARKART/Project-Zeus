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
