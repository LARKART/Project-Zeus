from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from zeus.schedule.cron import hhmm_to_cron, next_occurrence, occurrences_between

LAGOS = ZoneInfo("Africa/Lagos")       # UTC+1, no DST — the user's real zone
NEW_YORK = ZoneInfo("America/New_York")  # has DST — used to prove DST handling


def test_hhmm_to_cron():
    assert hhmm_to_cron("11:00") == "0 11 * * *"
    assert hhmm_to_cron("09:30") == "30 9 * * *"


@pytest.mark.parametrize("bad", ["", "25:00", "11", "11:60", "abc"])
def test_hhmm_to_cron_rejects_garbage(bad):
    with pytest.raises(ValueError):
        hhmm_to_cron(bad)


def test_next_occurrence_in_local_zone():
    # 08:00 UTC == 09:00 Lagos; next 11:00 Lagos is 10:00 UTC same day
    after = datetime(2026, 8, 5, 8, 0, tzinfo=timezone.utc)
    result = next_occurrence("0 11 * * *", after, LAGOS)
    assert result == datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)


def test_next_occurrence_rolls_to_tomorrow():
    after = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)  # 13:00 Lagos
    result = next_occurrence("0 11 * * *", after, LAGOS)
    assert result == datetime(2026, 8, 6, 10, 0, tzinfo=timezone.utc)


def test_occurrences_between_is_exclusive_of_start():
    start = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)   # exactly 11:00 Lagos
    end = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)
    found = occurrences_between("0 11 * * *", start, end, LAGOS)
    assert found == [
        datetime(2026, 8, 6, 10, 0, tzinfo=timezone.utc),
        datetime(2026, 8, 7, 10, 0, tzinfo=timezone.utc),
    ]


def test_occurrences_between_empty_when_none_due():
    start = datetime(2026, 8, 5, 10, 30, tzinfo=timezone.utc)
    end = datetime(2026, 8, 5, 11, 0, tzinfo=timezone.utc)
    assert occurrences_between("0 11 * * *", start, end, LAGOS) == []


def test_interval_expression_for_watchman_cadence():
    start = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)
    end = datetime(2026, 8, 5, 11, 0, tzinfo=timezone.utc)
    found = occurrences_between("*/30 * * * *", start, end, LAGOS)
    assert len(found) == 2  # 10:30 and 11:00 UTC


def test_local_wall_clock_survives_dst_transition():
    """11:00 must stay 11:00 local across the US spring-forward.

    Pinned to America/New_York deliberately: the machine's own zone
    (Africa/Lagos) has no DST, so testing against it would prove nothing.
    """
    # 2026-03-08 is the US spring-forward date.
    before = datetime(2026, 3, 7, 12, 0, tzinfo=timezone.utc)  # 07:00 EST
    first = next_occurrence("0 11 * * *", before, NEW_YORK)
    second = next_occurrence("0 11 * * *", first, NEW_YORK)

    assert first.astimezone(NEW_YORK).hour == 11
    assert second.astimezone(NEW_YORK).hour == 11
    # UTC offset shifts by an hour across the boundary — proving DST applied
    assert first.hour == 16   # EST  = UTC-5
    assert second.hour == 15  # EDT  = UTC-4
