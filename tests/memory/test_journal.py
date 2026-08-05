from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from zeus.clock import FakeClock
from zeus.memory.journal import Journal

LAGOS = ZoneInfo("Africa/Lagos")
# 10:00 UTC == 11:00 Lagos (UTC+1)
START = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)


def test_creates_file_with_header(tmp_path):
    journal = Journal(tmp_path, FakeClock(START), LAGOS)
    journal.append("Goal set: Finish the auth flow")
    assert journal.path_for("2026-08-05").exists()
    assert journal.read("2026-08-05").startswith("# 2026-08-05\n")


def test_entry_uses_local_time_not_utc(tmp_path):
    journal = Journal(tmp_path, FakeClock(START), LAGOS)
    journal.append("Goal set")
    assert "- 11:00 — Goal set" in journal.read("2026-08-05")


def test_appends_without_duplicating_header(tmp_path):
    clock = FakeClock(START)
    journal = Journal(tmp_path, clock, LAGOS)
    journal.append("First")
    clock.sleep(3600)
    journal.append("Second")
    text = journal.read("2026-08-05")
    assert text.count("# 2026-08-05") == 1
    assert "- 11:00 — First" in text
    assert "- 12:00 — Second" in text


def test_rolls_over_to_a_new_file_next_day(tmp_path):
    clock = FakeClock(START)
    journal = Journal(tmp_path, clock, LAGOS)
    journal.append("Day one")
    clock.sleep(24 * 3600)
    journal.append("Day two")
    assert "Day one" in journal.read("2026-08-05")
    assert "Day two" in journal.read("2026-08-06")


def test_read_missing_day_is_empty(tmp_path):
    journal = Journal(tmp_path, FakeClock(START), LAGOS)
    assert journal.read("1999-01-01") == ""
