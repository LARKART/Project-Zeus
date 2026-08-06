import threading
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


def test_concurrent_appends_do_not_lose_entries(tmp_path):
    """append() is check-then-act: it tests for the file, writes a header if
    absent, then opens for append. Two threads racing the first write of a
    day can both see "absent", and the second write_text (mode "w")
    truncates the first's entry.

    The daemon (Task 16) wires the scheduler thread and the wake-word
    activation thread to the SAME Journal, so this is reachable, not
    theoretical: measured 10-16 lost entries across 40 trials of 8
    concurrent writers, with no error raised — the line simply is not
    there.

    A race is probabilistic, so a single round proves nothing either way.
    Looping over many rounds, each with a fresh directory and a Barrier
    that releases all writers together so they genuinely contend on the
    first write, makes this a real guard rather than a coin flip.
    """
    n = 8
    rounds = 20
    for round_ in range(rounds):
        journal = Journal(tmp_path / f"round{round_}", FakeClock(START), LAGOS)
        barrier = threading.Barrier(n, timeout=5)

        def write(i: int) -> None:
            barrier.wait()
            journal.append(f"entry {i}")

        threads = [
            threading.Thread(target=write, args=(i,), daemon=True)
            for i in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive(), f"round {round_}: writer thread hung"

        text = journal.read("2026-08-05")
        for i in range(n):
            assert f"entry {i}" in text, (
                f"round {round_}: entry {i} lost to a concurrent-write race"
            )
        assert text.count("# 2026-08-05") == 1, (
            f"round {round_}: header duplicated or entries split across files"
        )
