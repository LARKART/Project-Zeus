from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import pytest

from zeus.clock import (
    FakeClock,
    SystemClock,
    from_utc_iso,
    resolve_timezone,
    to_utc_iso,
)


def test_resolve_system_timezone_returns_real_zone():
    tz = resolve_timezone("system")
    assert isinstance(tz, ZoneInfo)
    assert tz.key  # a real IANA name, not a fixed offset


def test_resolve_named_timezone():
    assert resolve_timezone("America/New_York").key == "America/New_York"


def test_system_clock_is_utc_aware():
    now = SystemClock().now_utc()
    assert now.tzinfo is timezone.utc


def test_fake_clock_advances_instead_of_blocking():
    start = datetime(2026, 8, 5, 11, 0, tzinfo=timezone.utc)
    clock = FakeClock(start)
    clock.sleep(90)
    assert clock.now_utc() == start + timedelta(seconds=90)
    assert clock.slept == [90]


def test_fake_clock_advance():
    start = datetime(2026, 8, 5, 11, 0, tzinfo=timezone.utc)
    clock = FakeClock(start)
    clock.advance(timedelta(hours=2))
    assert clock.now_utc() == datetime(2026, 8, 5, 13, 0, tzinfo=timezone.utc)


def test_iso_roundtrip_preserves_utc():
    dt = datetime(2026, 8, 5, 11, 0, tzinfo=timezone.utc)
    assert from_utc_iso(to_utc_iso(dt)) == dt


def test_iso_normalises_non_utc_input_to_utc():
    lagos = datetime(2026, 8, 5, 12, 0, tzinfo=ZoneInfo("Africa/Lagos"))
    text = to_utc_iso(lagos)
    assert text.endswith("+00:00")
    assert from_utc_iso(text) == datetime(2026, 8, 5, 11, 0, tzinfo=timezone.utc)


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError):
        to_utc_iso(datetime(2026, 8, 5, 11, 0))


def test_resolve_system_timezone_malformed_falls_back(monkeypatch):
    """Malformed system timezone (invalid IANA key) falls back to UTC."""
    with TemporaryDirectory() as tmpdir:
        # Create a fake zoneinfo directory with an invalid zone key
        zoneinfo_dir = Path(tmpdir) / "zoneinfo"
        zoneinfo_dir.mkdir()
        (zoneinfo_dir / "Bogus").mkdir()

        # Create a symlink pointing to the invalid zone
        broken_link = Path(tmpdir) / "localtime_broken"
        broken_link.symlink_to(zoneinfo_dir / "Bogus" / "Zone")

        # Patch _LOCALTIME to point to our broken symlink
        monkeypatch.setattr("zeus.clock._LOCALTIME", broken_link)

        # Should fall back to UTC instead of raising ZoneInfoNotFoundError
        tz = resolve_timezone("system")
        assert isinstance(tz, ZoneInfo)
        assert tz.key == "UTC"


def test_from_utc_iso_rejects_naive_string():
    """Naive ISO strings (no offset) are rejected symmetrically with to_utc_iso."""
    with pytest.raises(ValueError):
        from_utc_iso("2026-08-05T11:00:00")


def test_from_utc_iso_accepts_aware_strings():
    """Aware ISO strings with any offset are accepted and normalized to UTC."""
    # UTC offset
    utc_str = "2026-08-05T11:00:00+00:00"
    result = from_utc_iso(utc_str)
    assert result.tzinfo is timezone.utc
    assert result == datetime(2026, 8, 5, 11, 0, tzinfo=timezone.utc)

    # Non-UTC offset (Lagos is UTC+1)
    lagos_str = "2026-08-05T12:00:00+01:00"
    result = from_utc_iso(lagos_str)
    assert result.tzinfo is timezone.utc
    assert result == datetime(2026, 8, 5, 11, 0, tzinfo=timezone.utc)
