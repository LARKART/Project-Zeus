from datetime import timedelta

import pytest

from zeus.config import ContextConfig
from zeus.context.presence import Presence, Signals, Verdict, decide

THRESHOLD = timedelta(minutes=15)


def sig(**kwargs) -> Signals:
    base = dict(
        screen_locked=False,
        idle=timedelta(0),
        focus_active=False,
        call_app_running=False,
    )
    base.update(kwargs)
    return Signals(**base)


@pytest.mark.parametrize(
    "signals,expected",
    [
        # DEFER wins over everything
        (sig(screen_locked=True), Verdict.DEFER),
        (sig(idle=timedelta(minutes=16)), Verdict.DEFER),
        (sig(screen_locked=True, focus_active=True), Verdict.DEFER),
        (sig(idle=timedelta(minutes=20), call_app_running=True), Verdict.DEFER),
        # NOTIFY when busy but present
        (sig(focus_active=True), Verdict.NOTIFY),
        (sig(call_app_running=True), Verdict.NOTIFY),
        (sig(focus_active=True, call_app_running=True), Verdict.NOTIFY),
        # SPEAK only when clearly free
        (sig(), Verdict.SPEAK),
        (sig(idle=timedelta(minutes=14, seconds=59)), Verdict.SPEAK),
    ],
)
def test_decision_table(signals, expected):
    assert decide(signals, THRESHOLD) == expected


def test_idle_threshold_boundary_is_exclusive():
    assert decide(sig(idle=THRESHOLD), THRESHOLD) == Verdict.SPEAK
    assert decide(sig(idle=THRESHOLD + timedelta(seconds=1)), THRESHOLD) == Verdict.DEFER


def test_presence_reads_all_four_probes(monkeypatch):
    import zeus.context.presence as mod

    monkeypatch.setattr(mod, "screen_locked", lambda: True)
    monkeypatch.setattr(mod, "idle_time", lambda: timedelta(seconds=5))
    monkeypatch.setattr(mod, "focus_active", lambda: False)
    monkeypatch.setattr(mod, "call_app_running", lambda names: False)

    presence = Presence(ContextConfig())
    signals = presence.read()
    assert signals.screen_locked is True
    assert signals.idle == timedelta(seconds=5)
    assert presence.verdict() == Verdict.DEFER


def test_probe_failure_degrades_to_safe_default(monkeypatch):
    """A broken probe must not crash the daemon; it reports the safe value."""
    import zeus.context.presence as mod

    def boom():
        raise OSError("ioreg exploded")

    monkeypatch.setattr(mod, "_raw_idle_nanoseconds", boom)
    assert mod.idle_time() == timedelta(0)
