"""Activation sources. See spec §5.1.

Two implementations ship from day one: the wake word the user asked for,
and a deterministic hotkey fallback that needs no Accessibility permission
and gives a reliable path when wake-word accuracy disappoints (risk R5).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

from zeus.config import WakeConfig


@dataclass(frozen=True)
class ActivationEvent:
    source: str


class Activator(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def events(self) -> Iterator[ActivationEvent]: ...


class FakeActivator:
    def __init__(self, count: int = 1) -> None:
        self._count = count
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def events(self) -> Iterator[ActivationEvent]:
        for _ in range(self._count):
            yield ActivationEvent("fake")


class HotkeyActivator:
    """Fires when a sentinel file appears, then deletes it.

    A file rather than a real global hotkey keeps Slice 1 free of
    Accessibility permissions. A shell alias or Shortcut can `touch` the
    file; Slice 2 can bind a real key to that.
    """

    def __init__(self, trigger_file: Path, poll_seconds: float = 0.25) -> None:
        self._trigger = trigger_file
        self._poll = poll_seconds
        self._running = False

    def start(self) -> None:
        self._running = True
        self._trigger.parent.mkdir(parents=True, exist_ok=True)
        self._trigger.unlink(missing_ok=True)

    def stop(self) -> None:
        self._running = False

    def events(self) -> Iterator[ActivationEvent]:
        while self._running:
            if self._trigger.exists():
                self._trigger.unlink(missing_ok=True)
                yield ActivationEvent("hotkey")
            if self._poll:
                time.sleep(self._poll)


def build_activator(config: WakeConfig, mic) -> Activator:
    from zeus.audio.wakeword import WakeWordActivator

    if config.provider == "openwakeword":
        return WakeWordActivator(mic, config)
    raise ValueError(f"unknown wake provider: {config.provider!r}")
