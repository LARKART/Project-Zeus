"""Speaker protocol. See spec §5.1."""
from __future__ import annotations

from typing import Protocol


class Speaker(Protocol):
    def say(self, text: str) -> None:
        """Speak `text`, blocking until playback finishes."""
        ...

    def stop(self) -> None:
        """Interrupt any playback in progress."""
        ...
