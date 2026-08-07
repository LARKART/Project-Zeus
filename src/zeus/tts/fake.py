"""Speaker test double. Records rather than plays."""
from __future__ import annotations


class FakeSpeaker:
    def __init__(self) -> None:
        self.said: list[str] = []
        self.stopped: int = 0

    def say(self, text: str) -> None:
        self.said.append(text)

    def stop(self) -> None:
        self.stopped += 1
