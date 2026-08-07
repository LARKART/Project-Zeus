"""Transcriber test double. Returns scripted strings."""
from __future__ import annotations


class FakeTranscriber:
    def __init__(self, script: list[str] | None = None) -> None:
        self._script = list(script or [])
        self.calls: list[int] = []

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        self.calls.append(len(pcm))
        return self._script.pop(0) if self._script else ""
