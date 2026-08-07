"""Transcriber protocol. See spec §5.1."""
from __future__ import annotations

from typing import Protocol


class Transcriber(Protocol):
    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        """Transcribe 16-bit mono PCM. Returns '' if nothing intelligible."""
        ...
