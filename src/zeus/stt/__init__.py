from pathlib import Path

from zeus.config import SttConfig
from zeus.stt.base import Transcriber
from zeus.stt.fake import FakeTranscriber
from zeus.stt.local_whisper import LocalWhisper

__all__ = ["Transcriber", "LocalWhisper", "FakeTranscriber", "build_transcriber"]


def build_transcriber(config: SttConfig, models_dir: Path) -> Transcriber:
    if config.provider == "local_whisper":
        return LocalWhisper(config.model, config.compute, models_dir)
    raise ValueError(f"unknown STT provider: {config.provider!r}")
