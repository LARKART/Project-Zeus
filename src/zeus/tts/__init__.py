from zeus.config import TtsConfig
from zeus.tts.base import Speaker
from zeus.tts.fake import FakeSpeaker
from zeus.tts.mac_say import MacSay

__all__ = ["Speaker", "MacSay", "FakeSpeaker", "build_speaker"]


def build_speaker(config: TtsConfig) -> Speaker:
    if config.provider == "mac_say":
        return MacSay(voice=config.voice)
    raise ValueError(f"unknown TTS provider: {config.provider!r}")
