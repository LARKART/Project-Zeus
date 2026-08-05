import subprocess

import pytest

from zeus.config import TtsConfig
from zeus.tts.mac_say import MacSay
from zeus.tts import build_speaker


def test_invokes_say_with_the_configured_voice(monkeypatch):
    calls = []

    class DummyProcess:
        returncode = 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            calls.append("terminate")

    monkeypatch.setattr(
        "zeus.tts.mac_say.subprocess.Popen",
        lambda argv, **kw: calls.append(argv) or DummyProcess(),
    )
    MacSay(voice="Samantha").say("Hello there")
    assert calls[0] == ["/usr/bin/say", "-v", "Samantha", "Hello there"]


def test_empty_text_is_not_spoken(monkeypatch):
    called = []
    monkeypatch.setattr(
        "zeus.tts.mac_say.subprocess.Popen",
        lambda argv, **kw: called.append(argv),
    )
    MacSay(voice="Alex").say("   ")
    assert called == []


def test_stop_terminates_the_running_process(monkeypatch):
    terminated = []

    class DummyProcess:
        returncode = None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            terminated.append(True)

    monkeypatch.setattr(
        "zeus.tts.mac_say.subprocess.Popen", lambda argv, **kw: DummyProcess()
    )
    speaker = MacSay(voice="Alex")
    speaker.say("long sentence")
    speaker.stop()
    assert terminated == [True]


def test_a_wedged_say_is_killed_not_swallowed(monkeypatch):
    killed = []

    class DummyProcess:
        returncode = None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="say", timeout=120.0)

        def kill(self):
            killed.append(True)

    monkeypatch.setattr(
        "zeus.tts.mac_say.subprocess.Popen", lambda argv, **kw: DummyProcess()
    )
    MacSay(voice="Alex").say("anything")
    assert killed == [True]


def test_factory_builds_mac_say():
    speaker = build_speaker(TtsConfig(provider="mac_say", voice="Alex"))
    assert isinstance(speaker, MacSay)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown TTS provider"):
        build_speaker(TtsConfig(provider="elevenlabs"))
