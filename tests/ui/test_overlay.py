"""Overlay wiring.

No test here draws anything. Spec §13 forbids tests that need hardware, and
an NSApplication run loop never returns — it would hang the suite rather
than fail it. So these drive the seam through a recording double, and the
real panel is proved separately by tools/overlay_probe.py, which asks the
window server whether it was actually on screen.
"""
from __future__ import annotations

from zeus.ritual.checkin import VoiceIO
from zeus.ui.overlay import (
    LISTENING, SPEAKING, THINKING, NullOverlay, build_overlay,
)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.hidden = 0

    def show(self, state: str, text: str = "") -> None:
        self.calls.append((state, text))

    def hide(self) -> None:
        self.hidden += 1

    def states(self) -> list[str]:
        return [state for state, _ in self.calls]


class _Speaker:
    def __init__(self) -> None:
        self.said: list[str] = []

    def say(self, sentence: str) -> None:
        self.said.append(sentence)


class _Transcriber:
    def __init__(self, text: str = "ship zeus") -> None:
        self.text = text

    def transcribe(self, audio: bytes, rate: int) -> str:
        return self.text


class _Mic:
    """Speech, then enough silence for the endpointer to call it a turn.

    An empty frame source is NOT good enough here: capture_utterance returns
    empty, and listen() short-circuits before it ever transcribes — so the
    THINKING state these tests are about would never be reached, and the
    test would be asserting on a path the user never takes.
    """

    def frames(self):
        import numpy as np

        from zeus.audio.mic import FRAME_SAMPLES

        loud = (np.ones(FRAME_SAMPLES, dtype=np.int16) * 6000).tobytes()
        quiet = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()
        yield from [loud] * 10
        yield from [quiet] * 30


def _voice(overlay, transcriber=None):
    from zeus.config import AudioConfig
    return VoiceIO(None, _Mic(), transcriber or _Transcriber(), _Speaker(),
                   AudioConfig(), overlay=overlay)


def test_an_overlay_is_optional_everywhere():
    """Every existing caller builds a VoiceIO without one, and a LaunchAgent
    with no login session has nowhere to draw."""
    assert isinstance(_voice(None)._overlay, NullOverlay)
    assert isinstance(build_overlay(enabled=False), NullOverlay)


def test_speaking_paints_each_sentence_before_it_is_said():
    """`say` blocks for the whole utterance, so painting afterwards would
    show each line only once ZEUS had finished saying it."""
    recorder = _Recorder()
    voice = _voice(recorder)
    voice.speak(["Got it.", "Ship Zeus."])
    assert recorder.calls == [(SPEAKING, "Got it."), (SPEAKING, "Ship Zeus.")]


def test_listening_becomes_thinking_before_transcription():
    """Local Whisper is the slowest step in the turn. A panel still reading
    "Listening…" a second after the user stopped talking reads as "it did
    not hear me" and invites them to talk over the top of it."""
    recorder = _Recorder()
    voice = _voice(recorder)
    voice.listen()
    assert recorder.states()[0] == LISTENING
    assert THINKING in recorder.states()


def test_the_transcript_reaches_the_panel():
    recorder = _Recorder()
    _voice(recorder, _Transcriber("what is my one thing today")).listen()
    assert (THINKING, "what is my one thing today") in recorder.calls


def test_the_panel_is_raised_before_the_capture_and_always_dismissed():
    """The panel is the ONLY signal that ZEUS heard its name.

    Raised after the capture instead, the user would speak into a screen
    showing nothing, with no way to tell the wake word had registered — and
    an exception mid-turn must not leave it stuck on screen forever.
    """
    from types import SimpleNamespace

    from zeus.daemon import Daemon

    recorder = _Recorder()

    class _Boom:
        _overlay = recorder

        def listen(self):
            raise RuntimeError("the mic died mid-turn")

    daemon = Daemon.__new__(Daemon)
    daemon._voice = _Boom()

    try:
        daemon._handle_activation()
    except RuntimeError:
        pass
    assert recorder.states() == [LISTENING], "the panel must come up first"
    assert recorder.hidden == 1, "a failed turn left the panel on screen"


def test_a_null_overlay_refuses_to_pretend_it_has_a_run_loop():
    """cmd_run branches on this: with no panel, the daemon keeps the main
    thread. A NullOverlay that silently returned from run_forever() would
    make `zeus run --no-overlay` exit immediately instead."""
    import pytest

    with pytest.raises(RuntimeError, match="no run loop"):
        NullOverlay().run_forever()
