"""macOS `say` speaker. Default provider for Slice 1 (spec D2)."""
from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)

SAY = "/usr/bin/say"

# A safety valve, not a deadline: `say` is meant to block for as long as the
# speech takes. 120s is ~350 spoken words at say's default rate, far beyond
# any single ZEUS utterance, so this only fires when the audio subsystem has
# wedged. Without it a stuck `say` would hang the ritual thread forever.
_MAX_SPEECH_SECONDS = 120.0


class MacSay:
    def __init__(self, voice: str = "Alex") -> None:
        self._voice = voice
        self._process: subprocess.Popen | None = None

    def say(self, text: str) -> None:
        if not text.strip():
            return
        try:
            self._process = subprocess.Popen([SAY, "-v", self._voice, text])
            self._process.wait(timeout=_MAX_SPEECH_SECONDS)
        except subprocess.TimeoutExpired:
            # Must precede the broad handler below: TimeoutExpired subclasses
            # Exception, so catching it there would swallow the hang and leave
            # the process alive. kill(), not terminate() — a process that has
            # already ignored us for two minutes has earned it.
            log.error("TTS timed out after %ss, killing", _MAX_SPEECH_SECONDS)
            try:
                self._process.kill()
            except Exception:
                log.debug("failed to kill wedged say", exc_info=True)
        except Exception:
            log.error("TTS failed for %r", text[:60], exc_info=True)

    def stop(self) -> None:
        process = self._process
        if process is not None and process.returncode is None:
            try:
                process.terminate()
            except Exception:
                log.debug("failed to terminate say", exc_info=True)
