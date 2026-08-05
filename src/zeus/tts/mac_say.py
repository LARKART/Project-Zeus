"""macOS `say` speaker. Default provider for Slice 1 (spec D2)."""
from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)

SAY = "/usr/bin/say"


class MacSay:
    def __init__(self, voice: str = "Alex") -> None:
        self._voice = voice
        self._process: subprocess.Popen | None = None

    def say(self, text: str) -> None:
        if not text.strip():
            return
        try:
            self._process = subprocess.Popen([SAY, "-v", self._voice, text])
            self._process.wait()
        except Exception:
            log.error("TTS failed for %r", text[:60], exc_info=True)

    def stop(self) -> None:
        process = self._process
        if process is not None and process.returncode is None:
            try:
                process.terminate()
            except Exception:
                log.debug("failed to terminate say", exc_info=True)
