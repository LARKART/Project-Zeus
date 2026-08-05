"""Human-readable daily journal. See spec §6."""
from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

from zeus.clock import Clock


class Journal:
    """One markdown file per local day, appended in local time.

    The journal exists so the user can read their own history with any text
    tool. It is deliberately not parsed back by ZEUS — the database is the
    source of truth.
    """

    def __init__(self, directory: Path, clock: Clock, tz: ZoneInfo) -> None:
        self._dir = directory
        self._clock = clock
        self._tz = tz
        self._dir.mkdir(parents=True, exist_ok=True)

    def _local_now(self):
        return self._clock.now_utc().astimezone(self._tz)

    def path_for(self, date: str) -> Path:
        return self._dir / f"{date}.md"

    def append(self, line: str) -> None:
        now = self._local_now()
        date = now.strftime("%Y-%m-%d")
        path = self.path_for(date)
        if not path.exists():
            path.write_text(f"# {date}\n\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"- {now.strftime('%H:%M')} — {line}\n")

    def read(self, date: str) -> str:
        path = self.path_for(date)
        return path.read_text(encoding="utf-8") if path.exists() else ""
