# ZEUS Slice 1 (Spine + Daily Ritual) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a macOS daemon that wakes on a spoken keyword, holds short spoken conversations, and proactively asks for the day's goal at 11:00 and reviews it at 21:00 — recording every goal, outcome, and tool call to a local database.

**Architecture:** One always-on Python process (`zeusd`) under a macOS LaunchAgent owns the single microphone stream and fans it out to a wake-word detector and to utterance capture. Audio → local Whisper → Claude (Anthropic Tool Runner) → macOS `say`. A generic cron-driven scheduler fires check-ins through a context gate that decides whether to speak, notify quietly, or defer. All state lands in SQLite (WAL) plus a human-readable markdown journal.

**Tech Stack:** Python 3.12.13 (via `uv`), `sounddevice`, `openwakeword`/`onnxruntime`, `faster-whisper`/`ctranslate2`, `anthropic` (Tool Runner), `pyobjc-framework-quartz`, `croniter`, stdlib `sqlite3`/`tomllib`/`zoneinfo`, `pytest`.

**Spec:** [docs/superpowers/specs/2026-08-05-zeus-voice-assistant-design.md](../specs/2026-08-05-zeus-voice-assistant-design.md)

---

## Global Constraints

Every task's requirements implicitly include this section.

- **Python 3.12.13 exactly.** Provisioned by `uv python install 3.12`. The system Python 3.14 has no wheels for the audio stack and must never be used.
- **No Homebrew, no `ffmpeg`, no `sox`.** Not installed and not to be required. `av` wheels bundle FFmpeg.
- **Target platform: macOS 12.7.6 (Monterey), Intel x86_64.** No Ventura-or-later APIs.
- **Pinned dependency versions** (verified to resolve on this machine): `sounddevice==0.5.5`, `openwakeword==0.6.0`, `onnxruntime==1.19.2`, `faster-whisper==1.2.1`, `ctranslate2==4.8.1`, `av==18.0.0`, `numpy==2.5.1`, `pyobjc-framework-quartz==12.2.1`, `croniter==6.2.4`.
- **Model ID is exactly `claude-opus-5`.** Never a date-suffixed variant.
- **Adaptive thinking stays ON.** Never send `thinking={"type": "disabled"}` — on Opus 5 that risks tool calls being emitted as visible text that silently never run.
- **All timestamps stored in the database are timezone-aware ISO-8601 UTC strings.** Schedules are expressed in local wall-clock time. Never store naive datetimes.
- **Raw audio is never written to disk.** Frames are transcribed and discarded.
- **The Anthropic API key is read from the `ANTHROPIC_API_KEY` environment variable only.** Never written to config or source.
- **No automated test may require a microphone, speakers, or a network call.** Hardware checks live only in the manual `zeus selftest` command.
- **Audio format throughout: 16 kHz, mono, `int16`.** openWakeWord consumes 1280-sample (80 ms) chunks.
- **Every commit runs `pytest` green before being made.**
- **Test directories carry no `__init__.py`, and `pyproject.toml` sets `addopts = ["--import-mode=importlib"]`.** These two go together and both are required. This plan gives several tasks a test file named `test_fake.py` in its own subdirectory (`tests/tts/`, `tests/stt/`, …). Under pytest's legacy `prepend` import mode, same-basename test modules in `__init__.py`-less directories collide and the suite fails to *collect at all* — `import file mismatch: imported module 'test_fake' has this __file__ attribute …`. Discovered in T10, when `tests/stt/test_fake.py` met T9's `tests/tts/test_fake.py`. `importlib` mode resolves it without `__init__.py` files and without renaming any file the plan declares.

---

## File Structure

```
Project Zeus/
├── pyproject.toml                  deps, pytest config, console script
├── .python-version                 "3.12"
├── .gitignore
├── src/zeus/
│   ├── __init__.py
│   ├── config.py          T1  config.toml → typed dataclasses; duration parsing
│   ├── clock.py           T2  Clock protocol, SystemClock, FakeClock, timezone resolution
│   ├── memory/
│   │   ├── schema.sql     T3  DDL for all 8 tables
│   │   ├── store.py       T3  SQLite façade (WAL, busy_timeout, UTC discipline)
│   │   └── journal.py     T4  markdown daily journal
│   ├── schedule/
│   │   ├── cron.py        T5  cron expression → next/between occurrences (tz-aware)
│   │   └── scheduler.py   T6  job loop + startup catch-up (§9.2)
│   ├── context/
│   │   └── presence.py    T7  screen lock · idle · Focus · call apps → Verdict
│   ├── ritual/
│   │   ├── retry.py       T8  pure retry state machine (§9.3)
│   │   └── checkin.py     T15 morning/evening orchestration
│   ├── tts/
│   │   ├── base.py        T9  Speaker protocol
│   │   ├── mac_say.py     T9  `say` implementation
│   │   └── fake.py        T9  FakeSpeaker (records utterances)
│   ├── stt/
│   │   ├── base.py        T10 Transcriber protocol
│   │   ├── local_whisper.py T10 faster-whisper implementation
│   │   └── fake.py        T10 FakeTranscriber (scripted strings)
│   ├── audio/
│   │   ├── mic.py         T11 MicStream: single owner, ring buffer, fan-out
│   │   ├── endpointer.py  T12 energy + silence → utterance boundary
│   │   ├── activator.py   T13 Activator protocol, FakeActivator, HotkeyActivator
│   │   └── wakeword.py    T13 WakeWordActivator (openWakeWord)
│   ├── brain/
│   │   ├── prompts.py     T14 system prompt + check-in scripts
│   │   ├── tools.py       T14 @beta_tool definitions + action-log wrapper
│   │   └── conversation.py T14 Tool Runner loop, sentence streaming
│   ├── daemon.py          T16 wiring, startup self-test, heartbeat
│   └── cli.py             T17 run · selftest · doctor · install-agent
└── tests/                          mirrors src/zeus/
```

**Decomposition rationale.** Files split by responsibility, not layer: everything that changes when the retry rules change lives in `ritual/retry.py`; everything that changes when the audio device changes lives in `audio/mic.py`. The three provider protocols (`Speaker`, `Transcriber`, `Activator`) each get their own package with a real and a fake implementation, because that pairing is what keeps the test suite hardware-free and makes the cloud swap a config edit.

---

## Task Dependency Order

```
T1 config ──┬── T2 clock ──┬── T3 store ── T4 journal
            │              ├── T5 cron ── T6 scheduler
            │              └── T8 retry
            ├── T7 presence
            ├── T9 tts   ┐
            ├── T10 stt  ├── T11 mic ── T12 endpointer ── T13 activator
            └── T14 brain ┘
                              ↓
                    T15 ritual ── T16 daemon ── T17 cli ── T18 e2e
```

---

### Task 1: Project scaffolding and configuration

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`
- Create: `src/zeus/__init__.py`, `src/zeus/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `parse_duration(text: str) -> datetime.timedelta` — accepts `"30s"`, `"1.5s"`, `"20m"`, `"2h"`.
  - `Config` dataclass with attributes `schedule`, `audio`, `wake`, `stt`, `tts`, `brain`, `context`, `privacy`, and `root: Path`.
  - `ScheduleConfig(timezone, morning, evening, defer_retry_after, max_defer_retries, no_answer_retry_after, max_no_answer_retries)`
  - `AudioConfig(sample_rate, ring_seconds, silence_timeout, listen_timeout)`
  - `WakeConfig(provider, model)`, `SttConfig(provider, model, compute)`, `TtsConfig(provider, voice)`
  - `BrainConfig(model, effort_checkin, effort_adhoc)`
  - `ContextConfig(idle_threshold, call_apps)`, `PrivacyConfig(transcript_retention_days)`
  - `load_config(path: Path | None = None, root: Path | None = None) -> Config` — returns full defaults when the file is absent.

- [ ] **Step 1: Create the project skeleton**

`.python-version`:
```
3.12
```

`.gitignore` — **already exists; verify, do not overwrite.** It was written
before plan execution and additionally ignores `.worktrees/` and
`.superpowers/`, which must survive. Confirm it contains every entry below
and add any that are missing:

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
models/
*.db
*.db-wal
*.db-shm
```

`pyproject.toml`:
```toml
[project]
name = "zeus"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "anthropic",
    "croniter==6.2.4",
    "sounddevice==0.5.5",
    "openwakeword==0.6.0",
    "onnxruntime==1.19.2",
    "faster-whisper==1.2.1",
    "ctranslate2==4.8.1",
    "av==18.0.0",
    "numpy==2.5.1",
    "pyobjc-framework-quartz==12.2.1",
]

[project.scripts]
zeus = "zeus.cli:main"

[project.optional-dependencies]
dev = ["pytest==8.3.4"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/zeus"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Then run:
```bash
uv python install 3.12
uv venv --python 3.12
uv pip install -e ".[dev]"
mkdir -p src/zeus tests
touch src/zeus/__init__.py
```

- [ ] **Step 2: Write the failing test**

`tests/test_config.py`:
```python
from datetime import timedelta
from pathlib import Path

import pytest

from zeus.config import load_config, parse_duration


@pytest.mark.parametrize(
    "text,expected",
    [
        ("30s", timedelta(seconds=30)),
        ("1.5s", timedelta(seconds=1.5)),
        ("20m", timedelta(minutes=20)),
        ("2h", timedelta(hours=2)),
    ],
)
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected


def test_parse_duration_rejects_garbage():
    with pytest.raises(ValueError):
        parse_duration("soon")


def test_defaults_when_file_absent(tmp_path):
    cfg = load_config(path=tmp_path / "nope.toml", root=tmp_path)
    assert cfg.schedule.morning == "11:00"
    assert cfg.schedule.evening == "21:00"
    assert cfg.schedule.defer_retry_after == timedelta(minutes=20)
    assert cfg.schedule.max_defer_retries == 3
    assert cfg.schedule.no_answer_retry_after == timedelta(minutes=30)
    assert cfg.schedule.max_no_answer_retries == 1
    assert cfg.brain.model == "claude-opus-5"
    assert cfg.audio.sample_rate == 16000
    assert cfg.root == tmp_path


def test_file_overrides_defaults(tmp_path):
    (tmp_path / "config.toml").write_text(
        '[schedule]\nmorning = "09:30"\n[tts]\nvoice = "Samantha"\n'
    )
    cfg = load_config(path=tmp_path / "config.toml", root=tmp_path)
    assert cfg.schedule.morning == "09:30"
    assert cfg.tts.voice == "Samantha"
    assert cfg.schedule.evening == "21:00"  # untouched default survives
```

- [ ] **Step 3: Run the test and verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.config'`

- [ ] **Step 4: Implement `src/zeus/config.py`**

```python
"""Configuration loading for ZEUS. See spec §11."""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(s|m|h)\s*$")
_UNITS = {"s": "seconds", "m": "minutes", "h": "hours"}


def parse_duration(text: str) -> timedelta:
    """Parse '30s' / '1.5s' / '20m' / '2h' into a timedelta."""
    match = _DURATION.match(text)
    if not match:
        raise ValueError(f"invalid duration: {text!r} (expected e.g. '30s', '20m', '2h')")
    value, unit = match.groups()
    return timedelta(**{_UNITS[unit]: float(value)})


@dataclass
class ScheduleConfig:
    timezone: str = "system"
    morning: str = "11:00"
    evening: str = "21:00"
    defer_retry_after: timedelta = timedelta(minutes=20)
    max_defer_retries: int = 3
    no_answer_retry_after: timedelta = timedelta(minutes=30)
    max_no_answer_retries: int = 1


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    ring_seconds: int = 3
    silence_timeout: timedelta = timedelta(seconds=1.5)
    listen_timeout: timedelta = timedelta(seconds=30)


@dataclass
class WakeConfig:
    provider: str = "openwakeword"
    model: str = "hey_jarvis"


@dataclass
class SttConfig:
    provider: str = "local_whisper"
    model: str = "base.en"
    compute: str = "int8"


@dataclass
class TtsConfig:
    provider: str = "mac_say"
    voice: str = "Alex"


@dataclass
class BrainConfig:
    model: str = "claude-opus-5"
    effort_checkin: str = "low"
    effort_adhoc: str = "medium"


@dataclass
class ContextConfig:
    idle_threshold: timedelta = timedelta(minutes=15)
    call_apps: list[str] = field(
        default_factory=lambda: ["zoom.us", "Microsoft Teams", "FaceTime"]
    )


@dataclass
class PrivacyConfig:
    transcript_retention_days: int = 90


@dataclass
class Config:
    root: Path
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)

    @property
    def db_path(self) -> Path:
        return self.root / "zeus.db"

    @property
    def journal_dir(self) -> Path:
        return self.root / "journal"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def log_path(self) -> Path:
        return self.root / "logs" / "zeusd.log"

    @property
    def env_path(self) -> Path:
        """Where the API key lives for launchd's benefit.

        NOT config.toml, and never written by ZEUS — it holds a secret, and
        the spec says the key is environment-only. cmd_run loads this file
        into os.environ at startup because launchd inherits no shell
        environment; the LaunchAgent plist carries only this PATH, never the
        key itself.
        """
        return self.root / "env"


def _apply(section: Any, values: dict[str, Any]) -> None:
    """Overlay TOML values onto a dataclass, converting duration strings."""
    known = {f.name: f for f in fields(section)}
    for key, raw in values.items():
        if key not in known:
            raise ValueError(f"unknown config key: {key!r}")
        if known[key].type is timedelta or known[key].type == "timedelta":
            raw = parse_duration(raw)
        setattr(section, key, raw)


DEFAULT_ROOT = Path.home() / ".zeus"


def load_config(path: Path | None = None, root: Path | None = None) -> Config:
    """Load config.toml, falling back to full defaults when absent."""
    root = root or DEFAULT_ROOT
    path = path if path is not None else root / "config.toml"
    config = Config(root=root)
    if not path.exists():
        return config
    data = tomllib.loads(path.read_text())
    for name, values in data.items():
        section = getattr(config, name, None)
        if section is None or not is_dataclass(section):
            raise ValueError(f"unknown config section: {name!r}")
        _apply(section, values)
    return config
```

- [ ] **Step 5: Run the test and verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS — 7 tests

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version .gitignore src/zeus/ tests/test_config.py
git commit -m "feat: project scaffolding and configuration loading"
```

---

### Task 2: Clock and timezone resolution

**Files:**
- Create: `src/zeus/clock.py`
- Test: `tests/test_clock.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `resolve_timezone(name: str) -> ZoneInfo` — `"system"` reads the `/etc/localtime` symlink.
  - `Clock` protocol: `now_utc() -> datetime` (tz-aware UTC), `sleep(seconds: float) -> None`.
  - `SystemClock()` — real implementation.
  - `FakeClock(start: datetime)` — `advance(delta)`, `sleep()` advances instead of blocking, `slept: list[float]` records calls.
  - `to_utc_iso(dt: datetime) -> str` and `from_utc_iso(text: str) -> datetime` — the database serialization pair.

- [ ] **Step 1: Write the failing test**

`tests/test_clock.py`:
```python
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from zeus.clock import (
    FakeClock,
    SystemClock,
    from_utc_iso,
    resolve_timezone,
    to_utc_iso,
)


def test_resolve_system_timezone_returns_real_zone():
    tz = resolve_timezone("system")
    assert isinstance(tz, ZoneInfo)
    assert tz.key  # a real IANA name, not a fixed offset


def test_resolve_named_timezone():
    assert resolve_timezone("America/New_York").key == "America/New_York"


def test_system_clock_is_utc_aware():
    now = SystemClock().now_utc()
    assert now.tzinfo is timezone.utc


def test_fake_clock_advances_instead_of_blocking():
    start = datetime(2026, 8, 5, 11, 0, tzinfo=timezone.utc)
    clock = FakeClock(start)
    clock.sleep(90)
    assert clock.now_utc() == start + timedelta(seconds=90)
    assert clock.slept == [90]


def test_fake_clock_advance():
    start = datetime(2026, 8, 5, 11, 0, tzinfo=timezone.utc)
    clock = FakeClock(start)
    clock.advance(timedelta(hours=2))
    assert clock.now_utc() == datetime(2026, 8, 5, 13, 0, tzinfo=timezone.utc)


def test_iso_roundtrip_preserves_utc():
    dt = datetime(2026, 8, 5, 11, 0, tzinfo=timezone.utc)
    assert from_utc_iso(to_utc_iso(dt)) == dt


def test_iso_normalises_non_utc_input_to_utc():
    lagos = datetime(2026, 8, 5, 12, 0, tzinfo=ZoneInfo("Africa/Lagos"))
    text = to_utc_iso(lagos)
    assert text.endswith("+00:00")
    assert from_utc_iso(text) == datetime(2026, 8, 5, 11, 0, tzinfo=timezone.utc)


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError):
        to_utc_iso(datetime(2026, 8, 5, 11, 0))
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv/bin/pytest tests/test_clock.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.clock'`

- [ ] **Step 3: Implement `src/zeus/clock.py`**

```python
"""Time handling. See spec §6.3: store UTC, schedule in local wall-clock."""
from __future__ import annotations

import os
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

_LOCALTIME = Path("/etc/localtime")


def resolve_timezone(name: str) -> ZoneInfo:
    """Resolve a config timezone value into a DST-aware ZoneInfo.

    'system' follows the /etc/localtime symlink to recover the IANA zone
    name. A fixed-offset tzinfo (what datetime.astimezone() yields) is NOT
    acceptable here: it cannot represent a future DST transition.
    """
    if name != "system":
        return ZoneInfo(name)
    try:
        parts = _LOCALTIME.resolve().parts
        index = len(parts) - 1 - parts[::-1].index("zoneinfo")
        return ZoneInfo("/".join(parts[index + 1 :]))
    except (OSError, ValueError, KeyError):
        # KeyError is required, not defensive padding: ZoneInfoNotFoundError
        # subclasses KeyError, and it is exactly what ZoneInfo() raises when
        # the symlink resolves through a `zoneinfo` directory but the trailing
        # segment is not a valid IANA key. Without it a malformed system
        # timezone raises straight past the TZ and UTC fallbacks below.
        pass
    env = os.environ.get("TZ")
    if env:
        try:
            return ZoneInfo(env)
        except Exception:
            pass
    return ZoneInfo("UTC")


def to_utc_iso(dt: datetime) -> str:
    """Serialize an aware datetime as an ISO-8601 UTC string."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime rejected; all timestamps must be aware")
    return dt.astimezone(timezone.utc).isoformat()


def from_utc_iso(text: str) -> datetime:
    """Parse an ISO-8601 string back into an aware UTC datetime.

    Rejects naive strings rather than converting them. `.astimezone()` on a
    naive datetime silently interprets it in the *system local* zone, which
    would quietly mis-time any row that ever lost its offset. The write side
    already refuses naive input; the read side must match.
    """
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"naive timestamp rejected: {text!r}")
    return parsed.astimezone(timezone.utc)


class Clock(Protocol):
    def now_utc(self) -> datetime: ...
    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def sleep(self, seconds: float) -> None:
        _time.sleep(seconds)


class FakeClock:
    """Deterministic clock. sleep() advances time instead of blocking."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FakeClock requires an aware datetime")
        self._now = start.astimezone(timezone.utc)
        self.slept: list[float] = []

    def now_utc(self) -> datetime:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self._now += timedelta(seconds=seconds)

    def advance(self, delta: timedelta) -> None:
        self._now += delta
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `.venv/bin/pytest tests/test_clock.py -v`
Expected: PASS — 11 tests (8 above, plus 3 covering the two fallback/rejection
paths added in review: a malformed `/etc/localtime` degrading to UTC rather
than raising, and `from_utc_iso` rejecting a naive string.)

- [ ] **Step 5: Commit**

```bash
git add src/zeus/clock.py tests/test_clock.py
git commit -m "feat: clock abstraction and DST-aware timezone resolution"
```

---

### Task 3: SQLite store

**Files:**
- Create: `src/zeus/memory/__init__.py`, `src/zeus/memory/schema.sql`, `src/zeus/memory/store.py`
- Test: `tests/memory/test_store.py`

**Interfaces:**
- Consumes: `zeus.clock.Clock`, `to_utc_iso`, `from_utc_iso`.
- Produces `Store(db_path: Path, clock: Clock)` with:
  - `set_goal(date: str, text: str) -> int`, `get_goal(date: str) -> Goal | None`
  - `update_goal(goal_id: int, status: str, notes: str | None = None) -> None`
  - `open_checkin(kind: str, scheduled_for: datetime, local_date: str) -> int` — `local_date` is explicit, never inferred from `scheduled_for`; see the docstring for why.
  - `get_checkin(checkin_id: int) -> CheckIn`, `find_open_checkin(kind, date) -> CheckIn | None`
  - `update_checkin(checkin_id: int, *, outcome: str, attempts: int, fired_at: datetime | None = None) -> None`
  - `log_action(tool, args, result, ok, duration_ms, error=None, conversation_id=None) -> int`
  - `recent_actions(limit: int = 50) -> list[Action]`
  - `start_conversation(trigger: str) -> int`, `add_message(conversation_id, role, content) -> None`, `end_conversation(conversation_id) -> None`
  - `set_fact(key, value, source) -> None`, `get_fact(key) -> str | None`
  - `upsert_job(name, schedule) -> None`, `jobs() -> list[Job]`, `set_job_run(name, last_run_at) -> None`
  - `set_heartbeat() -> None`, `heartbeat() -> datetime | None`
  - Dataclasses `Goal`, `CheckIn`, `Action`, `Job`.

- [ ] **Step 1: Write `src/zeus/memory/schema.sql`**

```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS goals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL UNIQUE,          -- local YYYY-MM-DD
    text        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','done','partial','missed','carried')),
    set_at      TEXT NOT NULL,                 -- ISO-8601 UTC
    reviewed_at TEXT,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS checkins (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT NOT NULL CHECK (kind IN ('morning','evening')),
    local_date    TEXT NOT NULL,                 -- local YYYY-MM-DD, same convention as goals.date
    scheduled_for TEXT NOT NULL,
    fired_at      TEXT,
    outcome       TEXT NOT NULL DEFAULT 'deferred'
                  CHECK (outcome IN ('answered','no_answer','deferred','skipped')),
    attempts      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_checkins_sched ON checkins (scheduled_for);
CREATE INDEX IF NOT EXISTS idx_checkins_local_date ON checkins (local_date, kind);

CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    trigger    TEXT NOT NULL CHECK (trigger IN ('wake','schedule'))
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations (id),
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    ts              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages (conversation_id);

CREATE TABLE IF NOT EXISTS actions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    conversation_id INTEGER REFERENCES conversations (id),
    tool            TEXT NOT NULL,
    args_json       TEXT NOT NULL,
    result_json     TEXT,
    ok              INTEGER NOT NULL,
    duration_ms     INTEGER NOT NULL,
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_actions_ts ON actions (ts);

CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key        TEXT NOT NULL UNIQUE,
    value      TEXT NOT NULL,
    learned_at TEXT NOT NULL,
    source     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    name        TEXT PRIMARY KEY,
    schedule    TEXT NOT NULL,
    last_run_at TEXT,
    next_run_at TEXT,
    enabled     INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS heartbeat (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    ts TEXT NOT NULL
);
```

- [ ] **Step 2: Write the failing test**

`tests/memory/test_store.py`:
```python
from datetime import datetime, timezone

import pytest

from zeus.clock import FakeClock
from zeus.memory.store import Store

START = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "zeus.db", FakeClock(START))


def test_wal_mode_is_enabled(store):
    mode = store.connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_goal_roundtrip(store):
    store.set_goal("2026-08-05", "Finish the auth flow")
    goal = store.get_goal("2026-08-05")
    assert goal.text == "Finish the auth flow"
    assert goal.status == "pending"
    assert goal.set_at == START


def test_set_goal_replaces_same_day(store):
    store.set_goal("2026-08-05", "First")
    store.set_goal("2026-08-05", "Second")
    assert store.get_goal("2026-08-05").text == "Second"


def test_update_goal_status(store):
    goal_id = store.set_goal("2026-08-05", "Ship it")
    store.update_goal(goal_id, status="partial", notes="tests missing")
    goal = store.get_goal("2026-08-05")
    assert goal.status == "partial"
    assert goal.notes == "tests missing"
    assert goal.reviewed_at == START


def test_invalid_goal_status_rejected(store):
    goal_id = store.set_goal("2026-08-05", "Ship it")
    with pytest.raises(Exception):
        store.update_goal(goal_id, status="banana")


def test_checkin_lifecycle(store):
    cid = store.open_checkin("morning", START, "2026-08-05")
    assert store.get_checkin(cid).outcome == "deferred"
    store.update_checkin(cid, outcome="answered", attempts=1, fired_at=START)
    checkin = store.get_checkin(cid)
    assert checkin.outcome == "answered"
    assert checkin.attempts == 1
    assert checkin.fired_at == START


def test_action_log(store):
    store.log_action("save_goal", {"text": "x"}, {"ok": True}, True, 42)
    store.log_action("save_goal", {"text": "y"}, None, False, 7, error="boom")
    actions = store.recent_actions()
    assert len(actions) == 2
    assert actions[0].ok is False and actions[0].error == "boom"
    assert actions[1].tool == "save_goal" and actions[1].args["text"] == "x"


def test_conversation_and_messages(store):
    conv = store.start_conversation("schedule")
    store.add_message(conv, "assistant", "Morning. What's the goal?")
    store.add_message(conv, "user", "Finish auth.")
    store.end_conversation(conv)
    rows = store.messages(conv)
    assert [r.role for r in rows] == ["assistant", "user"]


def test_facts(store):
    store.set_fact("wake_hour", "07:30", source="observed")
    assert store.get_fact("wake_hour") == "07:30"
    store.set_fact("wake_hour", "08:00", source="observed")
    assert store.get_fact("wake_hour") == "08:00"


def test_jobs_and_heartbeat(store):
    store.upsert_job("checkin_morning", "0 11 * * *")
    store.upsert_job("checkin_morning", "0 9 * * *")  # idempotent update
    jobs = store.jobs()
    assert len(jobs) == 1 and jobs[0].schedule == "0 9 * * *"

    assert store.heartbeat() is None
    store.set_heartbeat()
    assert store.heartbeat() == START
```

- [ ] **Step 3: Run the test and verify it fails**

Run: `.venv/bin/pytest tests/memory/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.memory'`

- [ ] **Step 4: Implement `src/zeus/memory/store.py`**

```python
"""SQLite persistence. See spec §6."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from zeus.clock import Clock, from_utc_iso, to_utc_iso

_SCHEMA = Path(__file__).with_name("schema.sql")


def _dt(value: str | None) -> datetime | None:
    return from_utc_iso(value) if value else None


@dataclass
class Goal:
    id: int
    date: str
    text: str
    status: str
    set_at: datetime
    reviewed_at: datetime | None
    notes: str | None


@dataclass
class CheckIn:
    id: int
    kind: str
    local_date: str
    scheduled_for: datetime
    fired_at: datetime | None
    outcome: str
    attempts: int


@dataclass
class Action:
    id: int
    ts: datetime
    tool: str
    args: dict[str, Any]
    result: Any
    ok: bool
    duration_ms: int
    error: str | None


@dataclass
class Job:
    name: str
    schedule: str
    last_run_at: datetime | None
    next_run_at: datetime | None
    enabled: bool


@dataclass
class Message:
    role: str
    content: str
    ts: datetime


class Store:
    """Façade over the ZEUS database. All timestamps are aware UTC."""

    def __init__(self, db_path: Path, clock: Clock) -> None:
        self._clock = clock
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because the daemon writes to this Store
        # from TWO threads: the main thread (scheduled check-ins) and the
        # wake-word activation thread (_handle_activation ->
        # start_conversation). sqlite3's default binds the connection to its
        # creating thread and raises ProgrammingError anywhere else -- and
        # the daemon's activation loop catches Exception broadly, so the
        # error is swallowed and EVERY wake-word conversation dies leaving
        # nothing in the database. Verified: the wake-thread write raised
        # "SQLite objects created in a thread can only be used in that same
        # thread" and `select count(*) from conversations` stayed at 0.
        #
        # Turning the check off makes the connection usable across threads
        # but NOT safe on its own, hence _lock below: every method that
        # touches the connection takes it, so two threads cannot interleave.
        # This mirrors the lock Journal.append() already carries for the
        # exact same wiring -- the daemon hands one Journal AND one Store to
        # both threads.
        self.connection = sqlite3.connect(
            db_path, isolation_level=None, check_same_thread=False
        )
        self._lock = threading.Lock()
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self.connection.executescript(_SCHEMA.read_text())

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def _now(self) -> str:
        return to_utc_iso(self._clock.now_utc())

    # ---- goals -------------------------------------------------------
    def set_goal(self, date: str, text: str) -> int:
        # RETURNING, not cur.lastrowid: last_insert_rowid() is only updated by
        # a real insert, so on the ON CONFLICT DO UPDATE branch lastrowid
        # silently returns the id of whatever row was last inserted on this
        # connection — a caller that then calls update_goal(that_id, ...)
        # mutates an unrelated day's goal.
        row = self.connection.execute(
            "INSERT INTO goals (date, text, set_at) VALUES (?, ?, ?) "
            "ON CONFLICT(date) DO UPDATE SET text = excluded.text, "
            "set_at = excluded.set_at, status = 'pending', "
            "reviewed_at = NULL, notes = NULL "
            "RETURNING id",
            (date, text, self._now()),
        ).fetchone()
        return int(row["id"])

    def get_goal(self, date: str) -> Goal | None:
        row = self.connection.execute(
            "SELECT * FROM goals WHERE date = ?", (date,)
        ).fetchone()
        if row is None:
            return None
        return Goal(
            id=row["id"], date=row["date"], text=row["text"], status=row["status"],
            set_at=from_utc_iso(row["set_at"]), reviewed_at=_dt(row["reviewed_at"]),
            notes=row["notes"],
        )

    def update_goal(self, goal_id: int, status: str, notes: str | None = None) -> None:
        # COALESCE, matching update_checkin's fired_at below: omitting the
        # optional notes arg must not erase notes written by an earlier
        # review. Clearing notes is still reachable — set_goal's upsert
        # resets them to NULL.
        self.connection.execute(
            "UPDATE goals SET status = ?, notes = COALESCE(?, notes), "
            "reviewed_at = ? WHERE id = ?",
            (status, notes, self._now(), goal_id),
        )

    # ---- check-ins ---------------------------------------------------
    def open_checkin(
        self, kind: str, scheduled_for: datetime, local_date: str
    ) -> int:
        """Open a check-in row.

        local_date is a PARAMETER, not derived from scheduled_for. An earlier
        version inferred it via scheduled_for.date(), relying on the caller to
        hand over a local-zone-aware datetime — but the scheduler always
        produces UTC (cron.next_occurrence ends in .astimezone(utc)), so the
        row was written with the UTC date while find_open_checkin searched the
        local one. Every retry then opened a fresh row, attempts never advanced
        past 1, both retry budgets never exhausted, and fold_forward/skipped
        never fired. Africa/Lagos hid it — at UTC+1 the two dates coincide at
        check-in times. Keeping the key explicit makes the write and the lookup
        provably the same value, and keeps Store timezone-free.
        """
        cur = self.connection.execute(
            "INSERT INTO checkins (kind, local_date, scheduled_for) VALUES (?, ?, ?)",
            (kind, local_date, to_utc_iso(scheduled_for)),
        )
        return int(cur.lastrowid)

    def get_checkin(self, checkin_id: int) -> CheckIn:
        row = self.connection.execute(
            "SELECT * FROM checkins WHERE id = ?", (checkin_id,)
        ).fetchone()
        return CheckIn(
            id=row["id"], kind=row["kind"], local_date=row["local_date"],
            scheduled_for=from_utc_iso(row["scheduled_for"]),
            fired_at=_dt(row["fired_at"]), outcome=row["outcome"],
            attempts=row["attempts"],
        )

    def find_open_checkin(self, kind: str, date: str) -> CheckIn | None:
        """The unresolved check-in of this kind on this local date, if any.

        "Open" means not yet terminal: 'answered' and 'skipped' are settled
        outcomes, while 'deferred' and 'no_answer' are both still eligible
        for a retry and so still count as open.

        Matching on local_date rather than date(scheduled_for) is load-bearing:
        scheduled_for is stored as UTC, and evening 21:00 at UTC-5 lands on
        the *next* UTC day, so a UTC-date match would miss it.
        """
        row = self.connection.execute(
            "SELECT * FROM checkins WHERE kind = ? AND local_date = ? "
            "AND outcome NOT IN ('answered','skipped') "
            "ORDER BY id DESC LIMIT 1",
            (kind, date),
        ).fetchone()
        if row is None:
            return None
        return CheckIn(
            id=row["id"], kind=row["kind"], local_date=row["local_date"],
            scheduled_for=from_utc_iso(row["scheduled_for"]),
            fired_at=_dt(row["fired_at"]), outcome=row["outcome"],
            attempts=row["attempts"],
        )

    def update_checkin(
        self, checkin_id: int, *, outcome: str, attempts: int,
        fired_at: datetime | None = None,
    ) -> None:
        self.connection.execute(
            "UPDATE checkins SET outcome = ?, attempts = ?, "
            "fired_at = COALESCE(?, fired_at) WHERE id = ?",
            (outcome, attempts, to_utc_iso(fired_at) if fired_at else None, checkin_id),
        )

    # ---- actions -----------------------------------------------------
    def log_action(
        self, tool: str, args: dict[str, Any], result: Any, ok: bool,
        duration_ms: int, error: str | None = None,
        conversation_id: int | None = None,
    ) -> int:
        cur = self.connection.execute(
            "INSERT INTO actions (ts, conversation_id, tool, args_json, "
            "result_json, ok, duration_ms, error) VALUES (?,?,?,?,?,?,?,?)",
            (self._now(), conversation_id, tool, json.dumps(args),
             json.dumps(result) if result is not None else None,
             int(ok), duration_ms, error),
        )
        return int(cur.lastrowid)

    def recent_actions(self, limit: int = 50) -> list[Action]:
        rows = self.connection.execute(
            "SELECT * FROM actions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            Action(
                id=r["id"], ts=from_utc_iso(r["ts"]), tool=r["tool"],
                args=json.loads(r["args_json"]),
                result=json.loads(r["result_json"]) if r["result_json"] else None,
                ok=bool(r["ok"]), duration_ms=r["duration_ms"], error=r["error"],
            )
            for r in rows
        ]

    # ---- conversations ----------------------------------------------
    def start_conversation(self, trigger: str) -> int:
        cur = self.connection.execute(
            "INSERT INTO conversations (started_at, trigger) VALUES (?, ?)",
            (self._now(), trigger),
        )
        return int(cur.lastrowid)

    def add_message(self, conversation_id: int, role: str, content: str) -> None:
        self.connection.execute(
            "INSERT INTO messages (conversation_id, role, content, ts) "
            "VALUES (?,?,?,?)",
            (conversation_id, role, content, self._now()),
        )

    def messages(self, conversation_id: int) -> list[Message]:
        rows = self.connection.execute(
            "SELECT role, content, ts FROM messages WHERE conversation_id = ? "
            "ORDER BY id", (conversation_id,)
        ).fetchall()
        return [Message(r["role"], r["content"], from_utc_iso(r["ts"])) for r in rows]

    def end_conversation(self, conversation_id: int) -> None:
        self.connection.execute(
            "UPDATE conversations SET ended_at = ? WHERE id = ?",
            (self._now(), conversation_id),
        )

    # ---- facts -------------------------------------------------------
    def set_fact(self, key: str, value: str, source: str) -> None:
        self.connection.execute(
            "INSERT INTO facts (key, value, learned_at, source) VALUES (?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "learned_at = excluded.learned_at, source = excluded.source",
            (key, value, self._now(), source),
        )

    def get_fact(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM facts WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    # ---- jobs and heartbeat ------------------------------------------
    def upsert_job(self, name: str, schedule: str) -> None:
        self.connection.execute(
            "INSERT INTO jobs (name, schedule) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET schedule = excluded.schedule",
            (name, schedule),
        )

    def jobs(self) -> list[Job]:
        rows = self.connection.execute(
            "SELECT * FROM jobs WHERE enabled = 1 ORDER BY name"
        ).fetchall()
        return [
            Job(r["name"], r["schedule"], _dt(r["last_run_at"]),
                _dt(r["next_run_at"]), bool(r["enabled"]))
            for r in rows
        ]

    def set_job_run(self, name: str, last_run_at: datetime) -> None:
        self.connection.execute(
            "UPDATE jobs SET last_run_at = ? WHERE name = ?",
            (to_utc_iso(last_run_at), name),
        )

    def set_heartbeat(self) -> None:
        self.connection.execute(
            "INSERT INTO heartbeat (id, ts) VALUES (1, ?) "
            "ON CONFLICT(id) DO UPDATE SET ts = excluded.ts",
            (self._now(),),
        )

    def heartbeat(self) -> datetime | None:
        row = self.connection.execute("SELECT ts FROM heartbeat WHERE id = 1").fetchone()
        return from_utc_iso(row["ts"]) if row else None
```

Also create `src/zeus/memory/__init__.py` (empty) and `tests/memory/__init__.py` is not needed (pytest rootdir handles it).

Add to `pyproject.toml` under `[tool.hatch.build.targets.wheel]` so the SQL ships:
```toml
[tool.hatch.build]
include = ["src/zeus/**/*.py", "src/zeus/**/*.sql"]
```

- [ ] **Step 5: Run the test and verify it passes**

Run: `.venv/bin/pytest tests/memory/test_store.py -v`
Expected: PASS — 15 tests (10 above, plus 5 added in review: the interleaved
`set_goal` upsert-id test, three `find_open_checkin` tests including the
local-vs-UTC date test, and the `update_goal` notes-preservation test.)

- [ ] **Step 6: Commit**

```bash
git add src/zeus/memory/ tests/memory/ pyproject.toml
git commit -m "feat: SQLite store with WAL, UTC discipline, and action log"
```

---

### Task 4: Markdown journal

**Files:**
- Create: `src/zeus/memory/journal.py`
- Test: `tests/memory/test_journal.py`

**Interfaces:**
- Consumes: `zeus.clock.Clock`, `zeus.clock.resolve_timezone`.
- Produces `Journal(directory: Path, clock: Clock, tz: ZoneInfo)` with:
  - `append(line: str) -> None` — writes a `- HH:MM — <line>` entry to today's file, creating it with a `# YYYY-MM-DD` header on first write.
  - `path_for(date: str) -> Path`
  - `read(date: str) -> str` — returns `""` when the day has no file.

- [ ] **Step 1: Write the failing test**

`tests/memory/test_journal.py`:
```python
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from zeus.clock import FakeClock
from zeus.memory.journal import Journal

LAGOS = ZoneInfo("Africa/Lagos")
# 10:00 UTC == 11:00 Lagos (UTC+1)
START = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)


def test_creates_file_with_header(tmp_path):
    journal = Journal(tmp_path, FakeClock(START), LAGOS)
    journal.append("Goal set: Finish the auth flow")
    assert journal.path_for("2026-08-05").exists()
    assert journal.read("2026-08-05").startswith("# 2026-08-05\n")


def test_entry_uses_local_time_not_utc(tmp_path):
    journal = Journal(tmp_path, FakeClock(START), LAGOS)
    journal.append("Goal set")
    assert "- 11:00 — Goal set" in journal.read("2026-08-05")


def test_appends_without_duplicating_header(tmp_path):
    clock = FakeClock(START)
    journal = Journal(tmp_path, clock, LAGOS)
    journal.append("First")
    clock.sleep(3600)
    journal.append("Second")
    text = journal.read("2026-08-05")
    assert text.count("# 2026-08-05") == 1
    assert "- 11:00 — First" in text
    assert "- 12:00 — Second" in text


def test_rolls_over_to_a_new_file_next_day(tmp_path):
    clock = FakeClock(START)
    journal = Journal(tmp_path, clock, LAGOS)
    journal.append("Day one")
    clock.sleep(24 * 3600)
    journal.append("Day two")
    assert "Day one" in journal.read("2026-08-05")
    assert "Day two" in journal.read("2026-08-06")


def test_read_missing_day_is_empty(tmp_path):
    journal = Journal(tmp_path, FakeClock(START), LAGOS)
    assert journal.read("1999-01-01") == ""
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv/bin/pytest tests/memory/test_journal.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.memory.journal'`

- [ ] **Step 3: Implement `src/zeus/memory/journal.py`**

```python
"""Human-readable daily journal. See spec §6."""
from __future__ import annotations

import threading
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
        # append() is check-then-act: it tests for the file, writes a header
        # if absent, then reopens for append. Two threads racing the FIRST
        # write of a day can both see "absent", and the second write_text
        # (mode "w") truncates the first's entry. T16's build_daemon wires
        # the scheduler thread and the wake-word activation thread to the
        # SAME Journal, so this is reachable, not theoretical — measured at
        # 10 lost entries across 40 trials of 8 concurrent writers, with no
        # error raised. The line simply is not there.
        self._lock = threading.Lock()

    def _local_now(self):
        return self._clock.now_utc().astimezone(self._tz)

    def path_for(self, date: str) -> Path:
        return self._dir / f"{date}.md"

    def append(self, line: str) -> None:
        with self._lock:
            now = self._local_now()
            date = now.strftime("%Y-%m-%d")
            path = self.path_for(date)
            if not path.exists():
                path.write_text(f"# {date}\n\n", encoding="utf-8")
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"- {now.strftime('%H:%M')} — {line}\n")

    def read(self, date: str) -> str:
        path = self.path_for(date)
        return path.read_text() if path.exists() else ""
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `.venv/bin/pytest tests/memory/test_journal.py -v`
Expected: PASS — 6 tests (5 above, plus
`test_concurrent_appends_do_not_lose_entries`, added in review. append() is
check-then-act and T16's build_daemon gives the scheduler thread and the
wake-word thread the SAME Journal: measured 10 lost entries across 40
trials of 8 concurrent writers, silently — no error, the line is just
absent.)

- [ ] **Step 5: Commit**

```bash
git add src/zeus/memory/journal.py tests/memory/test_journal.py
git commit -m "feat: markdown daily journal in local time"
```

---

### Task 5: Cron expressions and occurrence calculation

**Files:**
- Create: `src/zeus/schedule/__init__.py`, `src/zeus/schedule/cron.py`
- Test: `tests/schedule/test_cron.py`

**Interfaces:**
- Consumes: `croniter`, `zoneinfo`.
- Produces:
  - `hhmm_to_cron(hhmm: str) -> str` — `"11:00"` → `"0 11 * * *"`; raises `ValueError` on malformed input.
  - `next_occurrence(expression: str, after_utc: datetime, tz: ZoneInfo) -> datetime` — returns aware UTC.
  - `occurrences_between(expression: str, start_utc: datetime, end_utc: datetime, tz: ZoneInfo) -> list[datetime]` — exclusive of `start`, inclusive of `end`, aware UTC.

- [ ] **Step 1: Write the failing test**

`tests/schedule/test_cron.py`:
```python
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from zeus.schedule.cron import hhmm_to_cron, next_occurrence, occurrences_between

LAGOS = ZoneInfo("Africa/Lagos")       # UTC+1, no DST — the user's real zone
NEW_YORK = ZoneInfo("America/New_York")  # has DST — used to prove DST handling


def test_hhmm_to_cron():
    assert hhmm_to_cron("11:00") == "0 11 * * *"
    assert hhmm_to_cron("09:30") == "30 9 * * *"


@pytest.mark.parametrize("bad", ["", "25:00", "11", "11:60", "abc"])
def test_hhmm_to_cron_rejects_garbage(bad):
    with pytest.raises(ValueError):
        hhmm_to_cron(bad)


def test_next_occurrence_in_local_zone():
    # 08:00 UTC == 09:00 Lagos; next 11:00 Lagos is 10:00 UTC same day
    after = datetime(2026, 8, 5, 8, 0, tzinfo=timezone.utc)
    result = next_occurrence("0 11 * * *", after, LAGOS)
    assert result == datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)


def test_next_occurrence_rolls_to_tomorrow():
    after = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)  # 13:00 Lagos
    result = next_occurrence("0 11 * * *", after, LAGOS)
    assert result == datetime(2026, 8, 6, 10, 0, tzinfo=timezone.utc)


def test_occurrences_between_is_exclusive_of_start():
    start = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)   # exactly 11:00 Lagos
    end = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)
    found = occurrences_between("0 11 * * *", start, end, LAGOS)
    assert found == [
        datetime(2026, 8, 6, 10, 0, tzinfo=timezone.utc),
        datetime(2026, 8, 7, 10, 0, tzinfo=timezone.utc),
    ]


def test_occurrences_between_empty_when_none_due():
    start = datetime(2026, 8, 5, 10, 30, tzinfo=timezone.utc)
    end = datetime(2026, 8, 5, 11, 0, tzinfo=timezone.utc)
    assert occurrences_between("0 11 * * *", start, end, LAGOS) == []


def test_interval_expression_for_watchman_cadence():
    start = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)
    end = datetime(2026, 8, 5, 11, 0, tzinfo=timezone.utc)
    found = occurrences_between("*/30 * * * *", start, end, LAGOS)
    assert len(found) == 2  # 10:30 and 11:00 UTC


def test_local_wall_clock_survives_dst_transition():
    """11:00 must stay 11:00 local across the US spring-forward.

    Pinned to America/New_York deliberately: the machine's own zone
    (Africa/Lagos) has no DST, so testing against it would prove nothing.
    """
    # 2026-03-08 is the US spring-forward date.
    before = datetime(2026, 3, 7, 12, 0, tzinfo=timezone.utc)  # 07:00 EST
    first = next_occurrence("0 11 * * *", before, NEW_YORK)
    second = next_occurrence("0 11 * * *", first, NEW_YORK)

    assert first.astimezone(NEW_YORK).hour == 11
    assert second.astimezone(NEW_YORK).hour == 11
    # UTC offset shifts by an hour across the boundary — proving DST applied
    assert first.hour == 16   # EST  = UTC-5
    assert second.hour == 15  # EDT  = UTC-4
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv/bin/pytest tests/schedule/test_cron.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.schedule'`

- [ ] **Step 3: Implement `src/zeus/schedule/cron.py`**

```python
"""Cron expression handling. See spec §9.1.

Cron is used rather than a simple 'daily at HH:MM' format because later
slices need interval cadences such as '*/30 * * * *' that a daily format
cannot express.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from croniter import croniter

_HHMM = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def hhmm_to_cron(hhmm: str) -> str:
    """Convert the user-facing 'HH:MM' config value to a cron expression."""
    match = _HHMM.match(hhmm)
    if not match:
        raise ValueError(f"invalid time of day: {hhmm!r} (expected 'HH:MM')")
    hour, minute = match.groups()
    return f"{int(minute)} {int(hour)} * * *"


def next_occurrence(expression: str, after_utc: datetime, tz: ZoneInfo) -> datetime:
    """Next firing strictly after `after_utc`, evaluated in local wall-clock."""
    local = after_utc.astimezone(tz)
    nxt = croniter(expression, local).get_next(datetime)
    return nxt.astimezone(timezone.utc)


def occurrences_between(
    expression: str, start_utc: datetime, end_utc: datetime, tz: ZoneInfo
) -> list[datetime]:
    """All firings in (start_utc, end_utc], evaluated in local wall-clock."""
    found: list[datetime] = []
    cursor = start_utc
    while True:
        cursor = next_occurrence(expression, cursor, tz)
        if cursor > end_utc:
            return found
        found.append(cursor)
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `.venv/bin/pytest tests/schedule/test_cron.py -v`
Expected: PASS — 12 tests

- [ ] **Step 5: Commit**

```bash
git add src/zeus/schedule/ tests/schedule/
git commit -m "feat: cron occurrence calculation in local wall-clock time"
```

---

### Task 6: Scheduler with startup catch-up

**Files:**
- Create: `src/zeus/schedule/scheduler.py`
- Test: `tests/schedule/test_scheduler.py`

**Interfaces:**
- Consumes: `Store`, `Clock`, `zeus.schedule.cron`.
- Produces:
  - `MissedRun(job: str, scheduled_for: datetime, same_local_day: bool)`
  - `Scheduler(store, clock, tz)` with:
    - `register(name: str, schedule: str, handler: Callable[[datetime], None]) -> None`
    - `catch_up() -> list[MissedRun]` — occurrences between the heartbeat and now, each tagged with whether it falls on today's local date (spec §9.2).
    - `run_pending(now_utc: datetime) -> list[str]` — fires handlers whose occurrence has arrived since `last_run_at`; returns names fired.
    - `seconds_until_next(now_utc: datetime) -> float` — for the daemon's sleep, capped at 60 s so config reloads and clock jumps are noticed promptly.

**§9.2 decision rules are encoded by `catch_up` returning `same_local_day`; the *policy* (fire vs. skip) lives in Task 15 so the scheduler stays check-in agnostic and reusable by the Slice 4 watchman.**

- [ ] **Step 1: Write the failing test**

`tests/schedule/test_scheduler.py`:
```python
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from zeus.clock import FakeClock
from zeus.memory.store import Store
from zeus.schedule.scheduler import Scheduler

LAGOS = ZoneInfo("Africa/Lagos")
NOW = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)  # 13:00 Lagos


@pytest.fixture
def scheduler(tmp_path):
    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    return Scheduler(store, clock, LAGOS), store, clock


def test_register_persists_the_job(scheduler):
    sched, store, _ = scheduler
    sched.register("checkin_morning", "0 11 * * *", lambda when: None)
    assert [j.name for j in store.jobs()] == ["checkin_morning"]


def test_catch_up_returns_nothing_without_a_heartbeat(scheduler):
    sched, _, _ = scheduler
    sched.register("checkin_morning", "0 11 * * *", lambda when: None)
    assert sched.catch_up() == []


def test_catch_up_flags_a_missed_run_on_the_same_local_day(scheduler):
    sched, store, clock = scheduler
    sched.register("checkin_morning", "0 11 * * *", lambda when: None)
    # Heartbeat at 09:00 Lagos today; now is 13:00 Lagos. 11:00 was missed.
    clock.advance(timedelta(hours=-4))
    store.set_heartbeat()
    clock.advance(timedelta(hours=4))

    missed = sched.catch_up()
    assert len(missed) == 1
    assert missed[0].job == "checkin_morning"
    assert missed[0].same_local_day is True


def test_catch_up_flags_a_stale_run_from_a_previous_day(scheduler):
    sched, store, clock = scheduler
    sched.register("checkin_morning", "0 11 * * *", lambda when: None)
    clock.advance(timedelta(days=-2))
    store.set_heartbeat()
    clock.advance(timedelta(days=2))

    missed = sched.catch_up()
    assert len(missed) == 2                      # two 11:00s elapsed
    assert missed[0].same_local_day is False     # the older one
    assert missed[-1].same_local_day is True     # today's


def test_run_pending_fires_the_handler_once(scheduler):
    sched, _, _ = scheduler
    fired: list[datetime] = []
    sched.register("checkin_morning", "0 11 * * *", fired.append)

    # First call establishes the baseline; nothing has come due yet.
    assert sched.run_pending(NOW) == []
    # Advance past tomorrow's 11:00 Lagos (== 10:00 UTC).
    later = datetime(2026, 8, 6, 10, 30, tzinfo=timezone.utc)
    assert sched.run_pending(later) == ["checkin_morning"]
    assert len(fired) == 1
    # Firing is not repeated for the same occurrence.
    assert sched.run_pending(later) == []


def test_seconds_until_next_is_capped_at_sixty(scheduler):
    sched, _, _ = scheduler
    sched.register("checkin_morning", "0 11 * * *", lambda when: None)
    assert sched.seconds_until_next(NOW) == 60.0


def test_seconds_until_next_when_no_jobs(scheduler):
    sched, _, _ = scheduler
    assert sched.seconds_until_next(NOW) == 60.0
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv/bin/pytest tests/schedule/test_scheduler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.schedule.scheduler'`

- [ ] **Step 3: Implement `src/zeus/schedule/scheduler.py`**

```python
"""Generic recurring-job scheduler with startup catch-up. See spec §9."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from zeus.clock import Clock
from zeus.memory.store import Store
from zeus.schedule.cron import next_occurrence, occurrences_between

Handler = Callable[[datetime], None]

_MAX_SLEEP = 60.0
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MissedRun:
    job: str
    scheduled_for: datetime
    same_local_day: bool


class Scheduler:
    """Fires named jobs on cron schedules evaluated in local wall-clock time.

    Deliberately knows nothing about check-ins: Slice 4's watchman registers
    its own jobs here without modification. Policy for what to do about a
    missed run lives with the caller.
    """

    def __init__(self, store: Store, clock: Clock, tz: ZoneInfo) -> None:
        self._store = store
        self._clock = clock
        self._tz = tz
        self._handlers: dict[str, Handler] = {}
        self._schedules: dict[str, str] = {}

    def register(self, name: str, schedule: str, handler: Handler) -> None:
        self._handlers[name] = handler
        self._schedules[name] = schedule
        self._store.upsert_job(name, schedule)

    def catch_up(self) -> list[MissedRun]:
        """Occurrences that elapsed between the last heartbeat and now."""
        since = self._store.heartbeat()
        if since is None:
            return []
        now = self._clock.now_utc()
        today = now.astimezone(self._tz).date()
        missed: list[MissedRun] = []
        for name, expression in self._schedules.items():
            for when in occurrences_between(expression, since, now, self._tz):
                missed.append(
                    MissedRun(
                        job=name,
                        scheduled_for=when,
                        same_local_day=when.astimezone(self._tz).date() == today,
                    )
                )
        missed.sort(key=lambda run: run.scheduled_for)
        return missed

    def run_pending(self, now_utc: datetime) -> list[str]:
        """Fire any job whose next occurrence has arrived."""
        fired: list[str] = []
        by_name = {job.name: job for job in self._store.jobs()}
        for name, expression in self._schedules.items():
            job = by_name.get(name)
            baseline = job.last_run_at if job and job.last_run_at else None
            if baseline is None:
                # First sight of this job: set the baseline, do not fire.
                self._store.set_job_run(name, now_utc)
                continue
            due = next_occurrence(expression, baseline, self._tz)
            if due <= now_utc:
                # set_job_run BEFORE the handler, deliberately: the occurrence
                # is consumed whether or not the handler succeeds. Persisting
                # after would make a permanently-failing job re-fire on every
                # poll. Retry policy for check-ins lives in the Task 8 state
                # machine, not here.
                self._store.set_job_run(name, due)
                try:
                    self._handlers[name](due)
                except Exception:
                    # One job must never abort the sweep for the jobs after it.
                    _log.exception("scheduled job %r failed at %s", name, due)
                fired.append(name)
        return fired

    def seconds_until_next(self, now_utc: datetime) -> float:
        """Sleep interval, capped so clock jumps are noticed promptly."""
        if not self._schedules:
            return _MAX_SLEEP
        soonest = min(
            next_occurrence(expression, now_utc, self._tz)
            for expression in self._schedules.values()
        )
        delta = (soonest - now_utc).total_seconds()
        return max(0.0, min(delta, _MAX_SLEEP))
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `.venv/bin/pytest tests/schedule/test_scheduler.py -v`
Expected: PASS — 10 tests (7 above, plus 3 added in review: a raising handler
must not abort the sweep; `seconds_until_next` must return a real sub-cap
interval, not just the 60 s cap; and `same_local_day` must be pinned at a
timestamp where the local and UTC dates genuinely diverge — 00:30 Lagos is
23:30 UTC the previous day.)

- [ ] **Step 5: Commit**

```bash
git add src/zeus/schedule/scheduler.py tests/schedule/test_scheduler.py
git commit -m "feat: cron scheduler with startup catch-up detection"
```

---

### Task 7: Presence and the context gate

**Files:**
- Create: `src/zeus/context/__init__.py`, `src/zeus/context/presence.py`
- Test: `tests/context/test_presence.py`

**Interfaces:**
- Consumes: `ContextConfig`.
- Produces:
  - `Verdict` enum: `SPEAK`, `NOTIFY`, `DEFER`.
  - `Signals(screen_locked: bool, idle: timedelta, focus_active: bool, call_app_running: bool)`
  - `decide(signals: Signals, idle_threshold: timedelta) -> Verdict` — **pure**, the whole of spec §8's decision table.
  - `Presence(config: ContextConfig)` with `read() -> Signals` and `verdict() -> Verdict`.
  - Probe functions, each independently monkeypatchable: `screen_locked()`, `idle_time()`, `focus_active()`, `call_app_running(names)`.

**Design note: `decide` is separated from `read` precisely so the decision table is testable without any macOS call.**

- [ ] **Step 1: Write the failing test**

`tests/context/test_presence.py`:
```python
from datetime import timedelta

import pytest

from zeus.config import ContextConfig
from zeus.context.presence import Presence, Signals, Verdict, decide

THRESHOLD = timedelta(minutes=15)


def sig(**kwargs) -> Signals:
    base = dict(
        screen_locked=False,
        idle=timedelta(0),
        focus_active=False,
        call_app_running=False,
    )
    base.update(kwargs)
    return Signals(**base)


@pytest.mark.parametrize(
    "signals,expected",
    [
        # DEFER wins over everything
        (sig(screen_locked=True), Verdict.DEFER),
        (sig(idle=timedelta(minutes=16)), Verdict.DEFER),
        (sig(screen_locked=True, focus_active=True), Verdict.DEFER),
        (sig(idle=timedelta(minutes=20), call_app_running=True), Verdict.DEFER),
        # NOTIFY when busy but present
        (sig(focus_active=True), Verdict.NOTIFY),
        (sig(call_app_running=True), Verdict.NOTIFY),
        (sig(focus_active=True, call_app_running=True), Verdict.NOTIFY),
        # SPEAK only when clearly free
        (sig(), Verdict.SPEAK),
        (sig(idle=timedelta(minutes=14, seconds=59)), Verdict.SPEAK),
    ],
)
def test_decision_table(signals, expected):
    assert decide(signals, THRESHOLD) == expected


def test_idle_threshold_boundary_is_exclusive():
    assert decide(sig(idle=THRESHOLD), THRESHOLD) == Verdict.SPEAK
    assert decide(sig(idle=THRESHOLD + timedelta(seconds=1)), THRESHOLD) == Verdict.DEFER


def test_presence_reads_all_four_probes(monkeypatch):
    import zeus.context.presence as mod

    monkeypatch.setattr(mod, "screen_locked", lambda: True)
    monkeypatch.setattr(mod, "idle_time", lambda: timedelta(seconds=5))
    monkeypatch.setattr(mod, "focus_active", lambda: False)
    monkeypatch.setattr(mod, "call_app_running", lambda names: False)

    presence = Presence(ContextConfig())
    signals = presence.read()
    assert signals.screen_locked is True
    assert signals.idle == timedelta(seconds=5)
    assert presence.verdict() == Verdict.DEFER


def test_probe_failure_degrades_to_safe_default(monkeypatch):
    """A broken probe must not crash the daemon; it reports the safe value."""
    import zeus.context.presence as mod

    def boom():
        raise OSError("ioreg exploded")

    monkeypatch.setattr(mod, "_raw_idle_nanoseconds", boom)
    assert mod.idle_time() == timedelta(0)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv/bin/pytest tests/context/test_presence.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.context'`

- [ ] **Step 3: Implement `src/zeus/context/presence.py`**

```python
"""Context gate. See spec §8.

Every probe degrades to the *safe* value on failure (not locked, not idle,
no focus, no call) so that a broken probe never silences ZEUS permanently
and never crashes the daemon.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path

from zeus.config import ContextConfig

log = logging.getLogger(__name__)

ASSERTIONS = Path.home() / "Library/DoNotDisturb/DB/Assertions.json"
_IDLE_RE = re.compile(r'"HIDIdleTime"\s*=\s*(\d+)')


class Verdict(Enum):
    SPEAK = "speak"
    NOTIFY = "notify"
    DEFER = "defer"


@dataclass(frozen=True)
class Signals:
    screen_locked: bool
    idle: timedelta
    focus_active: bool
    call_app_running: bool


def decide(signals: Signals, idle_threshold: timedelta) -> Verdict:
    """Pure implementation of the spec §8 decision table."""
    if signals.screen_locked or signals.idle > idle_threshold:
        return Verdict.DEFER
    if signals.focus_active or signals.call_app_running:
        return Verdict.NOTIFY
    return Verdict.SPEAK


def screen_locked() -> bool:
    try:
        import Quartz  # imported lazily: pyobjc is slow to load

        session = Quartz.CGSessionCopyCurrentDictionary()
        return bool(session and session.get("CGSSessionScreenIsLocked", False))
    except Exception:
        log.debug("screen_locked probe failed", exc_info=True)
        return False


def _raw_idle_nanoseconds() -> int:
    output = subprocess.run(
        ["ioreg", "-c", "IOHIDSystem"],
        capture_output=True, text=True, timeout=5, check=True,
    ).stdout
    match = _IDLE_RE.search(output)
    if not match:
        raise ValueError("HIDIdleTime not found in ioreg output")
    return int(match.group(1))


def idle_time() -> timedelta:
    try:
        return timedelta(seconds=_raw_idle_nanoseconds() / 1_000_000_000)
    except Exception:
        log.debug("idle_time probe failed", exc_info=True)
        return timedelta(0)


def focus_active() -> bool:
    """True when a macOS Focus mode is asserted.

    RISK R3: verified that Assertions.json is ABSENT with no Focus active.
    The converse — that it is present with records while Focus IS active —
    must be confirmed manually before this probe is trusted (see Task 7,
    Step 5).
    """
    try:
        if not ASSERTIONS.exists():
            return False
        data = json.loads(ASSERTIONS.read_text())
        records = data.get("data", [{}])[0].get("storeAssertionRecords", [])
        return bool(records)
    except Exception:
        log.debug("focus_active probe failed", exc_info=True)
        return False


def call_app_running(names: list[str]) -> bool:
    for name in names:
        try:
            found = subprocess.run(
                ["pgrep", "-x", name], capture_output=True, timeout=5
            )
            if found.returncode == 0:
                return True
        except Exception:
            log.debug("call_app probe failed for %s", name, exc_info=True)
    return False


class Presence:
    def __init__(self, config: ContextConfig) -> None:
        self._config = config

    def read(self) -> Signals:
        return Signals(
            screen_locked=screen_locked(),
            idle=idle_time(),
            focus_active=focus_active(),
            call_app_running=call_app_running(self._config.call_apps),
        )

    def verdict(self) -> Verdict:
        return decide(self.read(), self._config.idle_threshold)
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `.venv/bin/pytest tests/context/test_presence.py -v`
Expected: PASS — 12 tests (9 parametrized decision-table cases + the
threshold-boundary test + 2 Presence tests. An earlier draft of this plan
said 13; that was a miscount of the parametrize list, not a dropped case —
the table above covers every row of spec §8, both DEFER precedence rules,
and the exclusive `idle > threshold` boundary.)

- [ ] **Step 5: Manually resolve risk R3 — Focus detection**

This is the **first implementation task the spec flagged as unverified**. Do it now, not later.

```bash
# 1. Confirm no Focus is active and the file is absent:
ls ~/Library/DoNotDisturb/DB/Assertions.json   # expect: No such file

# 2. Turn ON a Focus mode (Control Centre → Focus → Do Not Disturb), then:
ls -l ~/Library/DoNotDisturb/DB/Assertions.json
cat  ~/Library/DoNotDisturb/DB/Assertions.json

# 3. With Focus still ON, confirm the probe agrees:
.venv/bin/python -c "from zeus.context.presence import focus_active; print(focus_active())"
# expect: True

# 4. Turn Focus OFF and re-run:
.venv/bin/python -c "from zeus.context.presence import focus_active; print(focus_active())"
# expect: False
```

**If step 3 prints `False`,** the JSON shape differs from the assumption. Print the file, adjust the parsing in `focus_active()` to match the real structure, and re-run. **If the file never appears,** delete the `focus_active` probe entirely, make it return `False` permanently with a comment citing R3, and note in the spec that the gate runs on lock + idle + call-app signals only — which the tests already prove works.

Record the outcome in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/zeus/context/ tests/context/
git commit -m "feat: context gate with lock, idle, Focus, and call-app probes

Resolves R3: Focus detection verified by toggling Do Not Disturb manually."
```

---

### Task 8: Check-in retry state machine

**Files:**
- Create: `src/zeus/ritual/__init__.py`, `src/zeus/ritual/retry.py`
- Test: `tests/ritual/test_retry.py`

**Interfaces:**
- Consumes: `ScheduleConfig`, `Verdict`.
- Produces:
  - `Outcome` enum: `ANSWERED`, `NO_ANSWER`, `DEFERRED`, `SKIPPED` (values match the DB `CHECK` constraint exactly).
  - `Decision(outcome: Outcome, retry_after: timedelta | None, fold_forward: bool)`
  - `next_step(kind: str, verdict: Verdict, answered: bool | None, attempts: int, config: ScheduleConfig) -> Decision`

This is **pure logic with no I/O** — the entirety of spec §9.3. It is the single most defect-prone part of the slice, so it is isolated and exhaustively tested.

- [ ] **Step 1: Write the failing test**

`tests/ritual/test_retry.py`:
```python
from datetime import timedelta

import pytest

from zeus.config import ScheduleConfig
from zeus.context.presence import Verdict
from zeus.ritual.retry import Decision, Outcome, next_step

CONFIG = ScheduleConfig()  # 20m/3 defer, 30m/1 no-answer


def test_speaking_and_getting_an_answer_ends_the_sequence():
    result = next_step("morning", Verdict.SPEAK, answered=True, attempts=0, config=CONFIG)
    assert result == Decision(Outcome.ANSWERED, None, False)


def test_defer_schedules_a_twenty_minute_retry():
    result = next_step("morning", Verdict.DEFER, answered=None, attempts=0, config=CONFIG)
    assert result == Decision(Outcome.DEFERRED, timedelta(minutes=20), False)


@pytest.mark.parametrize("attempts", [0, 1, 2])
def test_defer_retries_up_to_three_times(attempts):
    result = next_step("morning", Verdict.DEFER, answered=None, attempts=attempts, config=CONFIG)
    assert result.retry_after == timedelta(minutes=20)


def test_defer_exhaustion_folds_a_morning_checkin_forward():
    result = next_step("morning", Verdict.DEFER, answered=None, attempts=3, config=CONFIG)
    assert result == Decision(Outcome.DEFERRED, None, True)


def test_defer_exhaustion_skips_an_evening_checkin():
    result = next_step("evening", Verdict.DEFER, answered=None, attempts=3, config=CONFIG)
    assert result == Decision(Outcome.SKIPPED, None, False)


def test_no_answer_schedules_a_thirty_minute_retry():
    result = next_step("morning", Verdict.SPEAK, answered=False, attempts=0, config=CONFIG)
    assert result == Decision(Outcome.NO_ANSWER, timedelta(minutes=30), False)


def test_no_answer_retries_only_once_then_folds():
    result = next_step("morning", Verdict.SPEAK, answered=False, attempts=1, config=CONFIG)
    assert result == Decision(Outcome.NO_ANSWER, None, True)


def test_no_answer_exhaustion_skips_an_evening_checkin():
    result = next_step("evening", Verdict.SPEAK, answered=False, attempts=1, config=CONFIG)
    assert result == Decision(Outcome.SKIPPED, None, False)


def test_notify_is_treated_as_deferred_until_acknowledged():
    result = next_step("morning", Verdict.NOTIFY, answered=None, attempts=0, config=CONFIG)
    assert result == Decision(Outcome.DEFERRED, timedelta(minutes=20), False)


def test_notify_that_gets_answered_ends_the_sequence():
    result = next_step("evening", Verdict.NOTIFY, answered=True, attempts=2, config=CONFIG)
    assert result == Decision(Outcome.ANSWERED, None, False)


def test_outcome_values_match_the_database_constraint():
    assert {o.value for o in Outcome} == {"answered", "no_answer", "deferred", "skipped"}


def test_custom_config_is_honoured():
    config = ScheduleConfig(
        defer_retry_after=timedelta(minutes=5),
        max_defer_retries=1,
        no_answer_retry_after=timedelta(minutes=10),
        max_no_answer_retries=2,
    )
    assert next_step("morning", Verdict.DEFER, None, 0, config).retry_after == timedelta(minutes=5)
    assert next_step("morning", Verdict.DEFER, None, 1, config).retry_after is None
    assert next_step("morning", Verdict.SPEAK, False, 1, config).retry_after == timedelta(minutes=10)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv/bin/pytest tests/ritual/test_retry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.ritual'`

- [ ] **Step 3: Implement `src/zeus/ritual/retry.py`**

```python
"""Check-in retry state machine. See spec §9.3 — the authoritative source.

Two distinct retry paths with different causes, cadences, and limits:

  DEFER      user is away or the screen is locked   20 min × 3
  NO_ANSWER  ZEUS spoke into silence                30 min × 1

On exhaustion a morning check-in folds forward into the evening one; an
evening check-in is recorded as skipped, because there is nothing to fold
into.

Pure logic, no I/O — this is the most defect-prone rule set in the slice.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

from zeus.config import ScheduleConfig
from zeus.context.presence import Verdict


class Outcome(Enum):
    ANSWERED = "answered"
    NO_ANSWER = "no_answer"
    DEFERRED = "deferred"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    retry_after: timedelta | None
    fold_forward: bool


def _exhausted(kind: str, outcome: Outcome) -> Decision:
    """Morning folds into the evening check-in; evening has nowhere to go."""
    if kind == "morning":
        return Decision(outcome, None, True)
    return Decision(Outcome.SKIPPED, None, False)


def next_step(
    kind: str,
    verdict: Verdict,
    answered: bool | None,
    attempts: int,
    config: ScheduleConfig,
) -> Decision:
    """Decide what happens after one check-in attempt.

    `answered` is None when ZEUS never spoke (DEFER, or a NOTIFY that has
    not yet been acknowledged), True when the user replied, False when the
    listen window elapsed in silence.
    """
    if answered:
        return Decision(Outcome.ANSWERED, None, False)

    if verdict in (Verdict.DEFER, Verdict.NOTIFY) and answered is None:
        if attempts + 1 > config.max_defer_retries:
            return _exhausted(kind, Outcome.DEFERRED)
        return Decision(Outcome.DEFERRED, config.defer_retry_after, False)

    # ZEUS spoke and heard nothing back.
    if attempts + 1 > config.max_no_answer_retries:
        return _exhausted(kind, Outcome.NO_ANSWER)
    return Decision(Outcome.NO_ANSWER, config.no_answer_retry_after, False)
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `.venv/bin/pytest tests/ritual/test_retry.py -v`
Expected: PASS — 14 tests (11 plain tests + `test_defer_retries_up_to_three_times`
parametrized over 3 attempt counts. An earlier draft said 15; that was a
miscount of the parametrize expansion, not a dropped case — the file covers
both retry paths, both exhaustion branches for morning and evening, both
NOTIFY branches, the DB-constraint match, and a custom-config case.)

- [ ] **Step 5: Commit**

```bash
git add src/zeus/ritual/ tests/ritual/
git commit -m "feat: check-in retry state machine (spec §9.3)"
```

---

### Task 9: Text-to-speech

**Files:**
- Create: `src/zeus/tts/__init__.py`, `src/zeus/tts/base.py`, `src/zeus/tts/mac_say.py`, `src/zeus/tts/fake.py`
- Test: `tests/tts/test_mac_say.py`, `tests/tts/test_fake.py`

**Interfaces:**
- Consumes: `TtsConfig`.
- Produces:
  - `Speaker` protocol: `say(text: str) -> None` (blocks until playback finishes), `stop() -> None`.
  - `MacSay(voice: str)` — shells out to `/usr/bin/say`.
  - `FakeSpeaker()` — `said: list[str]`, `stopped: int`; used by every downstream test.
  - `build_speaker(config: TtsConfig) -> Speaker` — factory; raises `ValueError` on an unknown provider.

- [ ] **Step 1: Write the failing tests**

`tests/tts/test_fake.py`:
```python
from zeus.tts.fake import FakeSpeaker


def test_records_everything_said():
    speaker = FakeSpeaker()
    speaker.say("Morning.")
    speaker.say("What's the goal?")
    assert speaker.said == ["Morning.", "What's the goal?"]


def test_records_stop_calls():
    speaker = FakeSpeaker()
    speaker.stop()
    assert speaker.stopped == 1
```

`tests/tts/test_mac_say.py`:
```python
import pytest

from zeus.config import TtsConfig
from zeus.tts.mac_say import MacSay
from zeus.tts import build_speaker


def test_invokes_say_with_the_configured_voice(monkeypatch):
    calls = []

    class DummyProcess:
        returncode = 0

        def wait(self, timeout=None):  # say() passes _MAX_SPEECH_SECONDS
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

        def wait(self, timeout=None):  # say() passes _MAX_SPEECH_SECONDS
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


def test_factory_builds_mac_say():
    speaker = build_speaker(TtsConfig(provider="mac_say", voice="Alex"))
    assert isinstance(speaker, MacSay)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown TTS provider"):
        build_speaker(TtsConfig(provider="elevenlabs"))
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/bin/pytest tests/tts/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.tts'`

- [ ] **Step 3: Implement the three files**

`src/zeus/tts/base.py`:
```python
"""Speaker protocol. See spec §5.1."""
from __future__ import annotations

from typing import Protocol


class Speaker(Protocol):
    def say(self, text: str) -> None:
        """Speak `text`, blocking until playback finishes."""
        ...

    def stop(self) -> None:
        """Interrupt any playback in progress."""
        ...
```

`src/zeus/tts/mac_say.py`:
```python
"""macOS `say` speaker. Default provider for Slice 1 (spec D2)."""
from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)

SAY = "/usr/bin/say"

# A safety valve, not a deadline: say() is meant to block for as long as the
# speech takes. 120s is ~350 spoken words at say's default rate, far beyond any
# single ZEUS utterance, so this only fires when the audio subsystem has wedged.
# Without it a stuck `say` would hang the ritual thread forever.
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
            # Must precede the broad handler: TimeoutExpired subclasses
            # Exception, so catching it below would swallow the hang and leave
            # the process alive. kill(), not terminate() — a process that has
            # ignored us for two minutes has earned it.
            log.error("TTS timed out after %ss, killing", _MAX_SPEECH_SECONDS)
            try:
                self._process.kill()
            except Exception:
                log.debug("failed to kill wedged say", exc_info=True)
        except Exception:
            log.error("TTS failed for %r", text[:60], exc_info=True)
        # NOTE: no `finally: self._process = None`. An earlier draft had one,
        # which made stop() always see None and turned its terminate() path
        # into dead code — the plan's own test_stop_terminates_the_running_
        # process failed against the plan's own implementation. stop() already
        # guards on `returncode is None`, so keeping the handle is safe.

    def stop(self) -> None:
        process = self._process
        if process is not None and process.returncode is None:
            try:
                process.terminate()
            except Exception:
                log.debug("failed to terminate say", exc_info=True)
```

`src/zeus/tts/fake.py`:
```python
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
```

`src/zeus/tts/__init__.py`:
```python
from zeus.config import TtsConfig
from zeus.tts.base import Speaker
from zeus.tts.fake import FakeSpeaker
from zeus.tts.mac_say import MacSay

__all__ = ["Speaker", "MacSay", "FakeSpeaker", "build_speaker"]


def build_speaker(config: TtsConfig) -> Speaker:
    if config.provider == "mac_say":
        return MacSay(voice=config.voice)
    raise ValueError(f"unknown TTS provider: {config.provider!r}")
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv/bin/pytest tests/tts/ -v`
Expected: PASS — 8 tests (7 above, plus `test_a_wedged_say_is_killed_not_
swallowed` added in review: `TimeoutExpired` subclasses `Exception`, so
without its own handler a hung `say` was swallowed by the broad one and the
process left running.)

- [ ] **Step 5: Commit**

```bash
git add src/zeus/tts/ tests/tts/
git commit -m "feat: Speaker protocol with macOS say and fake implementations"
```

---

### Task 10: Speech-to-text

**Files:**
- Create: `src/zeus/stt/__init__.py`, `src/zeus/stt/base.py`, `src/zeus/stt/local_whisper.py`, `src/zeus/stt/fake.py`
- Modify: `pyproject.toml` — add `addopts = ["--import-mode=importlib"]` under `[tool.pytest.ini_options]`. Required, not optional: `tests/stt/test_fake.py` shares a basename with T9's `tests/tts/test_fake.py`, and with no `__init__.py` in either directory the legacy `prepend` import mode makes the whole suite fail to collect. See Global Constraints.
- Test: `tests/stt/test_fake.py`, `tests/stt/test_local_whisper.py`

**Interfaces:**
- Consumes: `SttConfig`.
- Produces:
  - `Transcriber` protocol: `transcribe(pcm: bytes, sample_rate: int) -> str` — returns `""` when nothing intelligible was heard.
  - `LocalWhisper(model: str, compute: str, models_dir: Path)` — lazily loads `faster_whisper.WhisperModel` on first use so daemon startup is not blocked.
  - `FakeTranscriber(script: list[str])` — pops scripted strings; returns `""` once exhausted; records `calls: list[int]` (byte lengths received).
  - `build_transcriber(config: SttConfig, models_dir: Path) -> Transcriber`.

- [ ] **Step 1: Write the failing tests**

`tests/stt/test_fake.py`:
```python
from zeus.stt.fake import FakeTranscriber


def test_returns_scripted_strings_in_order():
    stt = FakeTranscriber(["Finish the auth flow", "Yes, mostly"])
    assert stt.transcribe(b"\x00" * 100, 16000) == "Finish the auth flow"
    assert stt.transcribe(b"\x00" * 200, 16000) == "Yes, mostly"


def test_returns_empty_once_exhausted():
    stt = FakeTranscriber(["only one"])
    stt.transcribe(b"", 16000)
    assert stt.transcribe(b"", 16000) == ""


def test_records_input_sizes():
    stt = FakeTranscriber(["a", "b"])
    stt.transcribe(b"\x00" * 320, 16000)
    stt.transcribe(b"\x00" * 640, 16000)
    assert stt.calls == [320, 640]
```

`tests/stt/test_local_whisper.py`:
```python
import numpy as np
import pytest

from zeus.config import SttConfig
from zeus.stt import build_transcriber
from zeus.stt.local_whisper import LocalWhisper, pcm_to_float32


def test_pcm_to_float32_normalises_to_unit_range():
    pcm = np.array([0, 32767, -32768], dtype=np.int16).tobytes()
    floats = pcm_to_float32(pcm)
    assert floats.dtype == np.float32
    assert floats[0] == pytest.approx(0.0)
    assert floats[1] == pytest.approx(1.0, abs=1e-4)
    assert floats[2] == pytest.approx(-1.0, abs=1e-4)


def test_empty_audio_short_circuits_without_loading_a_model(tmp_path):
    stt = LocalWhisper("base.en", "int8", tmp_path)
    assert stt.transcribe(b"", 16000) == ""
    assert stt._model is None  # never loaded


def test_transcribe_joins_segments(monkeypatch, tmp_path):
    class Segment:
        def __init__(self, text):
            self.text = text

    class DummyModel:
        def transcribe(self, audio, **kwargs):
            return [Segment(" Finish the"), Segment(" auth flow ")], None

    stt = LocalWhisper("base.en", "int8", tmp_path)
    monkeypatch.setattr(stt, "_load", lambda: DummyModel())
    pcm = np.zeros(16000, dtype=np.int16).tobytes()
    assert stt.transcribe(pcm, 16000) == "Finish the auth flow"


def test_model_failure_returns_empty_not_an_exception(monkeypatch, tmp_path):
    class Exploding:
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("ctranslate2 exploded")

    stt = LocalWhisper("base.en", "int8", tmp_path)
    monkeypatch.setattr(stt, "_load", lambda: Exploding())
    pcm = np.zeros(16000, dtype=np.int16).tobytes()
    assert stt.transcribe(pcm, 16000) == ""


def test_factory_rejects_unknown_provider(tmp_path):
    with pytest.raises(ValueError, match="unknown STT provider"):
        build_transcriber(SttConfig(provider="deepgram"), tmp_path)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/bin/pytest tests/stt/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.stt'`

- [ ] **Step 3: Implement the four files**

`src/zeus/stt/base.py`:
```python
"""Transcriber protocol. See spec §5.1."""
from __future__ import annotations

from typing import Protocol


class Transcriber(Protocol):
    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        """Transcribe 16-bit mono PCM. Returns '' if nothing intelligible."""
        ...
```

`src/zeus/stt/local_whisper.py`:
```python
"""Local faster-whisper transcription. Default provider for Slice 1 (spec D2).

Runs on CPU with int8 quantisation — the machine is an Intel Mac with no
GPU, so this trades a few seconds of latency for zero cost and audio that
never leaves the device.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def pcm_to_float32(pcm: bytes) -> np.ndarray:
    """Convert 16-bit signed PCM to the float32 [-1, 1] range Whisper wants."""
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


class LocalWhisper:
    def __init__(self, model: str, compute: str, models_dir: Path) -> None:
        self._model_name = model
        self._compute = compute
        self._models_dir = models_dir
        self._model = None

    def _load(self):
        from faster_whisper import WhisperModel

        self._models_dir.mkdir(parents=True, exist_ok=True)
        log.info("loading whisper model %s (%s)", self._model_name, self._compute)
        return WhisperModel(
            self._model_name,
            device="cpu",
            compute_type=self._compute,
            download_root=str(self._models_dir),
        )

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        if not pcm:
            return ""
        try:
            if self._model is None:
                self._model = self._load()
            segments, _ = self._model.transcribe(
                pcm_to_float32(pcm), language="en", beam_size=1
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
        except Exception:
            log.error("transcription failed", exc_info=True)
            return ""
```

`src/zeus/stt/fake.py`:
```python
"""Transcriber test double. Returns scripted strings."""
from __future__ import annotations


class FakeTranscriber:
    def __init__(self, script: list[str] | None = None) -> None:
        self._script = list(script or [])
        self.calls: list[int] = []

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        self.calls.append(len(pcm))
        return self._script.pop(0) if self._script else ""
```

`src/zeus/stt/__init__.py`:
```python
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
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv/bin/pytest tests/stt/ -v`
Expected: PASS — 8 tests

- [ ] **Step 5: Commit**

```bash
git add src/zeus/stt/ tests/stt/
git commit -m "feat: Transcriber protocol with local Whisper and fake implementations"
```

---

### Task 11: MicStream — single owner with ring buffer

**Files:**
- Create: `src/zeus/audio/__init__.py`, `src/zeus/audio/mic.py`
- Test: `tests/audio/test_mic.py`

**Interfaces:**
- Consumes: `AudioConfig`.
- Produces:
  - `RingBuffer(max_frames: int)` — `push(frame: bytes)`, `snapshot() -> bytes`, `clear()`.
  - `MicStream(config: AudioConfig)` with:
    - `start()`, `stop()`, context-manager support
    - `frames() -> Iterator[bytes]` — blocking iterator over live 80 ms chunks
    - `pre_roll() -> bytes` — the ring buffer contents (**this is what prevents wake-word clipping**)
    - `_on_audio(indata, frames, time_info, status)` — the sounddevice callback, tested directly
  - `FRAME_SAMPLES = 1280` (80 ms at 16 kHz), `BYTES_PER_SAMPLE = 2`

**Design note: `_on_audio` is a separate method precisely so the fan-out and ring-buffer behaviour can be tested by calling it directly, with no audio device involved.**

- [ ] **Step 1: Write the failing test**

`tests/audio/test_mic.py`:
```python
import pytest

from zeus.audio.mic import FRAME_SAMPLES, MicStream, RingBuffer
from zeus.config import AudioConfig

FRAME = b"\x01\x02" * FRAME_SAMPLES  # one 80 ms chunk


def test_ring_buffer_keeps_only_the_last_n_frames():
    ring = RingBuffer(max_frames=2)
    ring.push(b"aa")
    ring.push(b"bb")
    ring.push(b"cc")
    assert ring.snapshot() == b"bbcc"


def test_ring_buffer_clear():
    ring = RingBuffer(max_frames=2)
    ring.push(b"aa")
    ring.clear()
    assert ring.snapshot() == b""


def test_ring_capacity_is_derived_from_config():
    # 3 seconds at 16 kHz in 1280-sample frames == 37 frames
    stream = MicStream(AudioConfig(sample_rate=16000, ring_seconds=3))
    assert stream._ring._frames.maxlen == 37


def test_callback_feeds_both_the_ring_and_the_queue():
    stream = MicStream(AudioConfig())
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)

    assert stream.pre_roll() == FRAME
    assert stream._queue.get_nowait() == FRAME


def test_pre_roll_survives_queue_consumption():
    """The whole point: pre-roll must still be there after frames are read."""
    stream = MicStream(AudioConfig())
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream._queue.get_nowait()
    assert stream.pre_roll() == FRAME


def test_frames_iterator_yields_pushed_audio():
    stream = MicStream(AudioConfig())
    stream._running = True
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream.stop()  # sentinel ends the iterator

    assert list(stream.frames()) == [FRAME, FRAME]


def test_dropped_frames_are_counted_not_raised():
    stream = MicStream(AudioConfig())
    stream._queue.maxsize = 1
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)
    stream._on_audio(FRAME, FRAME_SAMPLES, None, None)  # queue full
    assert stream.dropped == 1


def test_start_is_rejected_twice():
    stream = MicStream(AudioConfig())
    stream._running = True
    with pytest.raises(RuntimeError, match="already running"):
        stream.start()
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv/bin/pytest tests/audio/test_mic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.audio'`

- [ ] **Step 3: Implement `src/zeus/audio/mic.py`**

```python
"""Single-owner microphone stream with pre-roll ring buffer. See spec §5.2.

Exactly one CoreAudio input stream exists in the process. It fans out to
two consumers: the wake-word detector (continuous) and utterance capture
(on demand).

The ring buffer is the reason the first words after a wake word are not
lost. Without it, "Zeus, what's my battery" arrives as "battery", because
the detector is still deciding while the user keeps talking.
"""
from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from typing import Iterator

log = logging.getLogger(__name__)

FRAME_SAMPLES = 1280        # 80 ms at 16 kHz — openWakeWord's expected chunk
BYTES_PER_SAMPLE = 2        # int16
_QUEUE_MAX = 256            # 20.5 s of audio
_POLL_SECONDS = 0.1         # how promptly frames() notices stop()

# NOTE: an earlier draft signalled shutdown by pushing a module-level
# _SENTINEL onto _queue. That deadlocked: _queue is bounded, so a blocking
# put() on a full queue hangs stop() forever — and a full queue is exactly
# the state after ~20 s of unconsumed audio, e.g. the half-duplex window
# while ZEUS is speaking. It also leaked across restarts, because the
# sentinel was never cleared, so the next frames() after a restart consumed
# it and silently returned. Shutdown is out-of-band and now uses an Event.


class RingBuffer:
    """Fixed-length FIFO of raw audio frames."""

    def __init__(self, max_frames: int) -> None:
        self._frames: deque[bytes] = deque(maxlen=max_frames)
        self._lock = threading.Lock()

    def push(self, frame: bytes) -> None:
        with self._lock:
            self._frames.append(frame)

    def snapshot(self) -> bytes:
        with self._lock:
            return b"".join(self._frames)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()


class MicStream:
    def __init__(self, config) -> None:
        self._config = config
        frames_per_second = config.sample_rate / FRAME_SAMPLES
        self._ring = RingBuffer(int(config.ring_seconds * frames_per_second))
        # ONE QUEUE PER CONSUMER, not one queue for the stream. A single
        # shared queue is a hand-off, not a fan-out: queue.get() removes the
        # frame, so with two consumers each frame reaches exactly ONE of
        # them. The daemon has exactly that wiring -- the wake detector
        # iterates frames() forever on its own thread while a check-in calls
        # frames() on the main thread -- so the detector would eat roughly
        # half of the user's spoken answer and hand Whisper non-contiguous
        # 80 ms chunks. Measured before the fix: 20 frames pushed, consumer A
        # saw 20, consumer B saw 0, and ZERO frames reached both.
        #
        # Copy-on-write list: _on_audio runs on the real-time audio thread
        # and must never take a lock, so subscribe/_unsubscribe REPLACE the
        # list rather than mutating it, and _on_audio just reads the
        # attribute once (an atomic read of an immutable list).
        self._subscribers: list["Subscription"] = []
        self._subscriber_lock = threading.Lock()
        self._stream = None
        self._running = False
        self._stopping = threading.Event()
        self._lifecycle = threading.Lock()
        self.dropped = 0

    # -- fan-out -------------------------------------------------------
    def subscribe(self) -> "Subscription":
        """A private frame queue for one consumer.

        A fresh subscription starts EMPTY, which is why utterance capture no
        longer needs a drain() before listening: it cannot inherit the audio
        of ZEUS's own speech, because that audio went to queues that existed
        while ZEUS was speaking. Long-lived consumers (the wake detector)
        still need drain(), since their queue does accumulate.
        """
        subscription = Subscription(self)
        with self._subscriber_lock:
            self._subscribers = [*self._subscribers, subscription]
        return subscription

    def _unsubscribe(self, subscription: "Subscription") -> None:
        with self._subscriber_lock:
            self._subscribers = [
                s for s in self._subscribers if s is not subscription
            ]

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        # Locked check-and-set: without it two callers can both pass the
        # guard and each open a RawInputStream, breaking the single-owner
        # invariant this class exists to enforce. The lock is deliberately
        # NOT taken in _on_audio — that runs on the real-time audio thread
        # and must never contend with a slow caller.
        with self._lifecycle:
            if self._running:
                raise RuntimeError("MicStream already running")
            self._running = True
        import sounddevice as sd

        # A restarted stream must inherit neither the previous run's shutdown
        # signal nor its stale audio. Clearing EVERY subscriber is safe here
        # and ONLY here — but only because this runs BEFORE the device is
        # started below, so no capture can be in flight to lose. Keep it
        # before self._stream.start(); moving it after would make the
        # comment a lie and open a window where a live callback's frames are
        # discarded. Never drain stream-wide while running — that is exactly
        # what would delete an in-flight answer.
        self._stopping.clear()
        for subscription in self._subscribers:
            subscription.drain()
        self._stream = sd.RawInputStream(
            samplerate=self._config.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=self._on_audio,
        )
        self._stream.start()
        log.info("microphone stream started at %d Hz", self._config.sample_rate)

    def stop(self) -> None:
        with self._lifecycle:
            self._running = False
        # Set BEFORE closing the device, so a consumer blocked in frames() is
        # released even if the close path raises.
        self._stopping.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                log.debug("error closing stream", exc_info=True)
            self._stream = None

    def __enter__(self) -> "MicStream":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # -- audio path ----------------------------------------------------
    def _on_audio(self, indata, frames, time_info, status) -> None:
        """sounddevice callback. Must never raise — it runs on the audio thread."""
        if self._stopping.is_set():
            # Shutdown signalled: stop feeding the queue. frames() ends by
            # draining to empty, so a producer still pushing after stop()
            # starves it of that observation and hangs the consumer. The
            # device close in stop() is best-effort and swallows exceptions,
            # so we cannot assume the callback has actually ceased.
            # Event.is_set() is a cheap non-blocking read — safe here.
            return
        if status:
            log.debug("audio status: %s", status)
        frame = bytes(indata)
        self._ring.push(frame)
        # Broadcast: every subscriber gets its OWN copy of every frame.
        # Read the list once into a local — subscribe() may replace the
        # attribute concurrently, and iterating the local keeps this loop
        # over a stable snapshot without a lock.
        for subscription in self._subscribers:
            try:
                subscription._queue.put_nowait(frame)
            except queue.Full:
                subscription.dropped += 1
                self.dropped += 1

    def frames(self) -> Iterator[bytes]:
        """One-off subscription for a single consumer, closed on exit.

        Convenience for consumers that iterate once and stop — utterance
        capture and the audio self-test. The wake detector must NOT use this:
        it needs a stable handle so unmute() can drain its own queue, so it
        calls subscribe() and holds the Subscription.
        """
        with self.subscribe() as subscription:
            yield from subscription.frames()

    def pre_roll(self) -> bytes:
        """Audio captured just before now — prepended to a new utterance."""
        return self._ring.snapshot()


class Subscription:
    """One consumer's private view of the microphone.

    Holds its own queue, so what this consumer reads is unaffected by any
    other consumer's reads. Closing it unregisters the queue — without that,
    every finished check-in would leave a queue behind that _on_audio keeps
    filling forever.
    """

    def __init__(self, mic: MicStream) -> None:
        self._mic = mic
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self.dropped = 0

    def frames(self) -> Iterator[bytes]:
        """Blocking iterator over live frames. Ends when stop() is called.

        Polls with a short timeout rather than blocking indefinitely, so
        stop() is observed promptly without anything having to be pushed
        onto the queue.

        The stop check sits in the `Empty` branch, NOT at the top of the
        loop: this is drain-then-stop, not abandon-on-stop. Checking eagerly
        would discard frames already queued when stop() was called, which is
        what test_frames_iterator_yields_pushed_audio pins. Termination is
        guaranteed by _on_audio refusing to push once _stopping is set —
        the two halves only work together.
        """
        while True:
            try:
                yield self._queue.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                if self._mic._stopping.is_set():
                    return

    def drain(self) -> None:
        """Discard THIS consumer's queued frames.

        Only the caller's own queue: draining every subscriber would let the
        wake detector's unmute() wipe the audio of an in-flight check-in
        answer, which is the bug the fan-out exists to prevent.
        """
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        self._mic._unsubscribe(self)

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
```

Create `src/zeus/audio/__init__.py` (empty).

**Testing the shutdown path safely:** the three shutdown/restart tests must
never `join()` a thread without a timeout. Run `stop()` (or the `frames()`
consumer) in a `daemon=True` thread and assert on `Event.wait(2)`. A
regression in this exact area is a *hang*, and an unguarded join turns one
failing test into a suite that never finishes.

- [ ] **Step 4: Run the test and verify it passes**

Run: `.venv/bin/pytest tests/audio/test_mic.py -v`
Expected: PASS — 12 tests (8 above, plus 4 added in review: stop() must not
block on a full queue; frames() must terminate after stop() with an empty
queue; a restart must not inherit the previous run's stop signal; and
frames() must still terminate when the audio callback keeps firing after
stop(), which is reachable because stop()'s device close is best-effort.)

- [ ] **Step 5: Commit**

```bash
git add src/zeus/audio/ tests/audio/
git commit -m "feat: single-owner mic stream with pre-roll ring buffer"
```

---

### Task 12: Endpointer

**Files:**
- Create: `src/zeus/audio/endpointer.py`
- Test: `tests/audio/test_endpointer.py`

**Interfaces:**
- Consumes: `AudioConfig`, `zeus.audio.mic.FRAME_SAMPLES`.
- Produces:
  - `rms(frame: bytes) -> float` — root-mean-square amplitude in [0, 1].
  - `Endpointer(config: AudioConfig, threshold: float = 0.02)` with:
    - `feed(frame: bytes) -> bool` — returns `True` when the utterance is complete.
    - `reset() -> None`
    - `saw_speech: bool`
  - `capture_utterance(frames, endpointer, pre_roll, listen_timeout_frames) -> bytes` — assembles pre-roll plus live frames until the endpointer fires or the listen window elapses; returns `b""` if no speech was ever heard.

- [ ] **Step 1: Write the failing test**

`tests/audio/test_endpointer.py`:
```python
import numpy as np

from zeus.audio.endpointer import Endpointer, capture_utterance, rms
from zeus.audio.mic import FRAME_SAMPLES
from zeus.config import AudioConfig

SILENCE = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()
SPEECH = (np.ones(FRAME_SAMPLES, dtype=np.int16) * 8000).tobytes()


def test_rms_of_silence_is_zero():
    assert rms(SILENCE) == 0.0


def test_rms_of_loud_audio_is_high():
    assert rms(SPEECH) > 0.2


def test_rms_of_empty_frame_is_zero():
    assert rms(b"") == 0.0


def test_silence_before_speech_never_ends_the_utterance():
    endpointer = Endpointer(AudioConfig())
    for _ in range(100):
        assert endpointer.feed(SILENCE) is False
    assert endpointer.saw_speech is False


def test_speech_then_sustained_silence_ends_the_utterance():
    # silence_timeout defaults to 1.5 s == ~19 frames of 80 ms
    endpointer = Endpointer(AudioConfig())
    endpointer.feed(SPEECH)
    assert endpointer.saw_speech is True

    results = [endpointer.feed(SILENCE) for _ in range(19)]
    assert results[-1] is True
    assert True not in results[:-1]


def test_speech_resets_the_silence_run():
    endpointer = Endpointer(AudioConfig())
    endpointer.feed(SPEECH)
    for _ in range(10):
        endpointer.feed(SILENCE)
    endpointer.feed(SPEECH)          # resets the counter
    results = [endpointer.feed(SILENCE) for _ in range(10)]
    assert True not in results        # not enough silence yet


def test_reset_clears_state():
    endpointer = Endpointer(AudioConfig())
    endpointer.feed(SPEECH)
    endpointer.reset()
    assert endpointer.saw_speech is False


def test_capture_prepends_pre_roll():
    endpointer = Endpointer(AudioConfig())
    frames = iter([SPEECH] + [SILENCE] * 19)
    audio = capture_utterance(frames, endpointer, pre_roll=SPEECH, listen_timeout_frames=100)
    assert audio.startswith(SPEECH + SPEECH)


def test_capture_returns_empty_when_only_silence_is_heard():
    endpointer = Endpointer(AudioConfig())
    frames = iter([SILENCE] * 50)
    audio = capture_utterance(frames, endpointer, pre_roll=b"", listen_timeout_frames=50)
    assert audio == b""


def test_capture_stops_at_the_listen_timeout():
    endpointer = Endpointer(AudioConfig())
    frames = iter([SPEECH] * 1000)
    audio = capture_utterance(frames, endpointer, pre_roll=b"", listen_timeout_frames=10)
    assert len(audio) == 10 * FRAME_SAMPLES * 2
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv/bin/pytest tests/audio/test_endpointer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.audio.endpointer'`

- [ ] **Step 3: Implement `src/zeus/audio/endpointer.py`**

```python
"""Utterance boundary detection by energy and silence run-length.

Deliberately simple: no VAD model, no extra dependency. A neural VAD would
be more robust in noise, but this runs on every 80 ms frame on an Intel CPU
that is already paying for wake-word inference and Whisper.
"""
from __future__ import annotations

import math
from typing import Iterable, Iterator

from zeus.audio.mic import FRAME_SAMPLES

_INT16_MAX = 32768.0
_FRAME_SECONDS = FRAME_SAMPLES / 16000.0


def rms(frame: bytes) -> float:
    """Root-mean-square amplitude of an int16 frame, normalised to [0, 1]."""
    if not frame:
        return 0.0
    import numpy as np

    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(samples ** 2))) / _INT16_MAX)


class Endpointer:
    """Fires once speech has been heard and then a sustained silence run."""

    def __init__(self, config, threshold: float = 0.02) -> None:
        self._threshold = threshold
        self._silence_frames_needed = max(
            1, round(config.silence_timeout.total_seconds() / _FRAME_SECONDS)
        )
        self.saw_speech = False
        self._silence_run = 0

    def reset(self) -> None:
        self.saw_speech = False
        self._silence_run = 0

    def feed(self, frame: bytes) -> bool:
        """Returns True when the utterance is judged complete."""
        if rms(frame) >= self._threshold:
            self.saw_speech = True
            self._silence_run = 0
            return False
        if not self.saw_speech:
            return False        # leading silence does not end anything
        self._silence_run += 1
        return self._silence_run >= self._silence_frames_needed


def capture_utterance(
    frames: Iterable[bytes],
    endpointer: Endpointer,
    pre_roll: bytes,
    listen_timeout_frames: int,
) -> bytes:
    """Assemble pre-roll plus live audio until the endpointer or timeout fires.

    Returns b"" when no speech was heard at all, which the caller treats as
    a NO_ANSWER (spec §9.3).
    """
    endpointer.reset()
    collected: list[bytes] = [pre_roll] if pre_roll else []
    consumed = 0
    iterator: Iterator[bytes] = iter(frames)

    for frame in iterator:
        collected.append(frame)
        consumed += 1
        if endpointer.feed(frame):
            break
        if consumed >= listen_timeout_frames:
            break

    if not endpointer.saw_speech:
        return b""
    return b"".join(collected)
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `.venv/bin/pytest tests/audio/test_endpointer.py -v`
Expected: PASS — 10 tests

- [ ] **Step 5: Commit**

```bash
git add src/zeus/audio/endpointer.py tests/audio/test_endpointer.py
git commit -m "feat: energy-based utterance endpointing with pre-roll assembly"
```

---

### Task 13: Activator — wake word and hotkey

**Files:**
- Create: `src/zeus/audio/activator.py`, `src/zeus/audio/wakeword.py`
- Test: `tests/audio/test_activator.py`

**Interfaces:**
- Consumes: `WakeConfig`, `MicStream`.
- Produces:
  - `ActivationEvent(source: str)` — `"wake"` or `"hotkey"`.
  - `Activator` protocol: `start()`, `stop()`, `events() -> Iterator[ActivationEvent]`.
  - `FakeActivator(count: int = 1)` — yields `count` events then stops; `started`/`stopped` flags.
  - `WakeWordActivator(mic: MicStream, config: WakeConfig, threshold: float = 0.5)` — openWakeWord over `mic.frames()`; `mute()` / `unmute()` for the half-duplex rule (spec §7.3).
  - `HotkeyActivator(trigger_file: Path)` — deterministic fallback; fires when a sentinel file appears, so it needs no Accessibility permission and is trivially testable.
  - `build_activator(config: WakeConfig, mic: MicStream) -> Activator`.

- [ ] **Step 1: Write the failing test**

`tests/audio/test_activator.py`:
```python
import numpy as np
import pytest

from zeus.audio.activator import (
    ActivationEvent,
    FakeActivator,
    HotkeyActivator,
    build_activator,
)
from zeus.audio.mic import FRAME_SAMPLES, MicStream
from zeus.audio.wakeword import WakeWordActivator
from zeus.config import AudioConfig, WakeConfig

FRAME = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()


def test_fake_activator_yields_then_stops():
    activator = FakeActivator(count=2)
    activator.start()
    assert [e.source for e in activator.events()] == ["fake", "fake"]
    assert activator.started is True


def test_hotkey_activator_fires_on_sentinel_file(tmp_path):
    trigger = tmp_path / "trigger"
    activator = HotkeyActivator(trigger, poll_seconds=0)
    activator.start()
    trigger.touch()
    event = next(activator.events())
    assert event.source == "hotkey"
    assert not trigger.exists()  # consumed
    activator.stop()


class DummyModel:
    """Stands in for openwakeword.Model."""

    def __init__(self, scores):
        self._scores = list(scores)

    def predict(self, samples):
        return {"hey_jarvis": self._scores.pop(0) if self._scores else 0.0}


def _wake_activator(monkeypatch, scores, frame_count):
    mic = MicStream(AudioConfig())
    for _ in range(frame_count):
        mic._on_audio(FRAME, FRAME_SAMPLES, None, None)
    mic.stop()  # sentinel terminates frames()

    activator = WakeWordActivator(mic, WakeConfig(), threshold=0.5)
    monkeypatch.setattr(activator, "_load_model", lambda: DummyModel(scores))
    return activator


def test_wake_word_fires_above_threshold(monkeypatch):
    activator = _wake_activator(monkeypatch, [0.1, 0.9, 0.1], 3)
    activator.start()
    assert [e.source for e in activator.events()] == ["wake"]


def test_wake_word_ignores_scores_below_threshold(monkeypatch):
    activator = _wake_activator(monkeypatch, [0.1, 0.2, 0.3], 3)
    activator.start()
    assert list(activator.events()) == []


def test_muting_suppresses_detection(monkeypatch):
    """Half-duplex rule, spec §7.3: ZEUS must not hear itself speaking."""
    activator = _wake_activator(monkeypatch, [0.9, 0.9, 0.9], 3)
    activator.start()
    activator.mute()
    assert list(activator.events()) == []


def test_unmute_restores_detection(monkeypatch):
    activator = _wake_activator(monkeypatch, [0.9], 1)
    activator.start()
    # events() is created BEFORE muting, as in production: the detector runs
    # continuously and is already live long before ZEUS ever speaks. An
    # earlier draft muted and unmuted before events() existed, which forced
    # unmute()'s mic.drain() to be deleted to make it pass — reintroducing
    # self-triggering. See the unmute() docstring below.
    events = activator.events()
    activator.mute()
    activator.unmute()
    assert [e.source for e in events] == ["wake"]


def test_factory_builds_wake_word_activator():
    mic = MicStream(AudioConfig())
    assert isinstance(build_activator(WakeConfig(), mic), WakeWordActivator)


def test_factory_rejects_unknown_provider():
    mic = MicStream(AudioConfig())
    with pytest.raises(ValueError, match="unknown wake provider"):
        build_activator(WakeConfig(provider="porcupine"), mic)


def test_activation_event_is_hashable_and_comparable():
    assert ActivationEvent("wake") == ActivationEvent("wake")
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv/bin/pytest tests/audio/test_activator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.audio.activator'`

- [ ] **Step 3: Implement `src/zeus/audio/activator.py`**

```python
"""Activation sources. See spec §5.1.

Two implementations ship from day one: the wake word the user asked for,
and a deterministic hotkey fallback that needs no Accessibility permission
and gives a reliable path when wake-word accuracy disappoints (risk R5).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

from zeus.config import WakeConfig


@dataclass(frozen=True)
class ActivationEvent:
    source: str


class Activator(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def events(self) -> Iterator[ActivationEvent]: ...


class FakeActivator:
    def __init__(self, count: int = 1) -> None:
        self._count = count
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def events(self) -> Iterator[ActivationEvent]:
        for _ in range(self._count):
            yield ActivationEvent("fake")


class HotkeyActivator:
    """Fires when a sentinel file appears, then deletes it.

    A file rather than a real global hotkey keeps Slice 1 free of
    Accessibility permissions. A shell alias or Shortcut can `touch` the
    file; Slice 2 can bind a real key to that.
    """

    def __init__(self, trigger_file: Path, poll_seconds: float = 0.25) -> None:
        self._trigger = trigger_file
        self._poll = poll_seconds
        self._running = False

    def start(self) -> None:
        self._running = True
        self._trigger.parent.mkdir(parents=True, exist_ok=True)
        self._trigger.unlink(missing_ok=True)

    def stop(self) -> None:
        self._running = False

    def events(self) -> Iterator[ActivationEvent]:
        while self._running:
            if self._trigger.exists():
                self._trigger.unlink(missing_ok=True)
                yield ActivationEvent("hotkey")
            if self._poll:
                time.sleep(self._poll)


def build_activator(config: WakeConfig, mic) -> Activator:
    from zeus.audio.wakeword import WakeWordActivator

    if config.provider == "openwakeword":
        return WakeWordActivator(mic, config)
    raise ValueError(f"unknown wake provider: {config.provider!r}")
```

`src/zeus/audio/wakeword.py`:
```python
"""openWakeWord activation over the shared mic stream.

RISK R2: openWakeWord ships no 'zeus' model, so the default is
'hey_jarvis'. `WakeConfig.model` is a name or a path to an .onnx file, so a
custom model is a config edit rather than a code change.
"""
from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from zeus.audio.activator import ActivationEvent
from zeus.config import WakeConfig

log = logging.getLogger(__name__)


class WakeWordActivator:
    def __init__(self, mic, config: WakeConfig, threshold: float = 0.5) -> None:
        self._mic = mic
        self._config = config
        self._threshold = threshold
        self._model = None
        self._muted = False
        self._running = False
        # Set while events() is iterating; unmute() drains through it.
        self._subscription = None
        # Depth, not a flag: mute windows nest across two threads.
        self._mute_depth = 0
        self._mute_lock = threading.Lock()

    def _load_model(self):
        import openwakeword
        from openwakeword.model import Model

        try:
            openwakeword.utils.download_models()
        except Exception:
            log.debug("wake model download skipped", exc_info=True)
        return Model(wakeword_models=[self._config.model])

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def mute(self) -> None:
        """Suppress detection while ZEUS is speaking (spec §7.3, half-duplex).

        A DEPTH COUNTER, not a boolean. The daemon shares one VoiceIO and
        one activator between the main thread (scheduled check-ins) and the
        wake thread (ad-hoc conversations), and they overlap in practice:
        the user says "hey zeus" at 08:59:55, listen() mutes and opens a
        window of up to 30s; the morning check-in fires at 09:00:00 on the
        main thread, speaks its opener, and its `finally: unmute()` clears
        the mute while the wake thread is STILL capturing. Reproduced: the
        detector scored 5 of 5 frames of the user's in-flight answer. With
        a counter, the inner unmute decrements to 1 and the detector stays
        muted until the outer window closes.

        BOTH WRITES GO INSIDE THE LOCK. An earlier version of this snippet
        incremented the depth under the lock and then set _muted outside
        it, which loses a race: thread A increments 0->1 and releases the
        lock; before A sets _muted, thread B's unmute() takes the lock,
        decrements 1->0, sees no window open, sets _muted = False and
        returns; A then sets _muted = True. The result is depth 0 with
        _muted True — nobody holds a mute, but events() reads _muted, so
        the wake word is DEAF until some later matched mute/unmute pair
        happens to clear it. Reproduced deterministically by pausing A
        between the lock release and the write.
        """
        with self._mute_lock:
            self._mute_depth += 1
            self._muted = True

    def unmute(self) -> None:
        """Re-enable detection after ZEUS has finished speaking.

        The drain is load-bearing, not tidiness. It is tempting to remove it
        by arguing that events() keeps pulling frames and skipping them
        while muted — but events() is a generator, and mid-conversation it
        is SUSPENDED at its yield, consuming nothing. The queue therefore
        fills with ZEUS's own voice, and without this drain the detector
        scores all of it on resume and re-triggers itself. Measured against
        a build with the drain removed: 51 frames of self-audio queued, all
        51 scored after unmute. drain() on an empty queue is a no-op, so
        unmute() without a prior mute() stays safe.

        Drains THIS detector's own subscription, never the whole stream: a
        scheduled check-in capturing an answer on the main thread holds its
        own subscription, and a stream-wide drain would delete the user's
        in-flight reply.

        Only the OUTERMOST unmute actually unmutes — see mute() for why.

        Clamped at zero as DEFENCE IN DEPTH, not because anything emits a
        stray unmute today: both call sites (VoiceIO.speak and
        VoiceIO.listen) run mute() immediately before a try/finally that
        unmutes, and no activator defines unmute without mute, so the
        `if unmute:` guard never fires unpaired. An earlier version of this
        docstring claimed VoiceIO "calls unmute() in a finally whether or
        not the paired mute() ran" — that is false, and worth correcting
        rather than deleting: without the clamp, ONE stray unmute would
        leave the depth at -1 and the next nested window would unwind a
        level early, unmuting while an outer window is still open.
        """
        with self._mute_lock:
            self._mute_depth = max(0, self._mute_depth - 1)
            if self._mute_depth > 0:
                return
            self._muted = False
        if self._subscription is not None:
            self._subscription.drain()

    def events(self) -> Iterator[ActivationEvent]:
        if self._model is None:
            self._model = self._load_model()
        # subscribe() rather than mic.frames(): the detector is a long-lived
        # consumer that needs a stable handle for unmute() to drain. Closing
        # it on exit stops _on_audio filling a queue nobody reads.
        with self._mic.subscribe() as subscription:
            self._subscription = subscription
            try:
                yield from self._detect(subscription)
            finally:
                self._subscription = None

    def _detect(self, subscription) -> Iterator[ActivationEvent]:
        for frame in subscription.frames():
            if not self._running:
                return
            if self._muted:
                continue
            samples = np.frombuffer(frame, dtype=np.int16)
            try:
                scores = self._model.predict(samples)
            except Exception:
                log.error("wake-word inference failed", exc_info=True)
                continue
            if any(score >= self._threshold for score in scores.values()):
                yield ActivationEvent("wake")
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `.venv/bin/pytest tests/audio/test_activator.py -v`
Expected: PASS — 10 tests (the Step 1 file collects 9; an earlier draft said
10, miscounting. The tenth is `test_unmute_discards_audio_captured_while_speaking`,
added in review to pin *why* unmute() drains — without it nothing
distinguishes the correct implementation from one that lets ZEUS hear its
own voice and re-trigger.)

- [ ] **Step 5: Commit**

```bash
git add src/zeus/audio/activator.py src/zeus/audio/wakeword.py tests/audio/test_activator.py
git commit -m "feat: wake-word and hotkey activators with half-duplex muting"
```

---

### Task 14: Brain — prompts, tools, and the Tool Runner conversation

**Files:**
- Create: `src/zeus/brain/__init__.py`, `src/zeus/brain/prompts.py`, `src/zeus/brain/tools.py`, `src/zeus/brain/conversation.py`, `src/zeus/brain/fake.py`
- Test: `tests/brain/test_prompts.py`, `tests/brain/test_tools.py`, `tests/brain/test_conversation.py`

**Interfaces:**
- Consumes: `BrainConfig`, `Store`, `Journal`, `anthropic`.
- Produces:
  - `SYSTEM_PROMPT: str`, `MORNING_OPENER: str`, `EVENING_OPENER(goal_text: str) -> str`, `FOLDED_OPENER: str`
  - `split_sentences(buffer: str) -> tuple[list[str], str]` — pure; returns complete sentences plus the unflushed remainder.
  - `logged_tool(store, conversation_id, name, fn) -> Callable` — wraps a tool callable so every invocation writes an `actions` row with timing and success, and returns `is_error`-shaped text on failure.
  - `build_tool_callables(store, journal, conversation_id, local_date) -> dict[str, Callable]` — the action-logged **plain** callables, keyed by tool name. Returned separately from `build_tools` so tests and `FakeConversation` can invoke the real tool bodies without depending on what the `@beta_tool` decorator does to `__name__` or to direct callability — neither is guaranteed by the SDK.
  - `build_tools(store, journal, conversation_id, local_date) -> list` — the `@beta_tool`-decorated functions handed to the Tool Runner: `save_goal`, `record_outcome`. Thin wrappers over `build_tool_callables`.
  - `Conversation(client, config, store, journal, conversation_id, system, tools)` with `send(text: str) -> Iterator[str]` yielding **sentences** for TTS.
  - `FakeConversation(script=None, tools=None, tool_calls=None)` — same `send` shape, and can invoke **real** tool callables so an end-to-end test exercises brain → tool → action log → database without a network call. `tools` is the dict from `build_tool_callables`; `tool_calls[i]` is the list of `(name, kwargs)` to invoke on the i-th `send()`. Exposes `sent` and `invoked`. Used by Tasks 15 and 18.

- [ ] **Step 1: Write the failing tests**

`tests/brain/test_prompts.py`:
```python
from zeus.brain.prompts import (
    EVENING_OPENER,
    FOLDED_OPENER,
    MORNING_OPENER,
    SYSTEM_PROMPT,
    split_sentences,
)


def test_system_prompt_is_long_enough_to_cache():
    """Opus 5's prompt-cache minimum is 512 tokens.

    2048 chars is a deliberately conservative proxy. Measured against the
    live count_tokens endpoint, the shipped 2049-char prompt is ~640 tokens
    — about 3.2 chars/token, not the 4 this threshold assumes — so the real
    floor is nearer 1640 chars and this test leaves ~25% headroom. Below the
    minimum a cache_control marker does not error; it silently no-ops with
    cache_creation_input_tokens: 0, which is why this is pinned by a test at
    all rather than left to inspection.
    """
    assert len(SYSTEM_PROMPT) > 2048


def test_system_prompt_states_the_exchange_ceiling():
    assert "three exchanges" in SYSTEM_PROMPT.lower()


def test_evening_opener_embeds_the_goal():
    assert "Finish the auth flow" in EVENING_OPENER("Finish the auth flow")


def test_openers_are_distinct():
    assert len({MORNING_OPENER, EVENING_OPENER("x"), FOLDED_OPENER}) == 3


def test_split_sentences_emits_complete_sentences_only():
    done, rest = split_sentences("Morning. What's the one thing")
    assert done == ["Morning."]
    assert rest == " What's the one thing"


def test_split_sentences_handles_question_and_exclamation():
    done, rest = split_sentences("Done? Great! Now")
    assert done == ["Done?", "Great!"]
    assert rest == " Now"


def test_split_sentences_returns_nothing_when_incomplete():
    done, rest = split_sentences("Morning")
    assert done == []
    assert rest == "Morning"


def test_split_sentences_does_not_break_on_decimals():
    done, rest = split_sentences("It took 1.5 hours. Next")
    assert done == ["It took 1.5 hours."]
    assert rest == " Next"
```

`tests/brain/test_tools.py`:
```python
from datetime import datetime, timezone

import pytest

from zeus.brain.tools import build_tool_callables, build_tools, logged_tool
from zeus.clock import FakeClock
from zeus.memory.journal import Journal
from zeus.memory.store import Store
from zoneinfo import ZoneInfo

START = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def wiring(tmp_path):
    clock = FakeClock(START)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, ZoneInfo("Africa/Lagos"))
    conv = store.start_conversation("schedule")
    return store, journal, conv


def test_logged_tool_records_a_successful_action(wiring):
    store, _, conv = wiring
    wrapped = logged_tool(store, conv, "demo", lambda value: f"got {value}")
    assert wrapped(value="x") == "got x"

    action = store.recent_actions()[0]
    assert action.tool == "demo"
    assert action.ok is True
    assert action.args == {"value": "x"}
    assert action.error is None


def test_logged_tool_captures_failures_without_raising(wiring):
    store, _, conv = wiring

    def explode():
        raise RuntimeError("nope")

    wrapped = logged_tool(store, conv, "boom", explode)
    result = wrapped()
    assert "nope" in result

    action = store.recent_actions()[0]
    assert action.ok is False
    assert "nope" in action.error


def test_save_goal_writes_goal_and_journal(wiring):
    store, journal, conv = wiring
    tools = build_tool_callables(store, journal, conv, "2026-08-05")
    tools["save_goal"](text="Finish the auth flow")

    assert store.get_goal("2026-08-05").text == "Finish the auth flow"
    assert "Finish the auth flow" in journal.read("2026-08-05")
    assert store.recent_actions()[0].tool == "save_goal"


def test_record_outcome_updates_status(wiring):
    store, journal, conv = wiring
    tools = build_tool_callables(store, journal, conv, "2026-08-05")
    tools["save_goal"](text="Ship it")
    tools["record_outcome"](status="partial", notes="tests missing")

    goal = store.get_goal("2026-08-05")
    assert goal.status == "partial"
    assert goal.notes == "tests missing"


def test_record_outcome_without_a_goal_is_reported_not_raised(wiring):
    store, journal, conv = wiring
    tools = build_tool_callables(store, journal, conv, "2026-08-05")
    result = tools["record_outcome"](status="done")
    assert "no goal" in result.lower()
    assert store.recent_actions()[0].ok is True


def test_record_outcome_rejects_an_invalid_status(wiring):
    store, journal, conv = wiring
    tools = build_tool_callables(store, journal, conv, "2026-08-05")
    tools["save_goal"](text="Ship it")
    result = tools["record_outcome"](status="banana")
    assert "banana" in result
    assert store.get_goal("2026-08-05").status == "pending"


def test_callables_and_decorated_tools_cover_the_same_surface(wiring):
    """build_tools wraps exactly the callables build_tool_callables exposes."""
    store, journal, conv = wiring
    assert set(build_tool_callables(store, journal, conv, "2026-08-05")) == {
        "save_goal", "record_outcome",
    }
    assert len(build_tools(store, journal, conv, "2026-08-05")) == 2
```

`tests/brain/test_conversation.py`:
```python
from datetime import datetime, timezone

import pytest
from zoneinfo import ZoneInfo

from zeus.brain.conversation import Conversation
from zeus.brain.fake import FakeConversation
from zeus.clock import FakeClock
from zeus.config import BrainConfig
from zeus.memory.journal import Journal
from zeus.memory.store import Store

START = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)


# ---- stubs mirroring the anthropic Tool Runner shape --------------------
class StubEvent:
    def __init__(self, text):
        self.type = "content_block_delta"
        self.delta = type("D", (), {"type": "text_delta", "text": text})()


class StubStream:
    def __init__(self, chunks, stop_reason="end_turn", content=None):
        self._chunks = chunks
        self._stop_reason = stop_reason
        # `content` defaults to [] so existing tests are unaffected, but it
        # MUST be settable: an earlier draft hardcoded [], which made a
        # malformed multi-round history structurally indistinguishable from a
        # well-formed one and hid a Critical defect. A round that used a tool
        # needs a real tool_use block here.
        self._content = content if content is not None else []

    def __iter__(self):
        return iter(StubEvent(c) for c in self._chunks)

    def get_final_message(self):
        return type("M", (), {
            "stop_reason": self._stop_reason,
            "content": self._content,
            "stop_details": None,
        })()


class StubClient:
    def __init__(self, streams):
        """streams[i] is turn i's response.

        A bare StubStream means that turn is a single round. A LIST of
        StubStreams means the Tool Runner yielded several rounds within that
        one turn — model call, tool execution, model call again — which is
        what the real runner does whenever a tool is used.

        An earlier draft returned the whole list on every call, conflating
        "later turns" with "more rounds in this turn". A two-turn test then
        had both streams consumed inside the FIRST send(), so
        test_history_accumulates_across_turns saw
        [user, assistant, assistant, user] and failed against the plan's own
        Conversation.
        """
        self._streams = streams
        self.calls = []
        beta = type("B", (), {})()
        beta.messages = type("M", (), {"tool_runner": self._tool_runner})()
        self.beta = beta

    def _tool_runner(self, **kwargs):
        self.calls.append(kwargs)
        turn = self._streams[len(self.calls) - 1]
        # IndexError on more calls than turns is deliberate: loud beats a
        # silent replay of an earlier turn's stream.
        return iter(turn if isinstance(turn, list) else [turn])


@pytest.fixture
def wiring(tmp_path):
    clock = FakeClock(START)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, ZoneInfo("Africa/Lagos"))
    return store, journal, store.start_conversation("schedule")


def _conversation(client, wiring):
    store, journal, conv = wiring
    return Conversation(
        client=client, config=BrainConfig(), store=store, journal=journal,
        conversation_id=conv, system="SYSTEM", tools=[],
    )


def test_send_yields_complete_sentences(wiring):
    client = StubClient([StubStream(["Morning.", " What's the ", "one thing?"])])
    assert list(_conversation(client, wiring).send("[morning]")) == [
        "Morning.", "What's the one thing?",
    ]


def test_trailing_fragment_is_flushed_at_the_end(wiring):
    client = StubClient([StubStream(["No punctuation here"])])
    assert list(_conversation(client, wiring).send("hi")) == ["No punctuation here"]


def test_messages_are_persisted(wiring):
    store, _, conv = wiring
    client = StubClient([StubStream(["Morning."])])
    list(_conversation(client, wiring).send("[morning]"))

    roles = [m.role for m in store.messages(conv)]
    assert roles == ["user", "assistant"]
    assert store.messages(conv)[1].content == "Morning."


def test_request_uses_the_required_model_settings(wiring):
    client = StubClient([StubStream(["ok."])])
    list(_conversation(client, wiring).send("hi"))
    kwargs = client.calls[0]

    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["thinking"] == {"type": "adaptive"}     # never "disabled"
    assert kwargs["output_config"]["effort"] == "low"
    assert kwargs["stream"] is True
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_refusal_is_handled_before_reading_content(wiring):
    client = StubClient([StubStream([], stop_reason="refusal")])
    spoken = list(_conversation(client, wiring).send("hi"))
    assert len(spoken) == 1
    assert "can't help" in spoken[0].lower()


def test_api_failure_yields_a_spoken_fallback(wiring):
    class Exploding(StubClient):
        def _tool_runner(self, **kwargs):
            raise RuntimeError("connection reset")

    spoken = list(_conversation(Exploding([]), wiring).send("hi"))
    assert len(spoken) == 1
    assert "reach" in spoken[0].lower()


def test_history_accumulates_across_turns(wiring):
    client = StubClient([StubStream(["One."]), StubStream(["Two."])])
    conversation = _conversation(client, wiring)
    list(conversation.send("first"))
    list(conversation.send("second"))

    second_call = client.calls[1]["messages"]
    assert [m["role"] for m in second_call] == ["user", "assistant", "user"]


def test_a_single_turn_can_span_several_tool_runner_rounds(wiring):
    """The real runner yields one stream per round whenever a tool is used;
    send() must surface sentences from every round, not just the first.

    Added in review: fixing StubClient to hand out one turn per call would
    otherwise have left this path — the whole reason send() iterates the
    runner rather than reading a single response — with no coverage at all.
    """
    client = StubClient([[StubStream(["Saving that."]), StubStream(["Done."])]])
    assert list(_conversation(client, wiring).send("[morning]")) == [
        "Saving that.", "Done.",
    ]


def test_fake_conversation_replays_a_script():
    fake = FakeConversation({"[morning]": ["Morning.", "What's the goal?"]})
    assert list(fake.send("[morning]")) == ["Morning.", "What's the goal?"]
    assert fake.sent == ["[morning]"]


def test_fake_conversation_falls_back_for_unscripted_input():
    fake = FakeConversation({})
    assert list(fake.send("anything")) == ["Got it."]


def test_fake_conversation_invokes_real_tools(wiring):
    """This is what lets the end-to-end test prove the real tool path."""
    from zeus.brain.tools import build_tool_callables

    store, journal, conv = wiring
    tools = build_tool_callables(store, journal, conv, "2026-08-05")
    fake = FakeConversation(
        tools=tools,
        tool_calls=[[("save_goal", {"text": "Finish the auth flow"})]],
    )

    list(fake.send("[morning]"))

    assert store.get_goal("2026-08-05").text == "Finish the auth flow"
    assert store.recent_actions()[0].tool == "save_goal"
    assert fake.invoked == [("save_goal", {"text": "Finish the auth flow"})]


def test_fake_conversation_invokes_tools_per_turn(wiring):
    from zeus.brain.tools import build_tool_callables

    store, journal, conv = wiring
    tools = build_tool_callables(store, journal, conv, "2026-08-05")
    fake = FakeConversation(
        tools=tools,
        tool_calls=[
            [("save_goal", {"text": "Ship it"})],
            [],
            [("record_outcome", {"status": "done"})],
        ],
    )

    list(fake.send("turn one"))
    list(fake.send("turn two"))
    list(fake.send("turn three"))

    assert [name for name, _ in fake.invoked] == ["save_goal", "record_outcome"]
    assert store.get_goal("2026-08-05").status == "done"


def test_fake_conversation_rejects_an_unknown_tool():
    fake = FakeConversation(tools={}, tool_calls=[[("nope", {})]])
    with pytest.raises(KeyError, match="nope"):
        list(fake.send("x"))
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/bin/pytest tests/brain/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.brain'`

- [ ] **Step 3: Implement `src/zeus/brain/prompts.py`**

```python
"""System prompt, check-in openers, and sentence splitting. See spec §7.2."""
from __future__ import annotations

import re

SYSTEM_PROMPT = """\
You are ZEUS, a voice assistant running on the user's Mac. Everything you \
say is converted to speech and played aloud, and everything the user says \
reaches you as an imperfect speech-to-text transcript.

Write for the ear, not the page. No markdown, no bullet points, no headings, \
no emoji, no code blocks, no URLs — none of it survives text-to-speech. Write \
numbers and abbreviations the way they should be spoken.

Keep replies to one or two short sentences. The user is listening, not \
reading, and cannot skim. A long reply is worse than an incomplete one.

You are an accountability partner, not a coach and not a logger. Your job in \
the daily check-ins is to capture what the user commits to and what actually \
happened, with as little friction as possible.

Morning check-in: ask what the one thing is that has to happen today. If the \
answer is vague — "work on the app", "be productive" — ask once for something \
concrete that could be judged done or not done by tonight. Ask only once, then \
accept whatever you get and save it with the save_goal tool. Do not negotiate, \
do not suggest goals, do not offer encouragement.

Evening check-in: you will be told what the goal was. Ask whether it happened. \
Record the answer with the record_outcome tool, choosing done when it was \
finished, partial when it was started but not completed, and missed when it \
did not happen. Do not judge, do not console, do not analyse why. If it was \
not done you may offer once to carry it to tomorrow, and accept the answer \
either way.

Never exceed three exchanges in a check-in. When you have what you need, say \
one short closing line and stop. Silence is better than filler.

The transcript you receive may contain speech-recognition errors. If a reply \
is garbled, ask once for a repeat; if it is still unclear, save your best \
interpretation rather than asking a third time.

Messages wrapped in square brackets, such as [morning check-in], are \
instructions from the system rather than speech from the user. Act on them \
but never read them aloud or mention them.
"""

MORNING_OPENER = (
    "[morning check-in] Greet the user briefly and ask what the one thing is "
    "that has to happen today."
)

FOLDED_OPENER = (
    "[evening check-in, goal never captured] The morning check-in was missed, "
    "so no goal was recorded today. Ask what the user ended up focusing on, "
    "save it with save_goal, then ask how it went."
)


def EVENING_OPENER(goal_text: str) -> str:
    return (
        f"[evening check-in] Today's goal was: {goal_text}. "
        "Ask whether it happened."
    )


# A sentence ends at . ? or ! that is followed by whitespace or end-of-string,
# and is not part of a decimal number.
_SENTENCE_END = re.compile(r"(?<!\d)([.!?])(?=\s|$)")


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split a streaming buffer into complete sentences plus a remainder.

    Returns (sentences, remainder). The remainder is whatever has not yet
    been terminated and must be carried into the next chunk.
    """
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(buffer):
        end = match.end()
        piece = buffer[start:end].strip()
        if piece:
            sentences.append(piece)
        start = end
    return sentences, buffer[start:]
```

- [ ] **Step 4: Implement `src/zeus/brain/tools.py`**

```python
"""Tool definitions and the action-logging wrapper. See spec §6.2, §12.

Every tool call writes an `actions` row. That table is the spine of the
Slice 2 dashboard, which can only show history that was recorded from the
beginning — hence it exists in Slice 1 despite having nothing to display yet.
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Callable

from zeus.memory.journal import Journal
from zeus.memory.store import Store

log = logging.getLogger(__name__)

VALID_STATUSES = {"done", "partial", "missed", "carried"}


def logged_tool(
    store: Store, conversation_id: int, name: str, fn: Callable
) -> Callable:
    """Wrap a tool so every call is timed and recorded.

    A raising tool returns an error *string* rather than propagating, so the
    model can adapt (spec §10) instead of the turn collapsing.
    """

    @functools.wraps(fn)
    def wrapper(**kwargs):
        started = time.monotonic()
        try:
            result = fn(**kwargs)
            elapsed = int((time.monotonic() - started) * 1000)
            store.log_action(name, kwargs, result, True, elapsed,
                             conversation_id=conversation_id)
            return result
        except Exception as exc:  # noqa: BLE001 — deliberately broad
            elapsed = int((time.monotonic() - started) * 1000)
            log.error("tool %s failed", name, exc_info=True)
            store.log_action(name, kwargs, None, False, elapsed,
                             error=str(exc), conversation_id=conversation_id)
            return f"The {name} tool failed: {exc}"

    return wrapper


def build_tool_callables(
    store: Store, journal: Journal, conversation_id: int, local_date: str
) -> dict[str, Callable]:
    """The action-logged tool bodies, keyed by tool name.

    Kept separate from `build_tools` so tests and FakeConversation can call
    the real bodies directly. Depending on the @beta_tool decorator to
    preserve __name__ or to stay callable is an SDK-internals assumption
    this codebase does not make.
    """

    def _save_goal(text: str) -> str:
        store.set_goal(local_date, text)
        journal.append(f"Goal set: {text}")
        return f"Saved today's goal: {text}"

    def _record_outcome(status: str, notes: str | None = None) -> str:
        if status not in VALID_STATUSES:
            return (
                f"Cannot record status {status!r}. "
                f"Valid values are: {', '.join(sorted(VALID_STATUSES))}."
            )
        goal = store.get_goal(local_date)
        if goal is None:
            return "There is no goal recorded for today, so nothing to update."
        store.update_goal(goal.id, status=status, notes=notes)
        journal.append(f"Outcome: {status}" + (f" — {notes}" if notes else ""))
        return f"Recorded today's goal as {status}."

    return {
        "save_goal": logged_tool(store, conversation_id, "save_goal", _save_goal),
        "record_outcome": logged_tool(
            store, conversation_id, "record_outcome", _record_outcome
        ),
    }


def build_tools(
    store: Store, journal: Journal, conversation_id: int, local_date: str
) -> list[Callable]:
    """The Slice 1 tool surface, decorated for the Tool Runner.

    Deliberately tiny — two tools. Their purpose is to make the Tool Runner
    path real from day one so later slices add tools to a working mechanism
    rather than building the mechanism alongside their first tool.
    """
    from anthropic import beta_tool

    callables = build_tool_callables(store, journal, conversation_id, local_date)

    @beta_tool
    def save_goal(text: str) -> str:
        """Save the user's single goal for today.

        Args:
            text: The goal, in the user's own words, as concretely as they gave it.
        """
        return callables["save_goal"](text=text)

    @beta_tool
    def record_outcome(status: str, notes: str = "") -> str:
        """Record how today's goal turned out.

        Args:
            status: One of "done", "partial", "missed", or "carried".
            notes: Optional short detail the user gave, in their own words.
        """
        return callables["record_outcome"](status=status, notes=notes or None)

    return [save_goal, record_outcome]
```

- [ ] **Step 5: Implement `src/zeus/brain/conversation.py` and `fake.py`**

`src/zeus/brain/conversation.py`:
```python
"""Claude conversation over the Anthropic Tool Runner. See spec §7.1.

Streams the reply and yields complete sentences so text-to-speech can begin
before the model has finished — the difference between a responsive voice
interface and a laggy one.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterator

from zeus.brain.prompts import split_sentences
from zeus.config import BrainConfig
from zeus.memory.journal import Journal
from zeus.memory.store import Store

log = logging.getLogger(__name__)

MAX_TOKENS = 1024

REFUSAL_LINE = "Sorry, I can't help with that one."
ERROR_LINE = "I can't reach my brain right now. I'll try again later."


class Conversation:
    def __init__(
        self,
        client: Any,
        config: BrainConfig,
        store: Store,
        journal: Journal,
        conversation_id: int,
        system: str,
        tools: list[Callable],
        effort: str | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._store = store
        self._journal = journal
        self._conversation_id = conversation_id
        self._system = system
        self._tools = tools
        self._effort = effort or config.effort_checkin
        self._messages: list[dict[str, Any]] = []

    def send(self, text: str) -> Iterator[str]:
        """Send a turn and yield the reply one complete sentence at a time."""
        self._messages.append({"role": "user", "content": text})
        self._store.add_message(self._conversation_id, "user", text)

        buffer = ""
        spoken: list[str] = []
        final_content = None
        try:
            runner = self._client.beta.messages.tool_runner(
                model=self._config.model,
                max_tokens=MAX_TOKENS,
                # Adaptive thinking stays on. Disabling it on Opus 5 risks
                # tool calls being emitted as visible text that never run.
                thinking={"type": "adaptive"},
                output_config={"effort": self._effort},
                system=[{
                    "type": "text",
                    "text": self._system,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=self._tools,
                messages=list(self._messages),
                stream=True,
            )

            for stream in runner:
                for event in stream:
                    if getattr(event, "type", None) != "content_block_delta":
                        continue
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", None) != "text_delta":
                        continue
                    buffer += delta.text
                    sentences, buffer = split_sentences(buffer)
                    for sentence in sentences:
                        spoken.append(sentence)
                        yield sentence

                final = stream.get_final_message()
                # Check stop_reason BEFORE touching content: on a refusal
                # `content` is empty and indexing it would crash (spec §10).
                if getattr(final, "stop_reason", None) == "refusal":
                    log.warning("model refused the request")
                    spoken.append(REFUSAL_LINE)
                    yield REFUSAL_LINE
                    buffer = ""
                    break
                # Keep only the LAST round's content, appended once after the
                # loop. Appending per round produced consecutive assistant
                # messages carrying an unresolved tool_use with no
                # tool_result, which the API rejects with a 400 on the NEXT
                # send() — and the broad except below then reported a fake
                # outage right after the tool had actually succeeded. The
                # evening check-in hits exactly that shape: record_outcome
                # fires, then ZEUS asks whether to carry the goal forward.
                #
                # Safe because the runner COPIES the list we pass
                # (`messages = list(self._params["messages"])` in anthropic
                # 0.120.2) and resolves tool_use/tool_result internally. What
                # we keep here is only what the NEXT turn replays, and one
                # closing assistant message per turn keeps role alternation
                # valid by construction — without reading SDK internals.
                final_content = final.content

        except Exception:
            log.error("conversation failed", exc_info=True)
            yield ERROR_LINE
            spoken.append(ERROR_LINE)
            buffer = ""

        if final_content is not None:
            self._messages.append(
                {"role": "assistant", "content": final_content}
            )

        remainder = buffer.strip()
        if remainder:
            spoken.append(remainder)
            yield remainder

        if spoken:
            self._store.add_message(
                self._conversation_id, "assistant", " ".join(spoken)
            )
```

`src/zeus/brain/fake.py`:
```python
"""Conversation test double. Replays a script; never touches the network.

Can also invoke real tool callables, so an end-to-end test exercises the
genuine brain -> tool -> action log -> database path instead of writing the
data it then asserts on.
"""
from __future__ import annotations

from typing import Callable, Iterator

ToolCall = tuple[str, dict]


class FakeConversation:
    def __init__(
        self,
        script: dict[str, list[str]] | None = None,
        tools: dict[str, Callable] | None = None,
        tool_calls: list[list[ToolCall]] | None = None,
    ) -> None:
        self._script = script or {}
        self._tools = tools or {}
        self._tool_calls = list(tool_calls or [])
        self.sent: list[str] = []
        self.invoked: list[ToolCall] = []

    def send(self, text: str) -> Iterator[str]:
        turn = len(self.sent)
        self.sent.append(text)

        planned = self._tool_calls[turn] if turn < len(self._tool_calls) else []
        for name, kwargs in planned:
            tool = self._tools.get(name)
            if tool is None:
                raise KeyError(
                    f"FakeConversation was asked to call {name!r} but was given "
                    f"only {sorted(self._tools)}"
                )
            tool(**kwargs)
            self.invoked.append((name, kwargs))

        for sentence in self._script.get(text, ["Got it."]):
            yield sentence
```

`src/zeus/brain/__init__.py`:
```python
from zeus.brain.conversation import Conversation
from zeus.brain.fake import FakeConversation
from zeus.brain.prompts import (
    EVENING_OPENER,
    FOLDED_OPENER,
    MORNING_OPENER,
    SYSTEM_PROMPT,
)
from zeus.brain.tools import build_tools

__all__ = [
    "Conversation", "FakeConversation", "build_tools",
    "SYSTEM_PROMPT", "MORNING_OPENER", "EVENING_OPENER", "FOLDED_OPENER",
]
```

- [ ] **Step 6: Run the tests and verify they pass**

Run: `.venv/bin/pytest tests/brain/ -v`
Expected: PASS — 29 tests (27 above, plus two added in review:
`test_a_single_turn_can_span_several_tool_runner_rounds`, alongside the
StubClient fix so the multi-round path keeps coverage; and
`test_a_tool_using_turn_leaves_history_alternating`, which pins the Critical
— per-round appends produced consecutive assistant messages with a dangling
tool_use, which the API rejects with a 400 on the next send().)

- [ ] **Step 7: Commit**

```bash
git add src/zeus/brain/ tests/brain/
git commit -m "feat: Claude brain with Tool Runner, sentence streaming, action log"
```

---

### Task 15: Ritual orchestration

**Files:**
- Create: `src/zeus/ritual/checkin.py`
- Test: `tests/ritual/test_checkin.py`

**Interfaces:**
- Consumes: everything from Tasks 3–14.
- Produces:
  - `Notifier` protocol: `notify(title: str, body: str) -> None`; `MacNotifier` (via `osascript`), `FakeNotifier` (records).
  - `VoiceIO(activator, mic, endpointer, transcriber, speaker, config)` with:
    - `speak(sentences: Iterable[str]) -> None` — mutes the activator for the duration (half-duplex, spec §7.3).
    - `listen() -> str` — captures one utterance and transcribes it; `""` on silence.
  - `CheckIn(kind, store, journal, presence, voice, notifier, conversation_factory, config, tz, clock)` with `run(scheduled_for: datetime) -> Outcome`.
  - `local_date(dt, tz) -> str`

- [ ] **Step 1: Write the failing test**

`tests/ritual/test_checkin.py`:
```python
from datetime import datetime, timedelta, timezone

import pytest
from zoneinfo import ZoneInfo

from zeus.brain.fake import FakeConversation
from zeus.clock import FakeClock
from zeus.config import Config, ScheduleConfig
from zeus.context.presence import Signals, Verdict
from zeus.memory.journal import Journal
from zeus.memory.store import Store
from zeus.ritual.checkin import CheckIn, FakeNotifier, local_date
from zeus.ritual.retry import Outcome
from zeus.tts.fake import FakeSpeaker

LAGOS = ZoneInfo("Africa/Lagos")
NOW = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)  # 11:00 Lagos


class StubPresence:
    def __init__(self, verdict):
        self._verdict = verdict

    def verdict(self):
        return self._verdict


class StubVoice:
    """Stands in for VoiceIO: records what was spoken, replays what is heard."""

    def __init__(self, heard=None):
        self.spoken: list[str] = []
        self._heard = list(heard or [])

    def speak(self, sentences):
        self.spoken.extend(sentences)

    def listen(self):
        return self._heard.pop(0) if self._heard else ""


@pytest.fixture
def wiring(tmp_path):
    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    return clock, store, journal


def _checkin(kind, wiring, verdict, heard, script=None):
    clock, store, journal = wiring
    voice = StubVoice(heard)
    notifier = FakeNotifier()
    conversation = FakeConversation(script or {})
    return (
        CheckIn(
            kind=kind, store=store, journal=journal,
            presence=StubPresence(verdict), voice=voice, notifier=notifier,
            conversation_factory=lambda conv_id, local: conversation,
            config=ScheduleConfig(), tz=LAGOS, clock=clock,
        ),
        voice, notifier, store, conversation,
    )


def test_local_date_uses_the_local_zone():
    # 23:30 UTC is already the next day in Lagos (UTC+1)
    late = datetime(2026, 8, 5, 23, 30, tzinfo=timezone.utc)
    assert local_date(late, LAGOS) == "2026-08-06"


def test_speak_verdict_runs_the_conversation_and_records_the_answer(wiring):
    checkin, voice, _, store, conversation = _checkin(
        "morning", wiring, Verdict.SPEAK, heard=["Finish the auth flow"],
        script={"[morning check-in] Greet the user briefly and ask what the "
                "one thing is that has to happen today.": ["Morning.",
                                                           "What's the one thing?"]},
    )
    outcome = checkin.run(NOW)

    assert outcome is Outcome.ANSWERED
    assert voice.spoken[0] == "Morning."
    assert conversation.sent[1] == "Finish the auth flow"
    assert store.get_checkin(1).outcome == "answered"


def test_silence_produces_no_answer_and_a_retry(wiring):
    checkin, _, _, store, _ = _checkin("morning", wiring, Verdict.SPEAK, heard=[])
    outcome = checkin.run(NOW)

    assert outcome is Outcome.NO_ANSWER
    checkin_row = store.get_checkin(1)
    assert checkin_row.outcome == "no_answer"
    assert checkin_row.attempts == 1


def test_defer_verdict_never_speaks(wiring):
    checkin, voice, notifier, store, _ = _checkin(
        "morning", wiring, Verdict.DEFER, heard=["ignored"]
    )
    outcome = checkin.run(NOW)

    assert outcome is Outcome.DEFERRED
    assert voice.spoken == []
    assert notifier.sent == []


def test_notify_verdict_notifies_without_speaking(wiring):
    checkin, voice, notifier, _, _ = _checkin(
        "morning", wiring, Verdict.NOTIFY, heard=["ignored"]
    )
    outcome = checkin.run(NOW)

    assert outcome is Outcome.DEFERRED
    assert voice.spoken == []
    assert len(notifier.sent) == 1
    assert "ZEUS" in notifier.sent[0][0]


def test_evening_checkin_recalls_the_morning_goal(wiring):
    _, store, _ = wiring
    store.set_goal("2026-08-05", "Finish the auth flow")

    checkin, _, _, _, conversation = _checkin(
        "evening", wiring, Verdict.SPEAK, heard=["Mostly, tests are missing"]
    )
    checkin.run(NOW)

    assert "Finish the auth flow" in conversation.sent[0]


def test_evening_checkin_without_a_goal_uses_the_folded_opener(wiring):
    checkin, _, _, _, conversation = _checkin(
        "evening", wiring, Verdict.SPEAK, heard=["I worked on docs"]
    )
    checkin.run(NOW)

    assert "goal never captured" in conversation.sent[0]


def test_conversation_stops_at_three_exchanges(wiring):
    checkin, _, _, _, conversation = _checkin(
        "morning", wiring, Verdict.SPEAK,
        heard=["one", "two", "three", "four", "five"],
    )
    checkin.run(NOW)
    # opener + at most 3 user replies
    assert len(conversation.sent) <= 4


def test_attempts_accumulate_across_repeated_runs(wiring):
    checkin, _, _, store, _ = _checkin("morning", wiring, Verdict.DEFER, heard=[])
    checkin.run(NOW)
    checkin.run(NOW)
    assert store.get_checkin(1).attempts == 2


def test_journal_records_a_missed_checkin(wiring):
    _, _, journal = wiring
    checkin, _, _, _, _ = _checkin("evening", wiring, Verdict.SPEAK, heard=[])
    checkin.run(NOW)
    assert "no answer" in journal.read("2026-08-05").lower()
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv/bin/pytest tests/ritual/test_checkin.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.ritual.checkin'`

- [ ] **Step 3: Implement `src/zeus/ritual/checkin.py`**

```python
"""Check-in orchestration. See spec §7.2, §8, §9.3.

Wires the context gate, the retry state machine, the brain, and storage
into one run of the daily ritual.
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from typing import Callable, Iterable, Protocol
from zoneinfo import ZoneInfo

from zeus.audio.endpointer import capture_utterance
from zeus.audio.mic import FRAME_SAMPLES
from zeus.brain.prompts import EVENING_OPENER, FOLDED_OPENER, MORNING_OPENER
from zeus.clock import Clock
from zeus.config import ScheduleConfig
from zeus.context.presence import Verdict
from zeus.memory.journal import Journal
from zeus.memory.store import Store
from zeus.ritual.retry import Outcome, next_step

log = logging.getLogger(__name__)

MAX_EXCHANGES = 3


def local_date(moment: datetime, tz: ZoneInfo) -> str:
    """The calendar date in the user's zone — the key goals are stored under."""
    return moment.astimezone(tz).strftime("%Y-%m-%d")


# ---- notifications ----------------------------------------------------
class Notifier(Protocol):
    def notify(self, title: str, body: str) -> None: ...


class MacNotifier:
    def notify(self, title: str, body: str) -> None:
        script = f'display notification "{body}" with title "{title}"'
        try:
            subprocess.run(["osascript", "-e", script], timeout=5, check=False)
        except Exception:
            log.debug("notification failed", exc_info=True)


class FakeNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def notify(self, title: str, body: str) -> None:
        self.sent.append((title, body))


# ---- voice ------------------------------------------------------------
class VoiceIO:
    """Speak-then-listen, honouring the half-duplex rule (spec §7.3)."""

    def __init__(self, activator, mic, endpointer, transcriber, speaker, audio_config):
        self._activator = activator
        self._mic = mic
        self._endpointer = endpointer
        self._transcriber = transcriber
        self._speaker = speaker
        self._config = audio_config

    def speak(self, sentences: Iterable[str]) -> None:
        mute = getattr(self._activator, "mute", None)
        unmute = getattr(self._activator, "unmute", None)
        if mute:
            mute()
        try:
            for sentence in sentences:
                self._speaker.say(sentence)
        finally:
            if unmute:
                unmute()

    def listen(self) -> str:
        """Capture one utterance with the wake detector held muted.

        No drain() is needed any more: mic.frames() opens a FRESH
        subscription whose queue starts empty, so it cannot inherit the
        audio of ZEUS's own speech the way the old single shared queue
        could. That also fixes it for HotkeyActivator, which has no
        mute/unmute at all — the emptiness is structural now, not something
        an activator has to remember to do.

        The mute is the other half of the fan-out fix. Once every consumer
        gets its own copy of every frame, the detector no longer steals the
        user's answer — but it now HEARS all of it, and a "hey zeus" spoken
        mid-answer would launch an ad-hoc conversation on top of the running
        check-in. Muting for the whole listen window closes that. speak()
        mutes for its own duration; this covers the rest of the turn.
        """
        mute = getattr(self._activator, "mute", None)
        unmute = getattr(self._activator, "unmute", None)
        if mute:
            mute()
        try:
            frames_per_second = self._config.sample_rate / FRAME_SAMPLES
            timeout_frames = int(
                self._config.listen_timeout.total_seconds() * frames_per_second
            )
            audio = capture_utterance(
                self._mic.frames(), self._endpointer,
                pre_roll=b"", listen_timeout_frames=timeout_frames,
            )
        finally:
            if unmute:
                unmute()
        if not audio:
            return ""
        return self._transcriber.transcribe(audio, self._config.sample_rate)


# ---- the ritual -------------------------------------------------------
class CheckIn:
    def __init__(
        self, kind: str, store: Store, journal: Journal, presence, voice,
        notifier: Notifier, conversation_factory: Callable[[int, str], object],
        config: ScheduleConfig, tz: ZoneInfo, clock: Clock,
    ) -> None:
        self._kind = kind
        self._store = store
        self._journal = journal
        self._presence = presence
        self._voice = voice
        self._notifier = notifier
        self._conversation_factory = conversation_factory
        self._config = config
        self._tz = tz
        self._clock = clock

    def _opener(self, date: str) -> str:
        if self._kind == "morning":
            return MORNING_OPENER
        goal = self._store.get_goal(date)
        return EVENING_OPENER(goal.text) if goal else FOLDED_OPENER

    def _find_or_open(self, scheduled_for: datetime) -> int:
        """Reuse today's open check-in row so attempts accumulate across retries.

        Goes through Store.find_open_checkin rather than raw SQL. That method
        matches on the stored local_date, which matters twice: scheduled_for is
        stored as UTC (so a UTC-date match would miss an evening check-in in a
        western timezone), and the match is for THIS date only — an unresolved
        check-in left over from a previous day must not be reused, or today's
        first attempt would inherit yesterday's attempt count and could exhaust
        its retries before it has run once.
        """
        date = local_date(scheduled_for, self._tz)
        existing = self._store.find_open_checkin(self._kind, date)
        if existing is not None:
            return existing.id
        # Same `date` for the write as for the lookup. That identity is the
        # whole point — deriving it twice is exactly how they drifted apart.
        return self._store.open_checkin(self._kind, scheduled_for, date)

    def run(self, scheduled_for: datetime) -> Outcome:
        date = local_date(scheduled_for, self._tz)
        checkin_id = self._find_or_open(scheduled_for)
        previous = self._store.get_checkin(checkin_id).attempts
        verdict = self._presence.verdict()

        answered: bool | None = None

        if verdict is Verdict.NOTIFY:
            self._notifier.notify(
                "ZEUS", "Morning check-in" if self._kind == "morning"
                else "Evening check-in"
            )
        elif verdict is Verdict.SPEAK:
            answered = self._converse(checkin_id, date)

        decision = next_step(
            self._kind, verdict, answered, previous, self._config
        )
        self._store.update_checkin(
            checkin_id,
            outcome=decision.outcome.value,
            attempts=previous + 1,
            fired_at=self._clock.now_utc() if verdict is Verdict.SPEAK else None,
        )
        if decision.outcome is Outcome.NO_ANSWER:
            self._journal.append(f"{self._kind.title()} check-in: no answer")
        elif decision.outcome is Outcome.SKIPPED:
            self._journal.append(f"{self._kind.title()} check-in: skipped")
        return decision.outcome

    def _converse(self, checkin_id: int, date: str) -> bool:
        conversation_id = self._store.start_conversation("schedule")
        conversation = self._conversation_factory(conversation_id, date)
        try:
            self._voice.speak(conversation.send(self._opener(date)))
            heard_anything = False
            for _ in range(MAX_EXCHANGES):
                reply = self._voice.listen()
                if not reply:
                    break
                heard_anything = True
                self._voice.speak(conversation.send(reply))
            return heard_anything
        finally:
            self._store.end_conversation(conversation_id)
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `.venv/bin/pytest tests/ritual/test_checkin.py -v`
Expected: PASS — 16 tests (the Step 1 file collects 10; an earlier draft
said 9, miscounting. Plus 6 added in review: 4 covering `VoiceIO` directly,
which the Step 1 file never exercises — it drives `CheckIn` through a
`StubVoice` — and 2 more from the review round. The three that matter:
`speak()` must unmute in a `finally`, or a `say()` that raises leaves the
wake detector muted forever and ZEUS never wakes again until the daemon
restarts; a retry must find the SAME row when the local and UTC dates
differ, which requires a non-Lagos zone to express at all; and `listen()`
must discard audio buffered before it was called.)

- [ ] **Step 5: Commit**

```bash
git add src/zeus/ritual/checkin.py tests/ritual/test_checkin.py
git commit -m "feat: check-in orchestration wiring gate, retry, brain, and storage"
```

---

### Task 16: Daemon — wiring, self-test, catch-up policy, heartbeat

**Files:**
- Create: `src/zeus/daemon.py`
- Test: `tests/test_daemon.py`

**Interfaces:**
- Consumes: everything from Tasks 1–15.
- Produces:
  - `audio_self_test(mic, seconds: float = 1.0) -> bool` — captures briefly and asserts non-zero RMS. **This is the mitigation for risk R1.**
  - `catch_up_actions(missed: list[MissedRun]) -> list[tuple[str, str]]` — pure; applies spec §9.2, returning `(job_name, "fire" | "skip")`.
  - `Daemon(config, store, journal, scheduler, presence, voice, notifier, checkins, clock)` with:
    - `run_catch_up() -> list[tuple[str, str]]`
    - `tick() -> None` — one scheduler pass plus heartbeat
    - `run_forever() -> None`
    - `degraded: bool` — set when the self-test fails; check-ins then notify instead of speaking
  - `build_daemon(config) -> Daemon` — the real-component factory.

- [ ] **Step 1: Write the failing test**

`tests/test_daemon.py`:
```python
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from zoneinfo import ZoneInfo

from zeus.audio.mic import FRAME_SAMPLES, MicStream
from zeus.clock import FakeClock
from zeus.config import AudioConfig, Config
from zeus.daemon import Daemon, audio_self_test, catch_up_actions
from zeus.memory.journal import Journal
from zeus.memory.store import Store
from zeus.schedule.scheduler import MissedRun, Scheduler

LAGOS = ZoneInfo("Africa/Lagos")
NOW = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)

SILENCE = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()
SPEECH = (np.ones(FRAME_SAMPLES, dtype=np.int16) * 5000).tobytes()


def _mic(frames):
    mic = MicStream(AudioConfig())
    for frame in frames:
        mic._on_audio(frame, FRAME_SAMPLES, None, None)
    mic.stop()
    return mic


def test_self_test_passes_on_real_audio():
    assert audio_self_test(_mic([SPEECH] * 13), seconds=1.0) is True


def test_self_test_fails_on_pure_silence():
    """Risk R1: a TCC-denied stream opens fine and returns silence forever."""
    assert audio_self_test(_mic([SILENCE] * 13), seconds=1.0) is False


def test_self_test_fails_when_no_frames_arrive():
    assert audio_self_test(_mic([]), seconds=1.0) is False


@pytest.mark.parametrize(
    "missed,expected",
    [
        # Morning missed earlier today → still worth asking
        ([MissedRun("checkin_morning", NOW, True)], [("checkin_morning", "fire")]),
        # Morning missed on a previous day → asking now is noise
        ([MissedRun("checkin_morning", NOW, False)], [("checkin_morning", "skip")]),
        # Evening is never replayed, even on the same day
        ([MissedRun("checkin_evening", NOW, True)], [("checkin_evening", "skip")]),
        ([MissedRun("checkin_evening", NOW, False)], [("checkin_evening", "skip")]),
    ],
)
def test_catch_up_policy(missed, expected):
    assert catch_up_actions(missed) == expected


def test_catch_up_policy_preserves_order():
    missed = [
        MissedRun("checkin_morning", NOW - timedelta(days=1), False),
        MissedRun("checkin_evening", NOW - timedelta(days=1), False),
        MissedRun("checkin_morning", NOW, True),
    ]
    assert [action for _, action in catch_up_actions(missed)] == [
        "skip", "skip", "fire",
    ]


def test_unknown_job_is_never_fired_by_catch_up():
    assert catch_up_actions([MissedRun("watchman_scan", NOW, True)]) == [
        ("watchman_scan", "skip")
    ]


@pytest.fixture
def daemon(tmp_path):
    clock = FakeClock(NOW)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    scheduler = Scheduler(store, clock, LAGOS)
    fired: list[str] = []
    scheduler.register("checkin_morning", "0 11 * * *", lambda when: fired.append("m"))
    daemon = Daemon(
        config=Config(root=tmp_path), store=store, journal=journal,
        scheduler=scheduler, presence=None, voice=None, notifier=None,
        checkins={}, clock=clock,
    )
    return daemon, store, clock, fired


def test_tick_writes_a_heartbeat(daemon):
    instance, store, _, _ = daemon
    assert store.heartbeat() is None
    instance.tick()
    assert store.heartbeat() == NOW


def test_tick_advances_the_heartbeat(daemon):
    instance, store, clock, _ = daemon
    instance.tick()
    clock.advance(timedelta(minutes=5))
    instance.tick()
    assert store.heartbeat() == NOW + timedelta(minutes=5)


def test_degraded_flag_defaults_to_false(daemon):
    instance, _, _, _ = daemon
    assert instance.degraded is False


def test_run_catch_up_is_empty_without_a_heartbeat(daemon):
    instance, _, _, _ = daemon
    assert instance.run_catch_up() == []
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv/bin/pytest tests/test_daemon.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.daemon'`

- [ ] **Step 3: Implement `src/zeus/daemon.py`**

```python
"""The zeusd daemon: wiring, supervision, and the main loop. See spec §4, §9.2."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any

from zeus.audio.endpointer import Endpointer, rms
from zeus.audio.mic import FRAME_SAMPLES, MicStream
from zeus.clock import Clock, SystemClock, resolve_timezone
from zeus.config import Config, load_config
from zeus.context.presence import Verdict
from zeus.schedule.cron import hhmm_to_cron
from zeus.schedule.scheduler import MissedRun, Scheduler

log = logging.getLogger(__name__)

# Only the morning check-in is ever replayed, and only on the same local
# day. A goal question at 15:00 is useful; the same question at 09:00 the
# next morning is noise. See spec §9.2.
CATCH_UP_ELIGIBLE = {"checkin_morning"}


class DegradedPresence:
    """Presence adapter used when the microphone self-test failed.

    Speaking is pointless when the mic is dead: ZEUS would talk into a void,
    call listen() on it, hear nothing, and record NO_ANSWER — precisely the
    outcome audio_self_test exists to prevent (risk R1). So SPEAK becomes
    NOTIFY.

    DEFER passes through untouched. Being locked or idle still means defer,
    regardless of microphone health — a dead mic must not turn "the user is
    away" into "post a notification anyway".

    An earlier draft set Daemon.degraded and logged "check-ins will notify
    instead of speaking", but nothing read the flag on the check-in path:
    CheckIn chooses SPEAK from presence.verdict() alone, which knows nothing
    about microphone health. The mitigation was wired to nothing.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def verdict(self) -> Verdict:
        verdict = self._inner.verdict()
        return Verdict.NOTIFY if verdict is Verdict.SPEAK else verdict


class SwitchablePresence:
    """One level of indirection so the daemon can downgrade after startup.

    The self-test runs after the CheckIns are built, and CheckIn stores its
    presence at construction. Handing every CheckIn this wrapper means a
    single degrade() call reaches all of them — no rebuilding them, and no
    reaching into their private attributes from the daemon.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def verdict(self) -> Verdict:
        return self._inner.verdict()

    def degrade(self) -> None:
        self._inner = DegradedPresence(self._inner)


def audio_self_test(mic: MicStream, seconds: float = 1.0) -> bool:
    """Capture briefly and assert the microphone is actually producing audio.

    Risk R1: when macOS denies microphone access to a LaunchAgent-spawned
    process, the stream opens successfully and returns pure silence forever.
    Without this check ZEUS looks healthy while being completely deaf.
    """
    wanted = max(1, int(seconds * 16000 / FRAME_SAMPLES))
    energy = 0.0
    seen = 0
    done = threading.Event()

    # CONSUME ON A BACKGROUND THREAD. An in-loop deadline check cannot work
    # here, however natural it looks: `for frame in mic.frames()` only runs
    # the loop body when the generator YIELDS, and a microphone that has
    # gone silent mid-capture never yields again. The check would sit in
    # code that is never reached, Daemon.start() would never return, and the
    # tick loop would never begin. An earlier draft of this plan had exactly
    # that bug; the implementer caught it. Waiting on an Event is what makes
    # the deadline real even though the consuming thread itself may block
    # forever — the thread is a daemon thread, so it cannot hold up exit.
    def consume() -> None:
        nonlocal energy, seen
        for frame in mic.frames():
            energy = max(energy, rms(frame))
            seen += 1
            if seen >= wanted:
                break
        done.set()

    threading.Thread(target=consume, daemon=True).start()
    # Event.wait is backed by a monotonic clock, so no wall-clock adjustment
    # (an NTP correction, a DST transition) can defeat it. Budget generously
    # at 5s: this measures TOTAL capture, not time-to-first-frame, and a
    # Bluetooth input switching into its HFP profile can take seconds before
    # the first callback arrives. Too tight a budget makes a slow device
    # indistinguishable from a dead one — and the consequence is sticky,
    # because `degraded` is never re-tested or cleared, so ZEUS stays
    # notification-only until the process restarts.
    if not done.wait(timeout=max(seconds * 5.0, 5.0)):
        log.error(
            "audio self-test: timed out waiting for %d frames (got %d) — "
            "the microphone stopped delivering audio", wanted, seen,
        )
        return False
    if seen == 0:
        log.error("audio self-test: no frames received from the microphone")
        return False
    if energy <= 0.0:
        log.error(
            "audio self-test: %d frames captured but all silent — "
            "microphone permission is probably denied", seen,
        )
        return False
    return True


def catch_up_actions(missed: list[MissedRun]) -> list[tuple[str, str]]:
    """Apply the spec §9.2 replay policy to runs missed during downtime."""
    actions: list[tuple[str, str]] = []
    for run in missed:
        eligible = run.job in CATCH_UP_ELIGIBLE and run.same_local_day
        actions.append((run.job, "fire" if eligible else "skip"))
    return actions


class Daemon:
    def __init__(
        self, config: Config, store, journal, scheduler: Scheduler,
        presence, voice, notifier, checkins: dict[str, Any], clock: Clock,
        activator=None, mic=None,
    ) -> None:
        self._config = config
        self._store = store
        self._journal = journal
        self._scheduler = scheduler
        self._presence = presence
        self._voice = voice
        self._notifier = notifier
        self._checkins = checkins
        self._clock = clock
        self._activator = activator
        self._mic = mic
        self._running = False
        self.degraded = False

    def run_catch_up(self) -> list[tuple[str, str]]:
        missed = self._scheduler.catch_up()
        actions = catch_up_actions(missed)
        for run, (job, action) in zip(missed, actions):
            if action == "fire" and job in self._checkins:
                log.info("catch-up: firing %s missed at %s", job, run.scheduled_for)
                try:
                    self._checkins[job].run(run.scheduled_for)
                except Exception:
                    # Mirrors Scheduler.run_pending's per-handler isolation.
                    # Unguarded, one failing catch-up conversation kills the
                    # daemon before it ever reaches its tick loop.
                    log.exception("catch-up run for %r failed", job)
            else:
                log.info("catch-up: skipping %s missed at %s", job, run.scheduled_for)
            # Consume the occurrence either way — FIRED OR SKIPPED.
            # catch_up() reads the heartbeat while run_pending() reads
            # last_run_at; they are separate state. Without this the very
            # next tick() recomputes from a stale last_run_at and OVERRIDES
            # the decision just made: a "skip" gets fired anyway (spec §9.2
            # forbids asking about yesterday today), and a "fire" runs a
            # second time, either re-running _converse on the open row or
            # opening a second row and asking again.
            #
            # Only reproducible after a RESTART, when last_run_at already
            # exists — with a fresh store run_pending merely seeds the
            # baseline and fires nothing. That is why fixture fakes which
            # never populate last_run_at cannot catch this.
            self._store.set_job_run(job, run.scheduled_for)
        return actions

    def tick(self) -> None:
        now = self._clock.now_utc()
        self._scheduler.run_pending(now)
        self._store.set_heartbeat()

    def _activation_loop(self) -> None:
        if self._activator is None:
            return
        for event in self._activator.events():
            if not self._running:
                return
            log.info("activated via %s", event.source)
            try:
                self._handle_activation()
            except Exception:
                log.error("ad-hoc conversation failed", exc_info=True)

    def _handle_activation(self) -> None:
        """Ad-hoc conversation triggered by the wake word."""
        if self._voice is None:
            return
        heard = self._voice.listen()
        if not heard:
            return
        conversation_id = self._store.start_conversation("wake")
        try:
            conversation = self._checkins["_adhoc_factory"](conversation_id)
            self._voice.speak(conversation.send(heard))
        finally:
            self._store.end_conversation(conversation_id)

    def start(self) -> None:
        self._running = True
        if self._mic is not None:
            self._mic.start()
            if not audio_self_test(self._mic):
                self.degraded = True
                # Wire the flag to the behaviour it claims. Without this the
                # log line below is a lie: CheckIn picks SPEAK from
                # presence.verdict() alone and would still speak into a dead
                # microphone and record NO_ANSWER. Every CheckIn was handed
                # this same SwitchablePresence at construction, so one call
                # reaches all of them — no rebuild, no reaching into their
                # internals.
                self._presence.degrade()
                log.error(
                    "ZEUS is running in DEGRADED mode: no working microphone. "
                    "Check-ins will notify instead of speaking. "
                    "Run 'zeus doctor' for details."
                )
                if self._notifier is not None:
                    self._notifier.notify(
                        "ZEUS — microphone unavailable",
                        "Running in notification-only mode. Run 'zeus doctor'.",
                    )
        if self._activator is not None and not self.degraded:
            self._activator.start()
            threading.Thread(target=self._activation_loop, daemon=True).start()
        self.run_catch_up()

    def stop(self) -> None:
        self._running = False
        if self._activator is not None:
            self._activator.stop()
        if self._mic is not None:
            self._mic.stop()

    def run_forever(self) -> None:
        self.start()
        try:
            while self._running:
                self.tick()
                self._clock.sleep(
                    self._scheduler.seconds_until_next(self._clock.now_utc())
                )
        finally:
            self.stop()


def build_daemon(config: Config | None = None) -> Daemon:
    """Construct a daemon from real components. See spec §5 for the wiring."""
    import anthropic

    from zeus.audio.activator import build_activator
    from zeus.brain.conversation import Conversation
    from zeus.brain.prompts import SYSTEM_PROMPT
    from zeus.brain.tools import build_tools
    from zeus.memory.journal import Journal
    from zeus.memory.store import Store
    from zeus.ritual.checkin import CheckIn, MacNotifier, VoiceIO, local_date
    from zeus.stt import build_transcriber
    from zeus.tts import build_speaker

    config = config or load_config()
    config.log_path.parent.mkdir(parents=True, exist_ok=True)

    clock = SystemClock()
    tz = resolve_timezone(config.schedule.timezone)
    store = Store(config.db_path, clock)
    journal = Journal(config.journal_dir, clock, tz)

    mic = MicStream(config.audio)
    activator = build_activator(config.wake, mic)
    voice = VoiceIO(
        activator, mic, Endpointer(config.audio),
        build_transcriber(config.stt, config.models_dir),
        build_speaker(config.tts), config.audio,
    )
    notifier = MacNotifier()
    client = anthropic.Anthropic()

    def conversation_factory(conversation_id: int, date: str):
        return Conversation(
            client=client, config=config.brain, store=store, journal=journal,
            conversation_id=conversation_id, system=SYSTEM_PROMPT,
            tools=build_tools(store, journal, conversation_id, date),
        )

    def adhoc_factory(conversation_id: int):
        date = local_date(clock.now_utc(), tz)
        return Conversation(
            client=client, config=config.brain, store=store, journal=journal,
            conversation_id=conversation_id, system=SYSTEM_PROMPT,
            tools=build_tools(store, journal, conversation_id, date),
            effort=config.brain.effort_adhoc,
        )

    from zeus.context.presence import Presence

    # Wrapped so a failed self-test can downgrade every CheckIn at once.
    # The CheckIns below are built before start() runs the self-test, and
    # each stores its presence at construction — this indirection is what
    # lets Daemon.start() reach them afterwards.
    presence = SwitchablePresence(Presence(config.context))
    scheduler = Scheduler(store, clock, tz)

    checkins: dict[str, Any] = {"_adhoc_factory": adhoc_factory}
    for kind, hhmm in (
        ("morning", config.schedule.morning),
        ("evening", config.schedule.evening),
    ):
        name = f"checkin_{kind}"
        checkins[name] = CheckIn(
            kind=kind, store=store, journal=journal, presence=presence,
            voice=voice, notifier=notifier,
            conversation_factory=conversation_factory,
            config=config.schedule, tz=tz, clock=clock,
        )
        scheduler.register(
            name, hhmm_to_cron(hhmm),
            (lambda n: lambda when: checkins[n].run(when))(name),
        )

    return Daemon(
        config=config, store=store, journal=journal, scheduler=scheduler,
        presence=presence, voice=voice, notifier=notifier, checkins=checkins,
        clock=clock, activator=activator, mic=mic,
    )
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `.venv/bin/pytest tests/test_daemon.py -v`
Expected: PASS — 19 tests (13 above, plus 6 added in review):
- shutdown must stop the MIC, not just the activator —
  `WakeWordActivator.events()` checks `_running` only AFTER `mic.frames()`
  yields, so stopping the activator alone never terminates it;
- a **skipped** catch-up must not be re-fired by the next `tick()`, and a
  **fired** one must not run twice. Both need a RESTART setup —
  `store.set_job_run(...)` before the run — or they pass vacuously, since a
  fresh store makes `run_pending` merely seed the baseline;
- a failed self-test must make check-ins NOTIFY instead of speak (assert
  the notifier fired and the speaker said nothing — asserting the flag
  proves nothing), and `DegradedPresence` must pass DEFER through unchanged;
- `audio_self_test` must return `False` within a bounded wait when the mic
  goes silent mid-capture, rather than hanging `Daemon.start()` forever.

- [ ] **Step 5: Commit**

```bash
git add src/zeus/daemon.py tests/test_daemon.py
git commit -m "feat: daemon with audio self-test, catch-up policy, and heartbeat"
```

---

### Task 17: CLI and LaunchAgent installer

**Files:**
- Create: `src/zeus/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `build_daemon`, `load_config`, presence probes.
- Produces:
  - `launch_agent_plist(python_path: Path, log_path: Path, env_path: Path) -> str`
  - `_load_env_file(path: Path) -> int` — KEY=VALUE into os.environ; existing environment wins
  - `cmd_doctor(config) -> int` — prints an environment report; exit 0 healthy, 1 unhealthy.
  - `cmd_selftest(config) -> int` — the **only** code path that touches real hardware.
  - `cmd_install_agent(config) -> int`, `cmd_run(config) -> int`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
import os
import plistlib
import sys
from pathlib import Path

from zeus.cli import launch_agent_plist, main


ENV = Path("/Users/x/.zeus/env")


def test_plist_is_valid_and_uses_absolute_paths():
    xml = launch_agent_plist(
        Path("/opt/zeus/.venv/bin/python"), Path("/tmp/z.log"), ENV
    )
    parsed = plistlib.loads(xml.encode())

    assert parsed["Label"] == "com.zeus.daemon"
    assert parsed["ProgramArguments"][0] == "/opt/zeus/.venv/bin/python"
    assert all(Path(a).is_absolute() for a in parsed["ProgramArguments"][:1])


def test_plist_enables_keepalive_and_runatload():
    parsed = plistlib.loads(
        launch_agent_plist(Path("/x/python"), Path("/tmp/z.log"), ENV).encode()
    )
    assert parsed["KeepAlive"] is True
    assert parsed["RunAtLoad"] is True


def test_plist_routes_both_streams_to_the_log():
    parsed = plistlib.loads(
        launch_agent_plist(Path("/x/python"), Path("/tmp/z.log"), ENV).encode()
    )
    assert parsed["StandardOutPath"] == "/tmp/z.log"
    assert parsed["StandardErrorPath"] == "/tmp/z.log"


def test_plist_invokes_the_module_not_a_shell():
    parsed = plistlib.loads(
        launch_agent_plist(Path("/x/python"), Path("/tmp/z.log"), ENV).encode()
    )
    assert parsed["ProgramArguments"][1:] == ["-m", "zeus.cli", "run"]


def test_plist_points_at_the_env_file_and_never_holds_the_key():
    """launchd inherits no shell environment, so the daemon needs SOME way
    to find the key — but the spec says the key is environment-only and
    never written to config, and this plist is config. The path is what the
    plist may carry; the secret is not."""
    xml = launch_agent_plist(Path("/x/python"), Path("/tmp/z.log"), ENV)
    parsed = plistlib.loads(xml.encode())
    assert parsed["EnvironmentVariables"]["ZEUS_ENV_FILE"] == str(ENV)
    assert "ANTHROPIC_API_KEY" not in xml
    assert "sk-ant" not in xml


def test_env_file_is_loaded_into_the_environment(monkeypatch, tmp_path):
    from zeus.cli import _load_env_file

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / "env"
    env.write_text(
        "# a comment\n"
        "\n"
        "ANTHROPIC_API_KEY=sk-ant-test-value\n"
        'QUOTED="quoted-value"\n'
    )
    assert _load_env_file(env) == 2
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test-value"
    assert os.environ["QUOTED"] == "quoted-value"


def test_env_file_does_not_override_the_real_environment(monkeypatch, tmp_path):
    """A key exported in the shell wins over the file, so running the daemon
    by hand behaves the same as launchd loading it."""
    from zeus.cli import _load_env_file

    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-shell")
    env = tmp_path / "env"
    env.write_text("ANTHROPIC_API_KEY=from-the-file\n")
    assert _load_env_file(env) == 0
    assert os.environ["ANTHROPIC_API_KEY"] == "from-the-shell"


def test_a_missing_env_file_is_not_an_error(tmp_path):
    from zeus.cli import _load_env_file

    assert _load_env_file(tmp_path / "nope") == 0


def test_main_with_no_arguments_prints_usage(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_main_rejects_an_unknown_command(capsys):
    """RETURNS 2 — it does not raise. argparse calls parser.error() on an
    invalid subparser choice, which raises SystemExit(2); main() catches it
    so its declared `main(argv) -> int` contract holds on every path."""
    assert main(["frobnicate"]) == 2


def test_doctor_reports_and_returns_a_status(monkeypatch, tmp_path, capsys):
    import zeus.cli as cli

    monkeypatch.setattr(cli, "_probe_say", lambda: True)
    monkeypatch.setattr(cli, "_probe_afplay", lambda: True)
    monkeypatch.setattr(cli, "_probe_api_key", lambda: True)
    monkeypatch.setattr(cli, "_probe_transcriber", lambda: True)

    code = main(["doctor", "--root", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 0
    assert "say" in output and "ANTHROPIC_API_KEY" in output


def test_doctor_accepts_any_supported_python(monkeypatch, tmp_path, capsys):
    """Global Constraints say 3.11+, so doctor must not report a FAILURE on
    3.11 or 3.13 merely because it was developed on 3.12."""
    import zeus.cli as cli

    for probe in ("_probe_say", "_probe_afplay", "_probe_api_key",
                  "_probe_transcriber"):
        monkeypatch.setattr(cli, probe, lambda: True)
    assert sys.version_info[:2] >= (3, 11)
    assert main(["doctor", "--root", str(tmp_path)]) == 0


def test_doctor_fails_when_the_transcriber_is_missing(monkeypatch, tmp_path, capsys):
    """A broken transcriber is silent at runtime — LocalWhisper.transcribe()
    returns "" for every failure — so doctor is where it must surface."""
    import zeus.cli as cli

    monkeypatch.setattr(cli, "_probe_say", lambda: True)
    monkeypatch.setattr(cli, "_probe_afplay", lambda: True)
    monkeypatch.setattr(cli, "_probe_api_key", lambda: True)
    monkeypatch.setattr(cli, "_probe_transcriber", lambda: False)

    assert main(["doctor", "--root", str(tmp_path)]) == 1
    assert "transcriber" in capsys.readouterr().out


def test_doctor_fails_when_the_api_key_is_missing(monkeypatch, tmp_path, capsys):
    import zeus.cli as cli

    monkeypatch.setattr(cli, "_probe_say", lambda: True)
    monkeypatch.setattr(cli, "_probe_afplay", lambda: True)
    monkeypatch.setattr(cli, "_probe_transcriber", lambda: True)
    monkeypatch.setattr(cli, "_probe_api_key", lambda: False)

    assert main(["doctor", "--root", str(tmp_path)]) == 1
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'zeus.cli'`

- [ ] **Step 3: Implement `src/zeus/cli.py`**

```python
"""ZEUS command line. See spec §13 — `selftest` is the only hardware path."""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

from zeus.config import Config, load_config

log = logging.getLogger(__name__)

LABEL = "com.zeus.daemon"
PLIST_PATH = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"


def launch_agent_plist(python_path: Path, log_path: Path, env_path: Path) -> str:
    """Generate the LaunchAgent plist.

    The interpreter is referenced by absolute path so nothing depends on
    shell initialisation or PATH (spec §4.1).

    ENV_PATH, NOT THE KEY ITSELF. launchd does not read .env, and a
    LaunchAgent inherits none of your shell environment — so without this
    the daemon starts with no ANTHROPIC_API_KEY and every brain call fails
    at runtime, quietly. The obvious fix is an EnvironmentVariables entry
    holding the key, but the spec says the key lives in the environment
    only and is "never written to config or source", and a plist in
    ~/Library/LaunchAgents is config. So the plist carries only a PATH to
    the key file; cmd_run loads it (see _load_env_file). launchctl setenv
    was the other candidate and was rejected: it does not survive a reboot,
    so ZEUS would come back deaf to its own API after every restart.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>zeus.cli</string>
        <string>run</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>ZEUS_ENV_FILE</key><string>{env_path}</string>
    </dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>{log_path}</string>
    <key>StandardErrorPath</key><string>{log_path}</string>
</dict>
</plist>
"""


def _load_env_file(path: Path) -> int:
    """Load KEY=VALUE lines into os.environ. Returns how many were set.

    Ten lines of stdlib instead of python-dotenv: Global Constraints forbid
    new dependencies. Deliberately minimal — no interpolation, no `export`
    prefix, no multi-line values. It exists for one variable.

    Existing environment wins, so running the daemon by hand from a shell
    that already exported ANTHROPIC_API_KEY behaves the same as launchd
    loading it from the file.
    """
    if not path.exists():
        return 0
    loaded = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")
            loaded += 1
    return loaded


def _probe_say() -> bool:
    return Path("/usr/bin/say").exists()


def _probe_afplay() -> bool:
    return Path("/usr/bin/afplay").exists()


def _probe_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _probe_transcriber() -> bool:
    """Is the speech-to-text backend actually importable?

    LocalWhisper.transcribe() catches Exception and returns "", so a missing
    faster_whisper or a corrupt model is indistinguishable from a quiet room
    — ZEUS just never hears anything and says so to nobody. This is the
    cheap half of the check; cmd_selftest does the real one by transcribing
    audio it actually captured.
    """
    return importlib.util.find_spec("faster_whisper") is not None


def cmd_doctor(config: Config) -> int:
    # >= (3, 11), not == (3, 12): Global Constraints say Python 3.11+, so
    # exact equality would report a FAILURE on 3.11 or 3.13 while ZEUS runs
    # perfectly well on both. A doctor that lies about health is worse than
    # no doctor.
    checks = [
        ("python", f"{sys.version_info.major}.{sys.version_info.minor}",
         sys.version_info[:2] >= (3, 11)),
        ("say", "/usr/bin/say", _probe_say()),
        ("afplay", "/usr/bin/afplay", _probe_afplay()),
        ("ANTHROPIC_API_KEY", "environment", _probe_api_key()),
        ("transcriber", "faster_whisper", _probe_transcriber()),
        ("zeus root", str(config.root), config.root.exists()),
        ("database", str(config.db_path), config.db_path.exists()),
        ("LaunchAgent", str(PLIST_PATH), PLIST_PATH.exists()),
    ]
    print("ZEUS environment report\n")
    healthy = True
    for name, detail, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:<20} {detail}")
        # A missing database or LaunchAgent is expected before first run.
        if not ok and name in {
            "python", "say", "afplay", "ANTHROPIC_API_KEY", "transcriber",
        }:
            healthy = False
    print()
    if not healthy:
        print("Not ready. Fix the FAIL lines above.")
    return 0 if healthy else 1


def cmd_selftest(config: Config) -> int:
    """Capture, TRANSCRIBE, and speak. Requires real hardware — never in CI.

    The transcription step is the point, not a bonus. audio_self_test only
    proves frames are arriving with non-zero energy; it says nothing about
    whether those frames become words. LocalWhisper.transcribe() swallows
    every exception and returns "", so a missing model file, an
    uninstalled faster_whisper, or a corrupt download all present as ZEUS
    silently never understanding anything. Printing what came back is the
    only way a user can tell "you said nothing" from "I am broken".
    """
    from zeus.audio.endpointer import Endpointer, capture_utterance
    from zeus.audio.mic import FRAME_SAMPLES, MicStream
    from zeus.daemon import audio_self_test
    from zeus.stt import build_transcriber
    from zeus.tts import build_speaker

    print("Capturing one second of audio — say something now...")
    mic = MicStream(config.audio)
    mic.start()
    try:
        if not audio_self_test(mic):
            print(
                "FAIL: the microphone produced no audio.\n"
                "  Grant microphone access to this interpreter in\n"
                "  System Preferences > Security & Privacy > Privacy > Microphone,\n"
                "  then run 'zeus selftest' again from Terminal."
            )
            return 1
        print("OK: microphone is producing audio.")

        print("Now say a short sentence, then stop...")
        frames_per_second = config.audio.sample_rate / FRAME_SAMPLES
        audio = capture_utterance(
            mic.frames(), Endpointer(config.audio), pre_roll=b"",
            listen_timeout_frames=int(10 * frames_per_second),
        )
    finally:
        mic.stop()

    if not audio:
        print("FAIL: captured no utterance — the endpointer heard only silence.")
        return 1
    transcriber = build_transcriber(config.stt, config.models_dir)
    heard = transcriber.transcribe(audio, config.audio.sample_rate)
    if not heard:
        print(
            "FAIL: audio was captured but transcription returned nothing.\n"
            "  The model may be missing or corrupt. Check that faster_whisper\n"
            f"  is installed and that {config.models_dir} holds the model."
        )
        return 1
    print(f'OK: transcription works — I heard "{heard}"')

    speaker = build_speaker(config.tts)
    speaker.say("ZEUS self test complete. I can hear you and you can hear me.")
    print("OK: speech synthesis worked.")
    return 0


def cmd_install_agent(config: Config) -> int:
    python_path = Path(sys.executable).resolve()
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(
        launch_agent_plist(python_path, config.log_path, config.env_path)
    )
    print(f"Wrote {PLIST_PATH}\n")
    print("Load it with:")
    print(f"  launchctl unload {PLIST_PATH} 2>/dev/null")
    print(f"  launchctl load {PLIST_PATH}\n")
    if not config.env_path.exists():
        print(
            f"BEFORE loading it, put your API key in {config.env_path}:\n"
            f"  printf 'ANTHROPIC_API_KEY=sk-ant-...\\n' > {config.env_path}\n"
            f"  chmod 600 {config.env_path}\n"
            "launchd inherits none of your shell environment, so without this\n"
            "file the daemon starts fine and then fails on every request.\n"
        )
    print(
        "IMPORTANT: run 'zeus selftest' from Terminal FIRST so macOS prompts\n"
        "for microphone access. A LaunchAgent that has never been granted\n"
        "access opens the stream successfully and hears only silence."
    )
    return 0


def cmd_run(config: Config) -> int:
    from zeus.daemon import build_daemon

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Under launchd the environment is empty but ZEUS_ENV_FILE points at the
    # key file (see launch_agent_plist). Run from a shell and the file is
    # usually absent while the variable is already exported — both paths end
    # with ANTHROPIC_API_KEY in os.environ, which is the only place the
    # Anthropic client reads it from.
    env_file = Path(os.environ.get("ZEUS_ENV_FILE", config.env_path))
    _load_env_file(env_file)
    if not _probe_api_key():
        log.error(
            "ANTHROPIC_API_KEY is not set and %s did not supply it. "
            "Every conversation will fail. Run 'zeus doctor'.", env_file,
        )
    build_daemon(config).run_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zeus", description="ZEUS voice assistant")
    parser.add_argument("--root", type=Path, default=None, help="ZEUS data directory")
    sub = parser.add_subparsers(dest="command")
    for name, help_text in [
        ("run", "run the daemon in the foreground"),
        ("selftest", "check the microphone and speakers (requires hardware)"),
        ("doctor", "print an environment report"),
        ("install-agent", "write the LaunchAgent plist"),
    ]:
        sub.add_parser(name, help=help_text)

    # argparse RAISES SystemExit(2) on an unknown subcommand rather than
    # returning — so without this, main() only sometimes returns an int and
    # `main(["frobnicate"]) == 2` is unreachable. Catching it here keeps the
    # declared `main(argv) -> int` contract true on every path, which is
    # what makes main() callable as a library function and testable without
    # pytest.raises.
    try:
        args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    except SystemExit as exit_request:
        return int(exit_request.code or 0)
    if not args.command:
        parser.print_usage()
        return 2

    config = load_config(root=args.root) if args.root else load_config()
    return {
        "run": cmd_run,
        "selftest": cmd_selftest,
        "doctor": cmd_doctor,
        "install-agent": cmd_install_agent,
    }[args.command](config)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: PASS — 15 tests

- [ ] **Step 5: Commit**

```bash
git add src/zeus/cli.py tests/test_cli.py
git commit -m "feat: CLI with doctor, selftest, and LaunchAgent installer"
```

---

### Task 18: End-to-end golden path

**Files:**
- Create: `tests/test_end_to_end.py`
- Modify: `README.md` (create)

**Interfaces:**
- Consumes: every component, exclusively through its fake where hardware or network is involved.
- Produces: no new source — this task proves the assembled spine and documents how to run it.

- [ ] **Step 1: Write the failing test**

`tests/test_end_to_end.py`:
```python
"""Full-spine golden path. No microphone, no speakers, no network."""
from datetime import datetime, timedelta, timezone

import pytest
from zoneinfo import ZoneInfo

from zeus.brain.fake import FakeConversation
from zeus.brain.tools import build_tool_callables
from zeus.clock import FakeClock
from zeus.config import Config, ScheduleConfig
from zeus.context.presence import Verdict
from zeus.daemon import Daemon
from zeus.memory.journal import Journal
from zeus.memory.store import Store
from zeus.ritual.checkin import CheckIn, FakeNotifier
from zeus.schedule.scheduler import Scheduler
from zeus.tts.fake import FakeSpeaker

LAGOS = ZoneInfo("Africa/Lagos")
MORNING = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)   # 11:00 Lagos
EVENING = datetime(2026, 8, 5, 20, 0, tzinfo=timezone.utc)   # 21:00 Lagos


class StubPresence:
    def __init__(self, verdict=Verdict.SPEAK):
        self.verdict_value = verdict

    def verdict(self):
        return self.verdict_value


class ScriptedVoice:
    def __init__(self):
        self.spoken: list[str] = []
        self.replies: list[str] = []

    def speak(self, sentences):
        self.spoken.extend(sentences)

    def listen(self):
        return self.replies.pop(0) if self.replies else ""


@pytest.fixture
def rig(tmp_path):
    clock = FakeClock(MORNING)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, LAGOS)
    presence = StubPresence()
    voice = ScriptedVoice()
    notifier = FakeNotifier()

    def make_checkin(kind, script=None, tool_calls=None):
        """Build a CheckIn whose fake brain drives the REAL tool callables.

        The conversation is faked (no network), but save_goal and
        record_outcome are the genuine action-logged implementations, so
        the assertions below check data the production code wrote.
        """

        def factory(conversation_id, date):
            return FakeConversation(
                script=script,
                tools=build_tool_callables(store, journal, conversation_id, date),
                tool_calls=tool_calls,
            )

        return CheckIn(
            kind=kind, store=store, journal=journal, presence=presence,
            voice=voice, notifier=notifier, conversation_factory=factory,
            config=ScheduleConfig(), tz=LAGOS, clock=clock,
        )

    return {
        "clock": clock, "store": store, "journal": journal,
        "presence": presence, "voice": voice, "notifier": notifier,
        "make_checkin": make_checkin, "tmp_path": tmp_path,
    }


def test_full_day_morning_goal_to_evening_review(rig):
    """The golden path: nothing in this test writes the data it asserts on.

    Turn 0 of each conversation is ZEUS's opener; turn 1 is its reply to the
    user's answer, which is where the real tool fires.
    """
    store, journal, voice, clock = (
        rig["store"], rig["journal"], rig["voice"], rig["clock"]
    )

    # --- 11:00 — morning check-in --------------------------------------
    voice.replies = ["Finish the auth flow"]
    morning = rig["make_checkin"](
        "morning",
        tool_calls=[[], [("save_goal", {"text": "Finish the auth flow"})]],
    )
    assert morning.run(MORNING).value == "answered"

    # Written by the real save_goal tool, via the real action-log wrapper.
    goal = store.get_goal("2026-08-05")
    assert goal.text == "Finish the auth flow"
    assert goal.status == "pending"
    assert "Finish the auth flow" in journal.read("2026-08-05")
    assert voice.spoken                     # ZEUS actually said something

    save_action = store.recent_actions()[0]
    assert save_action.tool == "save_goal"
    assert save_action.ok is True

    # --- 21:00 — evening check-in --------------------------------------
    clock.advance(EVENING - MORNING)
    voice.replies = ["Mostly, the tests are still missing"]
    evening = rig["make_checkin"](
        "evening",
        tool_calls=[
            [],
            [("record_outcome", {"status": "partial", "notes": "tests missing"})],
        ],
    )
    assert evening.run(EVENING).value == "answered"

    goal = store.get_goal("2026-08-05")
    assert goal.status == "partial"
    assert goal.notes == "tests missing"
    assert goal.reviewed_at is not None

    assert [a.tool for a in store.recent_actions()] == [
        "record_outcome", "save_goal",      # recent_actions is newest-first
    ]


def test_away_all_morning_then_present_defers_then_speaks(rig):
    store, presence, voice = rig["store"], rig["presence"], rig["voice"]
    checkin = rig["make_checkin"]("morning")

    presence.verdict_value = Verdict.DEFER
    assert checkin.run(MORNING).value == "deferred"
    assert voice.spoken == []

    presence.verdict_value = Verdict.SPEAK
    voice.replies = ["Ship the parser"]
    assert checkin.run(MORNING).value == "answered"

    # Same check-in row reused, so attempts accumulated
    assert store.get_checkin(1).attempts == 2


def test_silence_all_day_never_loses_the_checkin_row(rig):
    store = rig["store"]
    checkin = rig["make_checkin"]("morning")

    assert checkin.run(MORNING).value == "no_answer"
    assert checkin.run(MORNING).value == "no_answer"   # retry exhausted → folds

    row = store.get_checkin(1)
    assert row.attempts == 2
    assert row.outcome == "no_answer"


def test_downtime_across_two_days_replays_only_today(rig):
    store, clock = rig["store"], rig["clock"]
    fired: list[str] = []

    scheduler = Scheduler(store, clock, LAGOS)
    scheduler.register("checkin_morning", "0 11 * * *", fired.append)
    scheduler.register("checkin_evening", "0 21 * * *", fired.append)

    # Heartbeat two days ago, now 13:00 Lagos today.
    clock.advance(timedelta(days=-2))
    store.set_heartbeat()
    clock.advance(timedelta(days=2) + timedelta(hours=2))

    daemon = Daemon(
        config=Config(root=rig["tmp_path"]), store=store, journal=rig["journal"],
        scheduler=scheduler, presence=rig["presence"], voice=rig["voice"],
        notifier=rig["notifier"], checkins={}, clock=clock,
    )
    actions = daemon.run_catch_up()

    fire_count = sum(1 for _, action in actions if action == "fire")
    assert fire_count == 1                      # only today's morning
    assert all(job == "checkin_morning" for job, action in actions if action == "fire")


def test_every_tool_call_is_visible_to_a_future_dashboard(rig):
    """Slice 2's dashboard can only show what Slice 1 recorded."""
    store, journal = rig["store"], rig["journal"]

    conv = store.start_conversation("schedule")
    tools = build_tool_callables(store, journal, conv, "2026-08-05")
    tools["save_goal"](text="Finish the auth flow")
    tools["record_outcome"](status="done")

    actions = store.recent_actions()
    assert {a.tool for a in actions} == {"save_goal", "record_outcome"}
    assert all(a.ok for a in actions)
    assert all(a.duration_ms >= 0 for a in actions)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv/bin/pytest tests/test_end_to_end.py -v`
Expected: FAIL — assertion errors or import errors, depending on what remains incomplete.

- [ ] **Step 3: Fix whatever the end-to-end test exposes**

No new modules should be required. If a test fails, the defect is in the wiring from Tasks 15–16, not in this test. Fix the source, not the assertion.

- [ ] **Step 4: Run the entire suite**

Run: `.venv/bin/pytest -v`
Expected: PASS — all tests across all 18 tasks green.

- [ ] **Step 5: Write `README.md`**

```markdown
# ZEUS

A voice assistant that runs on your Mac, wakes when you speak to it, and asks
what your one goal is each morning and whether it happened each evening.

## Requirements

macOS 12+ on Intel or Apple Silicon, and [uv](https://docs.astral.sh/uv/).
No Homebrew required.

## Install

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## First run — in this order

Run the self-test **from Terminal first**. macOS attributes microphone
permission to the process that asks, and a background LaunchAgent that has
never been granted access will open the microphone successfully and hear
nothing but silence.

```bash
.venv/bin/zeus doctor      # environment report
.venv/bin/zeus selftest    # grants microphone access, verifies audio in and out
.venv/bin/zeus install-agent
launchctl load ~/Library/LaunchAgents/com.zeus.daemon.plist
```

## Usage

Say **"hey jarvis"** to start a conversation. (openWakeWord ships no "zeus"
model; see `[wake] model` in `~/.zeus/config.toml` to swap in a custom one.)

ZEUS asks for your goal at 11:00 and reviews it at 21:00. If you are away,
on a call, or in a Focus mode, it defers or notifies quietly instead of
talking at you.

## Your data

Everything lives in `~/.zeus/` — delete that directory and ZEUS knows
nothing. Audio is never written to disk; only transcripts are stored, for
`transcript_retention_days` (default 90). Your transcribed text goes to the
Anthropic API; your audio does not, because transcription runs locally.

## Tests

```bash
.venv/bin/pytest
```

No test requires a microphone, speakers, or a network connection.
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_end_to_end.py README.md
git commit -m "test: end-to-end golden path across the assembled spine"
```

---

## Plan Self-Review

Run after the plan is written, before execution begins.

### Spec coverage

| Spec section | Covered by |
|---|---|
| §4.1 process model, LaunchAgent | T16, T17 |
| §4.2 runtime isolation (uv, 3.12) | T1, Global Constraints |
| §4.3 downtime recovery / catch-up | T6 (`catch_up`), T16 (`catch_up_actions`) |
| §4.4 data flow | T11–T13 (audio), T14 (brain), T15 (wiring) |
| §5 components and interfaces | T9, T10, T13 (protocols + two impls each) |
| §5.2 MicStream ring buffer | T11 |
| §6.1 WAL mode | T3 |
| §6.2 schema (all 8 tables) | T3 |
| §6.3 UTC storage / local scheduling | T2, T3, T5 |
| §6.3 audio never persisted | T12 (`capture_utterance` returns bytes, never writes) |
| §7.1 model configuration | T14 |
| §7.1 sentence streaming | T14 (`split_sentences`) |
| §7.2 check-in scripts, 3-exchange ceiling | T14 (prompts), T15 (`MAX_EXCHANGES`) |
| §7.3 half-duplex | T13 (`mute`/`unmute`), T15 (`VoiceIO.speak`) |
| §8 context gate | T7 |
| §9.1 generic cron jobs | T5, T6 |
| §9.2 catch-up policy | T16 |
| §9.3 retry paths | T8 |
| §10 failure handling | T7, T9, T10, T14, T16 |
| §10.1 startup self-test | T16 |
| §11 configuration | T1 |
| §12 security and privacy | T1 (retention), T14 (key from env), T17 (README) |
| §13 testing | every task; T18 for the golden path |
| §15 R1 microphone/TCC | T16 self-test, T17 selftest command + README ordering |
| §15 R2 no "zeus" model | T13 docstring, config `[wake] model`, README |
| §15 R3 Focus detection unverified | T7 Step 5 — **manual verification is an explicit step** |
| §15 R5 wake-word false triggers | T13 (`HotkeyActivator`, configurable threshold) |

No gaps.

### Placeholder scan

No `TBD`, no "add appropriate error handling", no "similar to Task N". Every
code step contains runnable code. Every test step contains real assertions.

### Type consistency

Verified across task boundaries:

- `Outcome` values (`answered`/`no_answer`/`deferred`/`skipped`) match the
  `checkins.outcome` CHECK constraint in T3 — asserted directly by
  `test_outcome_values_match_the_database_constraint` in T8.
- `Verdict` produced by T7 is consumed by T8 and T15 under the same name.
- `MissedRun(job, scheduled_for, same_local_day)` produced by T6 is consumed
  by T16 with identical field names.
- `Speaker.say`, `Transcriber.transcribe`, `Activator.events` signatures are
  identical in protocol, real implementation, and fake.
- `FRAME_SAMPLES` is defined once in T11 and imported by T12, T13, T15, T16.
- `Store.log_action(...)` signature in T3 matches the call in T14's
  `logged_tool`.
- `build_tools(store, journal, conversation_id, local_date)` in T14 matches
  its call sites in T16's `build_daemon` and T18.

### Amendments made before execution

Two defects found in the pre-flight scan and fixed in this document before
Task 1 was dispatched:

1. **Task 18's golden path asserted on data the test itself wrote.** Because
   `FakeConversation` could not invoke tools, the end-to-end test called
   `store.set_goal(...)` and then asserted the goal existed — proving
   nothing. `FakeConversation` now accepts real tool callables and a
   per-turn `tool_calls` list, so the golden path exercises the genuine
   brain → tool → action log → database chain. Nothing in that test now
   writes the data it checks.
2. **Tests assumed `@beta_tool` preserves `__name__` and direct
   callability.** Several tests did `{t.__name__: t for t in build_tools(...)}`,
   which depends on undocumented SDK decorator behaviour. `build_tools` is
   now a thin wrapper over a new `build_tool_callables`, which returns the
   plain action-logged bodies keyed by name. Tests and `FakeConversation`
   use the callables; only the Tool Runner gets the decorated versions.

`.gitignore` also already exists (it ignores `.worktrees/` and
`.superpowers/`), so Task 1 verifies and extends it rather than creating it.

### Resolved during execution (was: known simplification)

`CheckIn._find_or_open` originally reused an open check-in row via raw SQL
against `store.connection`, reaching past the `Store` façade, with
`Store.find_open_checkin` deferred to Slice 2. Task 3's review found that the
deferral left `find_open_checkin` named in Task 3's Produces block but
implemented nowhere. It is now implemented in Task 3 and `_find_or_open` calls
it, which also retired two latent defects in the raw SQL: it compared a UTC
`date(scheduled_for)` against a local date, and its `<=` would reuse an
unresolved check-in from an earlier day.

### Naming collision, flagged deliberately

Two different classes are named `CheckIn`: the `Store` row dataclass in
`zeus.memory.store` (Task 3) and the ritual runner in `zeus.ritual` (Task 15).
They live in different modules and do not collide in practice: Task 15 never
imports the store's dataclass by name, it only receives instances back from
`store.get_checkin()` / `store.find_open_checkin()` and reads `.id` and
`.attempts`. Renaming is a Slice 2 concern; do not rename mid-slice, as the
store dataclass name is already committed and referenced across tasks.

---

## Execution

18 tasks, each ending in a green test run and a commit. Suggested checkpoints
for review: after **T8** (all pure logic complete), after **T13** (audio stack
complete), and after **T18** (spine assembled).

The first thing to do in T7 Step 5 is resolve risk R3 by toggling Focus
manually. Do not defer it — the context gate's correctness depends on it, and
it takes two minutes.

