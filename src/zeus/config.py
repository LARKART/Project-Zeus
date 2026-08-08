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
    # DECLARED, NOT ENFORCED. Nothing in src/ reads this field: no purge job
    # is registered with the scheduler, no `DELETE FROM messages` exists
    # anywhere. Slice 1 keeps transcripts indefinitely regardless of this
    # value. Enforcing it (a scheduled purge job) is Slice 2 work — do not
    # assume this number does anything until that lands. See README.md
    # "Your data".
    transcript_retention_days: int = 90


@dataclass
class McpConfig:
    """MCP servers ZEUS may call tools on. See zeus.mcp.registry.

    `servers` is a free-form table, not a dataclass, because its keys are
    server names the user invents. _apply's unknown-key check therefore
    cannot police the inside of it -- load_server_configs validates each
    entry instead, and skips a malformed one loudly rather than raising,
    since this is parsed at daemon startup under KeepAlive.
    """

    enabled: bool = True
    servers: dict = field(default_factory=dict)


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
    mcp: McpConfig = field(default_factory=McpConfig)

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
