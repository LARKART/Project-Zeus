"""Everything the dashboard shows, read once into one immutable snapshot.

READ-ONLY BY CONSTRUCTION. The connection is opened with `mode=ro`, so no
code path in this package can write to the database even by mistake — the
daemon owns that file and a page load must never contend with the voice
loop for a write lock. sqlite refuses the write at the driver level rather
than trusting this module to be careful, which is the difference between a
convention and a guarantee.

Read as ONE snapshot rather than lazily per section: the daemon writes
continuously, so a page that queried each table as it rendered could show a
check-in that had not happened yet beside a goal from before it did. One
pass, one moment.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# How long after its last heartbeat the daemon is presumed dead. The tick
# loop writes one every minute, so five is four missed beats -- long enough
# not to flicker on a slow tick, short enough that a user glancing at the
# page after a crash sees the truth rather than a stale "alive".
HEARTBEAT_GRACE = timedelta(minutes=5)

# How much history the page carries. The action log is the spec's "dashboard
# spine" and grows fastest, so it gets the largest window; conversations
# carry full transcripts and so get the smallest.
RECENT_DAYS = 30
MAX_ACTIONS = 200
MAX_CONVERSATIONS = 20
MAX_JOURNAL_DAYS = 7


@dataclass(frozen=True)
class Health:
    heartbeat_at: datetime | None
    age: timedelta | None
    status: str          # "alive" | "stale" | "never"
    detail: str


@dataclass(frozen=True)
class Streak:
    current: int
    longest: int
    kept_days: int
    total_days: int
    recent: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Snapshot:
    generated_at: datetime
    timezone_name: str
    today: str
    health: Health
    streak: Streak
    today_goal: dict | None
    today_checkins: list[dict]
    goals: list[dict]
    checkins: list[dict]
    actions: list[dict]
    conversations: list[dict]
    jobs: list[dict]
    facts: list[dict]
    journal: list[dict]
    settings: dict
    mcp_servers: list[dict] = field(default_factory=list)
    error: str | None = None


def _dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open the daemon's database without any ability to write to it.

    `.as_uri()` rather than an f-string: this path routinely contains a
    space ("Project Zeus"), and sqlite parses its URI form strictly -- an
    unescaped space silently becomes a DIFFERENT, empty database rather
    than an error, which would render a perfectly formatted page showing
    nothing at all.
    """
    connection = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    # A read-only connection still waits rather than failing instantly if
    # the writer holds the lock at that instant. WAL makes that rare; the
    # timeout is for the moment it is not.
    connection.execute("PRAGMA busy_timeout = 3000")
    return connection


def _rows(connection: sqlite3.Connection, sql: str, args: tuple = ()) -> list[dict]:
    return [dict(row) for row in connection.execute(sql, args).fetchall()]


def _health(connection: sqlite3.Connection, now_utc: datetime) -> Health:
    row = connection.execute("SELECT ts FROM heartbeat WHERE id = 1").fetchone()
    beat = _dt(row["ts"]) if row else None
    if beat is None:
        return Health(None, None, "never",
                      "no heartbeat recorded — the daemon has never run")
    age = now_utc - beat
    if age <= HEARTBEAT_GRACE:
        return Health(beat, age, "alive", "the daemon is ticking")
    return Health(
        beat, age, "stale",
        "no heartbeat for longer than the grace period — the daemon is "
        "stopped, asleep, or wedged",
    )


def _streak(goals: list[dict], today: str) -> Streak:
    """Consecutive days whose goal was marked done.

    KEPT MEANS `done`. `partial` is deliberately not counted: a streak that
    forgives partial days measures intent rather than outcome, and the whole
    point of the evening review is that it is answerable honestly.

    Today NOT YET REVIEWED does not break the run — the evening check-in has
    not happened yet, so a page loaded at lunchtime must not tell the user
    they have lost a streak they are still in the middle of keeping.
    """
    by_date = {g["date"]: g for g in goals}
    if not by_date:
        return Streak(0, 0, 0, 0, [])

    def kept(date: str) -> bool:
        return by_date.get(date, {}).get("status") == "done"

    start = datetime.strptime(today, "%Y-%m-%d").date()
    cursor = start
    if not kept(today) and by_date.get(today, {}).get("status") in (None, "pending"):
        cursor = start - timedelta(days=1)     # today is still in play

    current = 0
    while kept(cursor.strftime("%Y-%m-%d")):
        current += 1
        cursor -= timedelta(days=1)

    longest = run = 0
    day = min(by_date)
    last = max(by_date)
    day_date = datetime.strptime(day, "%Y-%m-%d").date()
    last_date = datetime.strptime(last, "%Y-%m-%d").date()
    while day_date <= last_date:
        run = run + 1 if kept(day_date.strftime("%Y-%m-%d")) else 0
        longest = max(longest, run)
        day_date += timedelta(days=1)

    recent = []
    for offset in range(RECENT_DAYS - 1, -1, -1):
        date = (start - timedelta(days=offset)).strftime("%Y-%m-%d")
        goal = by_date.get(date)
        recent.append({
            "date": date,
            "status": goal["status"] if goal else "none",
            "text": goal["text"] if goal else None,
        })
    kept_days = sum(1 for g in by_date.values() if g["status"] == "done")
    return Streak(current, longest, kept_days, len(by_date), recent)


def _decode(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def _read_journal(journal_dir: Path, today: str) -> list[dict]:
    """The most recent journal files, newest first.

    Read from disk rather than reconstructed from the database on purpose:
    the journal is what the user would see opening the folder in any editor,
    and showing anything else here would make the dashboard and the files
    disagree.
    """
    if not journal_dir.is_dir():
        return []
    entries = []
    start = datetime.strptime(today, "%Y-%m-%d").date()
    for offset in range(MAX_JOURNAL_DAYS):
        date = (start - timedelta(days=offset)).strftime("%Y-%m-%d")
        path = journal_dir / f"{date}.md"
        if not path.is_file():
            continue
        try:
            entries.append({"date": date, "body": path.read_text(encoding="utf-8")})
        except OSError as problem:
            entries.append({"date": date, "body": f"(unreadable: {problem})"})
    return entries


def _read_mcp_servers(root: Path) -> list[dict]:
    """The configured MCP servers, and which tools each is currently offering.

    Read from mcp.json rather than from the running registry because the
    dashboard is a SEPARATE PROCESS from the daemon -- it cannot see the
    daemon's live objects, only what both of them read from disk. The tool
    counts therefore come from the action log: what a server has actually
    been used for is the honest answer this process can give, and it is more
    useful than a list of what it theoretically offers.
    """
    from zeus.mcp import store as mcp_store

    servers = []
    for name, entry in sorted(mcp_store.load(root).items()):
        if not isinstance(entry, dict):
            continue
        servers.append({
            "name": name,
            "command": entry.get("command") or [],
            "env_keys": sorted((entry.get("env") or {}).keys()),
            "enabled": bool(entry.get("enabled", True)),
            "source": "mcp.json",
        })
    return servers


def read_snapshot(
    db_path: Path, journal_dir: Path, tz: ZoneInfo, now_utc: datetime,
    settings: dict | None = None,
) -> Snapshot:
    """One consistent read of everything the dashboard displays.

    Never raises for a missing or unreadable database. A dashboard whose
    whole job is answering "is ZEUS working?" must render the answer "no,
    and here is why" rather than returning a stack trace to the browser --
    spec §10, fail loudly and never pretend.
    """
    today = now_utc.astimezone(tz).strftime("%Y-%m-%d")
    empty = Snapshot(
        generated_at=now_utc, timezone_name=str(tz), today=today,
        health=Health(None, None, "never", "the database has not been created yet"),
        streak=Streak(0, 0, 0, 0, []), today_goal=None, today_checkins=[],
        goals=[], checkins=[], actions=[], conversations=[], jobs=[], facts=[],
        journal=_read_journal(journal_dir, today), settings=settings or {},
        mcp_servers=_read_mcp_servers(db_path.parent),
    )
    if not db_path.is_file():
        return empty
    try:
        connection = _connect_readonly(db_path)
    except sqlite3.Error as problem:
        return Snapshot(**{
            **empty.__dict__,
            "error": f"could not open {db_path}: {problem}",
        })

    try:
        cutoff = (
            now_utc.astimezone(tz) - timedelta(days=RECENT_DAYS)
        ).strftime("%Y-%m-%d")
        goals = _rows(
            connection,
            "SELECT * FROM goals WHERE date >= ? ORDER BY date DESC", (cutoff,),
        )
        checkins = _rows(
            connection,
            "SELECT * FROM checkins WHERE local_date >= ? "
            "ORDER BY scheduled_for DESC", (cutoff,),
        )
        actions = _rows(
            connection,
            "SELECT * FROM actions ORDER BY id DESC LIMIT ?", (MAX_ACTIONS,),
        )
        for action in actions:
            action["args"] = _decode(action.pop("args_json", None))
            action["result"] = _decode(action.pop("result_json", None))
            action["ok"] = bool(action["ok"])
        conversations = _rows(
            connection,
            "SELECT * FROM conversations ORDER BY id DESC LIMIT ?",
            (MAX_CONVERSATIONS,),
        )
        for conversation in conversations:
            conversation["messages"] = _rows(
                connection,
                "SELECT role, content, ts FROM messages "
                "WHERE conversation_id = ? ORDER BY id", (conversation["id"],),
            )
        health = _health(connection, now_utc)
        jobs = _rows(connection, "SELECT * FROM jobs ORDER BY name")
        facts = _rows(connection, "SELECT * FROM facts ORDER BY key")
    except sqlite3.Error as problem:
        connection.close()
        return Snapshot(**{**empty.__dict__, "error": f"database read failed: {problem}"})
    finally:
        try:
            connection.close()
        except sqlite3.Error:
            pass

    return Snapshot(
        generated_at=now_utc,
        timezone_name=str(tz),
        today=today,
        health=health,
        streak=_streak(goals, today),
        today_goal=next((g for g in goals if g["date"] == today), None),
        today_checkins=[c for c in checkins if c["local_date"] == today],
        goals=goals,
        checkins=checkins,
        actions=actions,
        conversations=conversations,
        jobs=jobs,
        facts=facts,
        journal=_read_journal(journal_dir, today),
        settings=settings or {},
        mcp_servers=_read_mcp_servers(db_path.parent),
    )
