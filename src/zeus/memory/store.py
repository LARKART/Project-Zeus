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
        self.connection = sqlite3.connect(db_path, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self.connection.executescript(_SCHEMA.read_text())

    def close(self) -> None:
        self.connection.close()

    def _now(self) -> str:
        return to_utc_iso(self._clock.now_utc())

    # ---- goals -------------------------------------------------------
    def set_goal(self, date: str, text: str) -> int:
        # RETURNING, not cur.lastrowid: last_insert_rowid() is only updated by a
        # real insert, so on the ON CONFLICT DO UPDATE branch lastrowid silently
        # returns the id of whatever row was last inserted on this connection.
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
        # COALESCE, matching update_checkin's fired_at: omitting the optional
        # notes arg must not erase notes written by an earlier review. Clearing
        # notes is still reachable -- set_goal's upsert resets them to NULL.
        self.connection.execute(
            "UPDATE goals SET status = ?, notes = COALESCE(?, notes), "
            "reviewed_at = ? WHERE id = ?",
            (status, notes, self._now(), goal_id),
        )

    # ---- check-ins ---------------------------------------------------
    def open_checkin(self, kind: str, scheduled_for: datetime, local_date: str) -> int:
        """Open a check-in row.

        local_date is passed in rather than derived from scheduled_for. An
        earlier version inferred it via scheduled_for.date(), relying on the
        caller to hand over a local-zone-aware datetime -- but the scheduler
        always produces UTC (cron.next_occurrence ends in .astimezone(utc)),
        so the row was written with the UTC date while find_open_checkin
        searched the local one. Every retry then opened a fresh row and
        attempts never advanced past 1. Keeping the key explicit makes the
        write and the lookup provably the same value.
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
        outcomes, while 'deferred' and 'no_answer' are both still eligible for
        a retry and so still count as open.
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
