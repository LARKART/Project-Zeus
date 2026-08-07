"""SQLite persistence. See spec §6."""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from zeus.clock import Clock, from_utc_iso, to_utc_iso

_SCHEMA = Path(__file__).with_name("schema.sql")


def _dt(value: str | None) -> datetime | None:
    return from_utc_iso(value) if value else None


def _checkin_row(row: Any) -> "CheckIn":
    """Build a CheckIn from a `SELECT *` row.

    A module-level function, not a Store method: Store's lock is plain and
    non-reentrant, so a Store method that calls another Store method would
    deadlock rather than raise.
    """
    return CheckIn(
        id=row["id"], kind=row["kind"], local_date=row["local_date"],
        scheduled_for=from_utc_iso(row["scheduled_for"]),
        fired_at=_dt(row["fired_at"]), outcome=row["outcome"],
        attempts=row["attempts"], retry_at=_dt(row["retry_at"]),
        notified=bool(row["notified"]),
    )


@dataclass
class Goal:
    id: int
    date: str
    text: str
    status: str
    set_at: datetime
    reviewed_at: datetime | None
    notes: str | None


# A default of None would be indistinguishable from "clear the retry", which
# is a meaningful instruction here -- see update_checkin.
_UNSET: Any = object()


@dataclass
class CheckIn:
    id: int
    kind: str
    local_date: str
    scheduled_for: datetime
    fired_at: datetime | None
    outcome: str
    attempts: int
    retry_at: datetime | None = None
    # Has macOS already been notified about THIS check-in? One notification
    # per check-in, however many rungs the §9.3 ladder walks -- see
    # CheckIn.run(). Set once and never cleared; the row is the scope.
    notified: bool = False


@dataclass(frozen=True)
class DueRetry:
    """A check-in whose §9.3 retry has come due.

    Carries scheduled_for, not `now`: a retry at 11:20 belongs to the 11:00
    occurrence, so re-running it must use the ORIGINAL scheduled time or the
    ritual computes a fresh local_date and opens a second row.
    """

    id: int
    kind: str
    scheduled_for: datetime


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
        #
        # The lock is a plain, non-reentrant Lock, which is safe only
        # because no Store method calls another Store method; keep it that
        # way, or a nested call will deadlock rather than raise.
        self.connection = sqlite3.connect(
            db_path, isolation_level=None, check_same_thread=False
        )
        self._lock = threading.Lock()
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self.connection.executescript(_SCHEMA.read_text())
        self._migrate()

    def _migrate(self) -> None:
        """Additive schema migrations for databases already on disk.

        schema.sql runs through executescript with CREATE TABLE IF NOT
        EXISTS, which is a no-op against an existing table -- it will NOT
        add a column to it. A database that predates a column therefore
        never gains it from schema.sql alone, and this one is already on
        someone's disk. Each step is guarded so it is safe to run on every
        startup.
        """
        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(checkins)")
        }
        if "retry_at" not in columns:
            self.connection.execute("ALTER TABLE checkins ADD COLUMN retry_at TEXT")
        if "notified" not in columns:
            self.connection.execute(
                "ALTER TABLE checkins ADD COLUMN notified INTEGER NOT NULL DEFAULT 0"
            )

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def _now(self) -> str:
        return to_utc_iso(self._clock.now_utc())

    # ---- goals -------------------------------------------------------
    def set_goal(self, date: str, text: str) -> int:
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            cur = self.connection.execute(
                "INSERT INTO checkins (kind, local_date, scheduled_for) VALUES (?, ?, ?)",
                (kind, local_date, to_utc_iso(scheduled_for)),
            )
            return int(cur.lastrowid)

    def get_checkin(self, checkin_id: int) -> CheckIn:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM checkins WHERE id = ?", (checkin_id,)
            ).fetchone()
            return _checkin_row(row)

    def find_open_checkin(self, kind: str, date: str) -> CheckIn | None:
        """The unresolved check-in of this kind on this local date, if any.

        "Open" means not yet terminal: 'answered' and 'skipped' are settled
        outcomes, while 'deferred' and 'no_answer' are both still eligible for
        a retry and so still count as open.
        """
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM checkins WHERE kind = ? AND local_date = ? "
                "AND outcome NOT IN ('answered','skipped') "
                "ORDER BY id DESC LIMIT 1",
                (kind, date),
            ).fetchone()
            if row is None:
                return None
            return _checkin_row(row)

    def update_checkin(
        self, checkin_id: int, *, outcome: str, attempts: int,
        fired_at: datetime | None = None, retry_at: datetime | None = _UNSET,
    ) -> None:
        """Record the result of one check-in attempt.

        retry_at defaults to a SENTINEL, not to None. None is a meaningful
        value here -- it clears a pending §9.3 retry when the sequence
        terminates -- so a plain `= None` default would silently wipe the
        retry on every call that does not mention it, including every call
        site written before the retry ladder existed. Omitting the argument
        leaves whatever retry is already on the row untouched.
        """
        fired = to_utc_iso(fired_at) if fired_at else None
        with self._lock:
            if retry_at is _UNSET:
                self.connection.execute(
                    "UPDATE checkins SET outcome = ?, attempts = ?, "
                    "fired_at = COALESCE(?, fired_at) WHERE id = ?",
                    (outcome, attempts, fired, checkin_id),
                )
            else:
                self.connection.execute(
                    "UPDATE checkins SET outcome = ?, attempts = ?, "
                    "fired_at = COALESCE(?, fired_at), retry_at = ? WHERE id = ?",
                    (outcome, attempts, fired,
                     to_utc_iso(retry_at) if retry_at is not None else None,
                     checkin_id),
                )

    def mark_notified(self, checkin_id: int) -> None:
        """Record that macOS has been notified about this check-in.

        A method of its own rather than another optional argument on
        update_checkin: `notified` is only ever set, never cleared, so it
        needs none of retry_at's sentinel machinery -- and adding a third
        optional column there would have doubled that method's branches
        again.
        """
        with self._lock:
            self.connection.execute(
                "UPDATE checkins SET notified = 1 WHERE id = ?", (checkin_id,)
            )

    def due_retries(self, now_utc: datetime) -> list[DueRetry]:
        """Check-ins whose retry time has arrived, oldest first."""
        with self._lock:
            rows = self.connection.execute(
                "SELECT id, kind, scheduled_for FROM checkins "
                "WHERE retry_at IS NOT NULL AND retry_at <= ? "
                "ORDER BY retry_at",
                (to_utc_iso(now_utc),),
            ).fetchall()
        return [
            DueRetry(int(r["id"]), r["kind"], from_utc_iso(r["scheduled_for"]))
            for r in rows
        ]

    # ---- actions -----------------------------------------------------
    def log_action(
        self, tool: str, args: dict[str, Any], result: Any, ok: bool,
        duration_ms: int, error: str | None = None,
        conversation_id: int | None = None,
    ) -> int:
        with self._lock:
            cur = self.connection.execute(
                "INSERT INTO actions (ts, conversation_id, tool, args_json, "
                "result_json, ok, duration_ms, error) VALUES (?,?,?,?,?,?,?,?)",
                (self._now(), conversation_id, tool, json.dumps(args),
                 json.dumps(result) if result is not None else None,
                 int(ok), duration_ms, error),
            )
            return int(cur.lastrowid)

    def recent_actions(self, limit: int = 50) -> list[Action]:
        with self._lock:
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
        with self._lock:
            cur = self.connection.execute(
                "INSERT INTO conversations (started_at, trigger) VALUES (?, ?)",
                (self._now(), trigger),
            )
            return int(cur.lastrowid)

    def add_message(self, conversation_id: int, role: str, content: str) -> None:
        with self._lock:
            self.connection.execute(
                "INSERT INTO messages (conversation_id, role, content, ts) "
                "VALUES (?,?,?,?)",
                (conversation_id, role, content, self._now()),
            )

    def messages(self, conversation_id: int) -> list[Message]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT role, content, ts FROM messages WHERE conversation_id = ? "
                "ORDER BY id", (conversation_id,)
            ).fetchall()
            return [Message(r["role"], r["content"], from_utc_iso(r["ts"])) for r in rows]

    def end_conversation(self, conversation_id: int) -> None:
        with self._lock:
            self.connection.execute(
                "UPDATE conversations SET ended_at = ? WHERE id = ?",
                (self._now(), conversation_id),
            )

    # ---- facts -------------------------------------------------------
    def set_fact(self, key: str, value: str, source: str) -> None:
        with self._lock:
            self.connection.execute(
                "INSERT INTO facts (key, value, learned_at, source) VALUES (?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "learned_at = excluded.learned_at, source = excluded.source",
                (key, value, self._now(), source),
            )

    def get_fact(self, key: str) -> str | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT value FROM facts WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    # ---- jobs and heartbeat ------------------------------------------
    def upsert_job(self, name: str, schedule: str) -> None:
        with self._lock:
            self.connection.execute(
                "INSERT INTO jobs (name, schedule) VALUES (?, ?) "
                "ON CONFLICT(name) DO UPDATE SET schedule = excluded.schedule",
                (name, schedule),
            )

    def jobs(self) -> list[Job]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM jobs WHERE enabled = 1 ORDER BY name"
            ).fetchall()
            return [
                Job(r["name"], r["schedule"], _dt(r["last_run_at"]),
                    _dt(r["next_run_at"]), bool(r["enabled"]))
                for r in rows
            ]

    def set_job_run(self, name: str, last_run_at: datetime) -> None:
        with self._lock:
            self.connection.execute(
                "UPDATE jobs SET last_run_at = ? WHERE name = ?",
                (to_utc_iso(last_run_at), name),
            )

    def set_heartbeat(self) -> None:
        with self._lock:
            self.connection.execute(
                "INSERT INTO heartbeat (id, ts) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET ts = excluded.ts",
                (self._now(),),
            )

    def heartbeat(self) -> datetime | None:
        with self._lock:
            row = self.connection.execute("SELECT ts FROM heartbeat WHERE id = 1").fetchone()
            return from_utc_iso(row["ts"]) if row else None
