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
