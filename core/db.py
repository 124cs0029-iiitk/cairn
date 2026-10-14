"""SQLite schema and helpers for learners, questions, attempts and mastery."""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS learners (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  REAL NOT NULL
);

-- One row per learner and concept: current belief that the concept is known.
CREATE TABLE IF NOT EXISTS mastery (
    learner_id     TEXT NOT NULL REFERENCES learners(id),
    concept_id     TEXT NOT NULL,
    p_known        REAL NOT NULL,
    n_obs          INTEGER NOT NULL DEFAULT 0,
    half_life_days REAL NOT NULL DEFAULT 2.0,   -- forgetting-curve half-life
    updated_at     REAL NOT NULL,
    PRIMARY KEY (learner_id, concept_id)
);

-- Verified questions. payload is the full question JSON (stem, options, key, explanation).
CREATE TABLE IF NOT EXISTS questions (
    id          TEXT PRIMARY KEY,
    concept_id  TEXT NOT NULL,
    chunk_id    TEXT NOT NULL,
    qtype       TEXT NOT NULL CHECK (qtype IN ('mcq','short','numeric')),
    difficulty  INTEGER NOT NULL CHECK (difficulty BETWEEN 1 AND 5),
    payload     TEXT NOT NULL,
    embedding   BLOB,                            -- for novelty checks
    verified    INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS attempts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id  TEXT NOT NULL REFERENCES learners(id),
    question_id TEXT NOT NULL REFERENCES questions(id),
    concept_id  TEXT NOT NULL,
    correct     INTEGER NOT NULL,
    response    TEXT,
    session_id  TEXT,
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_learner ON attempts(learner_id, ts);

-- Weak evidence from tutor conversations (question asked, concept touched).
CREATE TABLE IF NOT EXISTS chat_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id  TEXT NOT NULL REFERENCES learners(id),
    concept_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    ts          REAL NOT NULL
);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    p = Path(path) if path else get_settings().db_path
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(path: Path | str | None = None) -> sqlite3.Connection:
    conn = connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


@contextmanager
def session(path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    conn = init_db(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_or_create_learner(conn: sqlite3.Connection, learner_id: str, name: str = "") -> sqlite3.Row:
    row = conn.execute("SELECT * FROM learners WHERE id = ?", (learner_id,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO learners (id, name, created_at) VALUES (?, ?, ?)",
            (learner_id, name or learner_id, time.time()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM learners WHERE id = ?", (learner_id,)).fetchone()
    return row
