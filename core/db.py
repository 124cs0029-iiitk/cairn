"""SQLite schema and helpers for learners, mastery, attempts and tutor-chat evidence."""
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

-- One row per learner and concept: the current belief that the concept is known.
CREATE TABLE IF NOT EXISTS mastery (
    learner_id     TEXT NOT NULL REFERENCES learners(id),
    concept_id     TEXT NOT NULL,
    p_known        REAL NOT NULL,
    n_obs          INTEGER NOT NULL DEFAULT 0,
    half_life_days REAL NOT NULL DEFAULT 1.5,   -- forgetting-curve half-life
    updated_at     REAL NOT NULL,               -- when the concept was last practised
    PRIMARY KEY (learner_id, concept_id)
);

-- Every answered question (the questions themselves live in data/kb/questions.json).
CREATE TABLE IF NOT EXISTS attempts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id  TEXT NOT NULL REFERENCES learners(id),
    question_id TEXT NOT NULL,
    concept_id  TEXT NOT NULL,
    correct     INTEGER NOT NULL,
    response    TEXT,
    session_id  TEXT,
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_learner ON attempts(learner_id, ts);

-- One row per quiz sitting; question_ids is the ordered list of questions in it (JSON).
CREATE TABLE IF NOT EXISTS quizzes (
    id           TEXT PRIMARY KEY,
    learner_id   TEXT NOT NULL REFERENCES learners(id),
    mode         TEXT NOT NULL,
    created_at   REAL NOT NULL,
    finished_at  REAL,
    question_ids TEXT NOT NULL
);

-- The answer given to each question of a quiz (a skipped question has no row), for the review screen.
CREATE TABLE IF NOT EXISTS quiz_answers (
    quiz_id     TEXT NOT NULL REFERENCES quizzes(id),
    position    INTEGER NOT NULL,
    question_id TEXT NOT NULL,
    response    TEXT,
    correct     INTEGER NOT NULL,
    feedback    TEXT NOT NULL DEFAULT '',
    answered_at REAL NOT NULL,
    PRIMARY KEY (quiz_id, position)
);

-- Weak evidence from tutor conversations (a question asked about a concept).
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