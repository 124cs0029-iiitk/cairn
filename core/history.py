"""Saved quizzes: starting one, recording each answer, finishing, listing and reviewing them."""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field

from .questions import Grade, Question


@dataclass
class QuizSummary:
    id: str
    mode: str
    created_at: float
    finished_at: float | None
    total: int
    answered: int
    correct: int

    @property
    def skipped(self) -> int:
        return self.total - self.answered

    @property
    def finished(self) -> bool:
        return self.finished_at is not None


@dataclass
class ReviewItem:
    position: int
    question_id: str
    question: Question | None            # None if the bank no longer holds the question
    answered: bool
    response: str | None = None
    correct: bool | None = None
    feedback: str = ""


@dataclass
class QuizReview:
    summary: QuizSummary
    items: list[ReviewItem] = field(default_factory=list)


def start_quiz(conn: sqlite3.Connection, learner_id: str, mode: str, questions: list[Question], now: float) -> str:
    quiz_id = uuid.uuid4().hex
    conn.execute("INSERT INTO quizzes (id, learner_id, mode, created_at, question_ids) VALUES (?, ?, ?, ?, ?)",
                 (quiz_id, learner_id, mode, now, json.dumps([q.id for q in questions])))
    conn.commit()
    return quiz_id


def save_answer(conn: sqlite3.Connection, quiz_id: str, position: int, question: Question,
                response: str | int | None, grade: Grade, now: float) -> None:
    conn.execute("INSERT OR REPLACE INTO quiz_answers (quiz_id, position, question_id, response, correct, feedback, "
                 "answered_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (quiz_id, position, question.id, None if response is None else str(response), int(grade.correct),
                  grade.feedback, now))
    conn.commit()


def finish_quiz(conn: sqlite3.Connection, quiz_id: str, now: float) -> None:
    conn.execute("UPDATE quizzes SET finished_at = ? WHERE id = ?", (now, quiz_id))
    conn.commit()


_SUMMARY_SQL = """
SELECT q.id, q.mode, q.created_at, q.finished_at, q.question_ids,
       COUNT(a.position) AS answered, COALESCE(SUM(a.correct), 0) AS correct
FROM quizzes q LEFT JOIN quiz_answers a ON a.quiz_id = q.id
WHERE {where} GROUP BY q.id ORDER BY q.created_at DESC
"""


def _summary(row: sqlite3.Row) -> QuizSummary:
    return QuizSummary(row["id"], row["mode"], row["created_at"], row["finished_at"],
                       len(json.loads(row["question_ids"])), row["answered"], row["correct"])


def list_quizzes(conn: sqlite3.Connection, learner_id: str) -> list[QuizSummary]:
    """A learner's quizzes, newest first. Quizzes with no answers at all are not worth listing."""
    rows = conn.execute(_SUMMARY_SQL.format(where="q.learner_id = ?"), (learner_id,)).fetchall()
    return [_summary(r) for r in rows if r["answered"] > 0 or r["finished_at"] is not None]


def load_review(conn: sqlite3.Connection, quiz_id: str, bank: dict[str, Question]) -> QuizReview | None:
    row = conn.execute(_SUMMARY_SQL.format(where="q.id = ?"), (quiz_id,)).fetchone()
    if row is None:
        return None
    answers = {r["position"]: r for r in conn.execute("SELECT * FROM quiz_answers WHERE quiz_id = ?", (quiz_id,))}
    items = []
    for position, qid in enumerate(json.loads(row["question_ids"])):
        a = answers.get(position)
        items.append(ReviewItem(position, qid, bank.get(qid), a is not None,
                                a["response"] if a else None, bool(a["correct"]) if a else None,
                                a["feedback"] if a else ""))
    return QuizReview(_summary(row), items)