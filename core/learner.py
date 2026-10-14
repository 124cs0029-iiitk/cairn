"""Learner model: per-concept mastery (Bayesian Knowledge Tracing) with forgetting, question selection
and the post-assessment report.

Mastery of a concept is the probability the student knows it. Each answer updates it with the BKT rule,
where harder questions are harder to guess and slightly easier to slip on. Between sessions mastery
decays along a forgetting curve whose half-life grows with every correct answer. A new student starts
from a prior that depends on how well their prerequisite concepts are known.
"""
from __future__ import annotations

import math
import random
import sqlite3
from dataclasses import dataclass, field

from .questions import Question
from .schema import Concept

P_INIT = 0.25                 # prior for a concept with no prerequisites
P_LEARN = 0.12                # chance of learning from one practice item
GUESS = {"mcq": 0.25, "short": 0.08}
DAY = 86400.0
MAX_HALF_LIFE = 60.0          # days
MIN_RETENTION = 0.35


@dataclass
class ConceptState:
    p_known: float
    n_obs: int = 0
    half_life_days: float = 1.5
    last_seen: float | None = None          # unix seconds


def bkt_update(p: float, correct: bool, guess: float, slip: float, learn: float) -> float:
    if correct:
        posterior = p * (1 - slip) / (p * (1 - slip) + (1 - p) * guess)
    else:
        posterior = p * slip / (p * slip + (1 - p) * (1 - guess))
    return posterior + (1 - posterior) * learn


def item_parameters(q: Question) -> tuple[float, float]:
    """(guess, slip) for a question: hard items are harder to guess, slightly easier to slip on."""
    guess = GUESS[q.qtype] * (1.15 - 0.15 * q.difficulty)
    slip = 0.06 + 0.02 * (q.difficulty - 1)
    return guess, slip


class MasteryModel:
    def __init__(self, concepts: list[Concept], states: dict[str, ConceptState] | None = None):
        self.concepts = {c.id: c for c in concepts}
        self.states: dict[str, ConceptState] = dict(states or {})

    def _prior(self, cid: str, now: float) -> float:
        prereqs = [p for p in self.concepts[cid].prerequisites if p in self.concepts]
        if not prereqs:
            return P_INIT
        return 0.10 + 0.30 * sum(self.mastery(p, now) for p in prereqs) / len(prereqs)

    def state(self, cid: str, now: float) -> ConceptState:
        if cid not in self.states:
            self.states[cid] = ConceptState(p_known=self._prior(cid, now))
        return self.states[cid]

    def retention(self, cid: str, now: float) -> float:
        s = self.states.get(cid)
        if s is None or s.last_seen is None:
            return 1.0
        days = max(0.0, (now - s.last_seen) / DAY)
        return max(MIN_RETENTION, 2 ** (-days / s.half_life_days))

    def mastery(self, cid: str, now: float) -> float:
        """Probability the student knows the concept right now (after forgetting)."""
        s = self.states.get(cid)
        return self._prior(cid, now) if s is None else s.p_known * self.retention(cid, now)

    def observe_answer(self, q: Question, correct: bool, now: float) -> float:
        s = self.state(q.concept_id, now)
        prior = s.p_known * self.retention(q.concept_id, now)
        guess, slip = item_parameters(q)
        s.p_known = bkt_update(prior, correct, guess, slip, P_LEARN)
        s.n_obs += 1
        s.half_life_days = min(MAX_HALF_LIFE, s.half_life_days * 1.8) if correct else max(1.0, s.half_life_days * 0.6)
        s.last_seen = now
        return s.p_known

    def observe_chat(self, concept_ids: list[str], now: float) -> None:
        """Asking the tutor about a concept is weak evidence of not knowing it yet (plus a little learning)."""
        for cid in concept_ids:
            if cid in self.concepts:
                s = self.state(cid, now)
                s.p_known = bkt_update(s.p_known * self.retention(cid, now), False, 0.8, 0.2, 0.04)
                s.last_seen = now

    def topic_mastery(self, now: float) -> dict[str, float]:
        by_topic: dict[str, list[float]] = {}
        for c in self.concepts.values():
            by_topic.setdefault(c.topic, []).append(self.mastery(c.id, now))
        return {t: sum(v) / len(v) for t, v in by_topic.items()}

    def ranked(self, now: float, scope: set[str] | None = None) -> list[tuple[str, float, int]]:
        """(concept id, mastery now, answers so far), weakest first."""
        rows = [(cid, self.mastery(cid, now), self.states[cid].n_obs if cid in self.states else 0)
                for cid in self.concepts if scope is None or cid in scope]
        return sorted(rows, key=lambda r: r[1])


# ---- choosing questions -------------------------------------------------------------------------

def select_questions(model: MasteryModel, bank: list[Question], n: int, now: float,
                     scope: set[str] | None = None, asked: frozenset[str] | set[str] = frozenset(),
                     rng: random.Random | None = None, policy: str = "adaptive",
                     per_concept: int = 2) -> list[Question]:
    """Pick n verified questions the student has not seen, inside the chosen scope.

    adaptive: favours weak concepts, questions whose difficulty fits the current mastery, concepts due for
    review, and concepts whose prerequisites are known. random: the baseline used in the evaluation.
    """
    rng = rng or random.Random()
    pool = [q for q in bank if q.verified and q.id not in asked and (scope is None or q.concept_id in scope)]
    if policy == "random":
        rng.shuffle(pool)
        return pool[:n]
    scored = []
    for q in pool:
        m = model.mastery(q.concept_id, now)
        fit = math.exp(-((q.difficulty - (1 + 4 * m)) ** 2) / 2)           # strong students get harder items
        due = 1.5 if model.retention(q.concept_id, now) < 0.6 else 1.0
        prereqs = model.concepts[q.concept_id].prerequisites if q.concept_id in model.concepts else []
        shaky = any(model.mastery(p, now) < 0.3 for p in prereqs if p in model.concepts)
        score = (1 - m + 0.05) * fit * due * (0.6 if shaky and m < 0.5 else 1.0) * (1 + 0.1 * rng.random())
        scored.append((score, q))
    scored.sort(key=lambda item: -item[0])
    chosen: list[Question] = []
    counts: dict[str, int] = {}
    for _, q in scored:
        if counts.get(q.concept_id, 0) < per_concept:
            chosen.append(q)
            counts[q.concept_id] = counts.get(q.concept_id, 0) + 1
            if len(chosen) == n:
                break
    return chosen


def diagnostic_set(bank: list[Question], concepts: list[Concept], per_topic: int = 2,
                   rng: random.Random | None = None) -> list[Question]:
    """Cold start for a new student: a few questions of middling difficulty from every topic."""
    rng = rng or random.Random(0)
    topics: dict[str, list[Concept]] = {}
    for c in concepts:
        topics.setdefault(c.topic, []).append(c)
    chosen: list[Question] = []
    for topic_concepts in topics.values():
        order = topic_concepts[:]
        rng.shuffle(order)
        picked = 0
        for concept in order:
            options = [q for q in bank if q.verified and q.concept_id == concept.id]
            if options and picked < per_topic:
                chosen.append(min(options, key=lambda q: (abs(q.difficulty - 2.5), q.id)))
                picked += 1
    return chosen


# ---- the post-assessment report -----------------------------------------------------------------

@dataclass
class Result:
    question: Question
    response: str | int | None
    correct: bool


@dataclass
class Report:
    score: tuple[int, int]
    by_topic: dict[str, tuple[int, int]] = field(default_factory=dict)
    weak_concepts: list[str] = field(default_factory=list)
    misconceptions: list[dict] = field(default_factory=list)
    review_next: list[str] = field(default_factory=list)


def assessment_report(results: list[Result], model: MasteryModel, now: float) -> Report:
    by_topic: dict[str, list[int]] = {}
    missed: dict[str, int] = {}
    misconceptions = []
    for r in results:
        concept = model.concepts.get(r.question.concept_id)
        topic = concept.topic if concept else "Other"
        tally = by_topic.setdefault(topic, [0, 0])
        tally[1] += 1
        if r.correct:
            tally[0] += 1
            continue
        missed[r.question.concept_id] = missed.get(r.question.concept_id, 0) + 1
        if r.question.qtype == "mcq" and isinstance(r.response, int) and r.response in range(4):
            misconceptions.append({"concept": concept.name if concept else r.question.concept_id,
                                   "question": r.question.stem,
                                   "chose": r.question.options[r.response], "correct": r.question.answer})
    weak = sorted(missed, key=lambda cid: -missed[cid])
    asked = {r.question.concept_id for r in results}
    review = [model.concepts[cid].name for cid, _, _ in model.ranked(now, asked) if cid in model.concepts][:3]
    return Report(score=(sum(r.correct for r in results), len(results)),
                  by_topic={t: (v[0], v[1]) for t, v in by_topic.items()},
                  weak_concepts=[model.concepts[c].name if c in model.concepts else c for c in weak],
                  misconceptions=misconceptions, review_next=review)


# ---- saving and loading (SQLite) ----------------------------------------------------------------

def load_model(conn: sqlite3.Connection, learner_id: str, concepts: list[Concept]) -> MasteryModel:
    rows = conn.execute("SELECT concept_id, p_known, n_obs, half_life_days, updated_at FROM mastery "
                        "WHERE learner_id = ?", (learner_id,)).fetchall()
    states = {r["concept_id"]: ConceptState(r["p_known"], r["n_obs"], r["half_life_days"], r["updated_at"])
              for r in rows}
    return MasteryModel(concepts, states)


def save_model(conn: sqlite3.Connection, learner_id: str, model: MasteryModel) -> None:
    for cid, s in model.states.items():
        conn.execute(
            "INSERT INTO mastery (learner_id, concept_id, p_known, n_obs, half_life_days, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(learner_id, concept_id) DO UPDATE SET "
            "p_known = excluded.p_known, n_obs = excluded.n_obs, half_life_days = excluded.half_life_days, "
            "updated_at = excluded.updated_at",
            (learner_id, cid, s.p_known, s.n_obs, s.half_life_days, s.last_seen or 0.0))
    conn.commit()


def log_attempt(conn: sqlite3.Connection, learner_id: str, q: Question, correct: bool,
                response: str | int | None, session_id: str, now: float) -> None:
    conn.execute("INSERT INTO attempts (learner_id, question_id, concept_id, correct, response, session_id, ts) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (learner_id, q.id, q.concept_id, int(correct), None if response is None else str(response),
                  session_id, now))
    conn.commit()


def asked_ids(conn: sqlite3.Connection, learner_id: str) -> set[str]:
    return {r["question_id"] for r in conn.execute(
        "SELECT DISTINCT question_id FROM attempts WHERE learner_id = ?", (learner_id,))}


def log_chat(conn: sqlite3.Connection, learner_id: str, concept_ids: list[str], now: float) -> None:
    conn.executemany("INSERT INTO chat_events (learner_id, concept_id, kind, ts) VALUES (?, ?, 'question', ?)",
                     [(learner_id, cid, now) for cid in concept_ids])
    conn.commit()