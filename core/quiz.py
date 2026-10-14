"""Quiz flow: choosing a quiz for a student and stepping through it."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from .learner import MasteryModel, Result, diagnostic_set, select_questions
from .questions import Grade, Question
from .schema import Concept

MODES = {"practice": "Adaptive practice", "exam": "Mock exam", "diagnostic": "Diagnostic (new students)"}


def learner_id_from(name: str) -> str:
    """A stable id from the name a student types; empty when there is no usable name."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]


def build_quiz(mode: str, model: MasteryModel, bank: list[Question], concepts: list[Concept], n: int, now: float,
               topics: list[str] | None = None, asked: set[str] | None = None,
               rng: random.Random | None = None) -> list[Question]:
    """Questions for one sitting. Never repeats a question the student has already answered."""
    rng = rng or random.Random()
    asked = asked or set()
    scope = {c.id for c in concepts if not topics or c.topic in topics}
    unseen = [q for q in bank if q.concept_id in scope and q.id not in asked]
    if mode == "diagnostic":
        return diagnostic_set(unseen, [c for c in concepts if c.id in scope], per_topic=2, rng=rng)
    if mode == "exam":
        questions = select_questions(model, unseen, n, now, scope=scope, rng=rng, policy="random")
        rng.shuffle(questions)
        return questions
    return select_questions(model, unseen, n, now, scope=scope, rng=rng, policy="adaptive")


@dataclass
class Answer:
    response: str | int | None
    grade: Grade


@dataclass
class QuizSession:
    """Where a student is in a quiz. They can go back, skip and come back; answers, once given, are final.

    Practice shows feedback straight after each answer; an exam records the answer and moves on.
    index == len(questions) is the "ready to finish?" screen.
    """
    questions: list[Question]
    immediate_feedback: bool = True
    quiz_id: str = ""
    index: int = 0
    answers: dict[int, Answer] = field(default_factory=dict)
    finished: bool = False

    @property
    def current(self) -> Question | None:
        return self.questions[self.index] if self.index < len(self.questions) else None

    @property
    def at_review(self) -> bool:
        return not self.finished and self.index >= len(self.questions)

    @property
    def n_answered(self) -> int:
        return len(self.answers)

    @property
    def n_correct(self) -> int:
        return sum(a.grade.correct for a in self.answers.values())

    @property
    def unanswered(self) -> list[int]:
        return [i for i in range(len(self.questions)) if i not in self.answers]

    @property
    def results(self) -> list[Result]:
        return [Result(self.questions[i], a.response, a.grade.correct) for i, a in sorted(self.answers.items())]

    def answer_at(self, index: int) -> Answer | None:
        return self.answers.get(index)

    def submit(self, response: str | int | None, grade: Grade) -> None:
        if self.finished or self.current is None:
            raise ValueError("there is no question to answer")
        if self.index in self.answers:
            raise ValueError("this question has already been answered")
        self.answers[self.index] = Answer(response, grade)
        if not self.immediate_feedback:
            self.index += 1

    def next(self) -> None:
        self.index = min(self.index + 1, len(self.questions))

    def previous(self) -> None:
        self.index = max(self.index - 1, 0)

    def skip(self) -> None:
        if self.index in self.answers:
            raise ValueError("an answered question cannot be skipped")
        self.next()

    def go_to(self, index: int) -> None:
        self.index = max(0, min(index, len(self.questions)))

    def finish(self) -> None:
        self.finished = True