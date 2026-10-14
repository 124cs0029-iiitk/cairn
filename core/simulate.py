"""Simulated students, used to evaluate the learner model and the question-selection policy.

Each simulated student has a hidden yes/no state of knowing every concept. They answer with their own
guess and slip rates (deliberately different from the ones the learner model assumes), learn from practice
(slowly, and less when a prerequisite is unknown), and forget between sittings. The same students are run
under each selection policy, so the comparison is paired.

What this can and cannot show: the estimation accuracy of the learner model (Brier score against the hidden
state) and the repetition rate do not depend on how the simulator makes students learn. The size of the
learning gains does, so those figures are a sanity check of the policy, not evidence about real learners.
"""
from __future__ import annotations

import copy
import math
import random
import statistics
import zlib
from dataclasses import dataclass, field

from .learner import MasteryModel, diagnostic_set, select_questions
from .questions import Question
from .schema import Concept

DAY = 86400.0
POLICIES = ("adaptive", "random", "random_no_history")
PROFILES = ("novice", "advanced", "gaps", "mixed")


@dataclass
class Student:
    known: dict[str, bool]
    practised_last: set[str] = field(default_factory=set)

    def p_correct(self, q: Question) -> float:
        if self.known.get(q.concept_id, False):
            return 1 - (0.10 + 0.03 * (q.difficulty - 1))              # slips more on hard questions
        return 0.25 if q.qtype == "mcq" else 0.05                      # guessing

    def answer(self, q: Question, rng: random.Random) -> bool:
        return rng.random() < self.p_correct(q)

    def learn(self, q: Question, prerequisites: list[str], rng: random.Random) -> None:
        if self.known.get(q.concept_id, False):
            return
        chance = 0.30 if q.difficulty <= 3 else 0.15
        if not all(self.known.get(p, True) for p in prerequisites):
            chance *= 0.4                                                # cannot build on missing foundations
        if rng.random() < chance:
            self.known[q.concept_id] = True

    def forget(self, rng: random.Random, probability: float = 0.08) -> None:
        for cid in list(self.known):
            if self.known[cid] and cid not in self.practised_last and rng.random() < probability:
                self.known[cid] = False
        self.practised_last = set()


def make_student(concepts: list[Concept], profile: str, rng: random.Random) -> Student:
    topics = list(dict.fromkeys(c.topic for c in concepts))
    weak_topics = set(rng.sample(topics, k=min(2, len(topics))))
    chance = {"novice": lambda c: 0.12, "advanced": lambda c: 0.70, "mixed": lambda c: rng.uniform(0.1, 0.9),
              "gaps": lambda c: 0.10 if c.topic in weak_topics else 0.75}[profile]
    return Student({c.id: rng.random() < chance(c) for c in concepts})


def known_fraction(student: Student) -> float:
    return sum(student.known.values()) / len(student.known)


def brier(model: MasteryModel, student: Student, now: float) -> float:
    """Mean squared gap between estimated mastery and the hidden truth (0 is perfect, 0.25 is guessing 50%)."""
    return statistics.fmean((model.mastery(cid, now) - float(k)) ** 2 for cid, k in student.known.items())


def choose(policy: str, model: MasteryModel, bank: list[Question], concepts: list[Concept], asked: set[str],
           n: int, now: float, first: bool, rng: random.Random) -> list[Question]:
    if policy == "random_no_history":
        return select_questions(model, bank, n, now, policy="random", rng=rng)
    unseen = [q for q in bank if q.id not in asked]
    if policy == "random":
        return select_questions(model, unseen, n, now, policy="random", rng=rng)
    if first:                                                            # cold start: diagnostic across topics
        picked = diagnostic_set(unseen, concepts, per_topic=2, rng=rng)[:n]
        if len(picked) < n:
            chosen = {q.id for q in picked}
            picked += select_questions(model, [q for q in unseen if q.id not in chosen], n - len(picked), now, rng=rng)
        return picked
    return select_questions(model, unseen, n, now, rng=rng)


def run_student(policy: str, start: Student, concepts: list[Concept], bank: list[Question], sessions: int,
                per_session: int, gap_days: float, seed: int, forget_probability: float = 0.25) -> dict:
    rng = random.Random(seed)
    student = copy.deepcopy(start)
    model = MasteryModel(concepts)
    prereqs = {c.id: c.prerequisites for c in concepts}
    asked: set[str] = set()
    t0 = 0.0
    out = {"initial_known": known_fraction(student), "brier": [brier(model, student, t0)],
           "known": [known_fraction(student)], "presented": 0, "repeated": 0, "targeted": 0,
           "unknown_at_start": [c for c, k in student.known.items() if not k]}
    seen_ever: set[str] = set()
    for s in range(sessions):
        now = t0 + s * gap_days * DAY
        for q in choose(policy, model, bank, concepts, asked, per_session, now, s == 0, rng):
            out["presented"] += 1
            out["repeated"] += q.id in seen_ever
            out["targeted"] += not student.known.get(q.concept_id, False)
            seen_ever.add(q.id)
            asked.add(q.id)
            now += 60.0
            model.observe_answer(q, student.answer(q, rng), now)
            student.learn(q, prereqs.get(q.concept_id, []), rng)
            student.practised_last.add(q.concept_id)
        out["brier"].append(brier(model, student, now))
        out["known"].append(known_fraction(student))
        student.forget(rng, forget_probability)
    out["final_known"] = out["known"][-1]
    out["converted"] = (sum(student.known[c] for c in out["unknown_at_start"]) / len(out["unknown_at_start"])
                        if out["unknown_at_start"] else None)
    return out


def _mean_se(values: list[float]) -> tuple[float, float]:
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return (values[0] if values else float("nan"), 0.0)
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values))


def evaluate(concepts: list[Concept], bank: list[Question], students_per_profile: int = 20, sessions: int = 6,
             per_session: int = 10, gap_days: float = 3.0, seed: int = 7, forget_probability: float = 0.25) -> dict:
    """Run every policy on the same simulated students and summarise."""
    runs: dict[str, list[dict]] = {p: [] for p in POLICIES}
    profile_of: list[str] = []
    for profile in PROFILES:
        for i in range(students_per_profile):
            start = make_student(concepts, profile, random.Random(f"{seed}-{profile}-{i}"))
            profile_of.append(profile)
            for policy in POLICIES:
                runs[policy].append(run_student(policy, start, concepts, bank, sessions, per_session, gap_days,
                                                seed=zlib.crc32(f"{seed}-{profile}-{i}".encode()),
                                                forget_probability=forget_probability))
    result = {"config": {"students": len(profile_of), "profiles": list(PROFILES), "sessions": sessions,
                         "questions_per_session": per_session, "gap_days": gap_days, "bank_size": len(bank), "seed": seed,
                         "forget_probability_per_gap": forget_probability},
              "policies": {}, "paired_gain_difference": {}}
    for policy, rs in runs.items():
        gains = [r["final_known"] - r["initial_known"] for r in rs]
        presented = sum(r["presented"] for r in rs)
        result["policies"][policy] = {
            "mastery_gain": _mean_se(gains),
            "unknown_concepts_learned": _mean_se([r["converted"] for r in rs]),
            "repetition_rate": sum(r["repeated"] for r in rs) / presented,
            "share_on_unknown_concepts": sum(r["targeted"] for r in rs) / presented,
            "brier_by_sitting": [statistics.fmean(r["brier"][k] for r in rs) for k in range(sessions + 1)],
            "known_by_sitting": [statistics.fmean(r["known"][k] for r in rs) for k in range(sessions + 1)],
            "gain_by_profile": {p: statistics.fmean(g for g, prof in zip(gains, profile_of) if prof == p) for p in PROFILES},
        }
    for other in ("random", "random_no_history"):
        diffs = [a["final_known"] - a["initial_known"] - (b["final_known"] - b["initial_known"])
                 for a, b in zip(runs["adaptive"], runs[other])]
        result["paired_gain_difference"][f"adaptive_minus_{other}"] = _mean_se(diffs)
    return result