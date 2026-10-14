"""What the dashboard shows: concept status, the prerequisite map coloured by mastery, and history."""
from __future__ import annotations

import sqlite3

from .learner import MasteryModel
from .schema import Concept

COLOURS = {"unseen": "#DDE3E1", "weak": "#EBA9A4", "developing": "#F1D38A", "strong": "#9DCFAA"}
WEAK_BELOW, STRONG_FROM, DUE_BELOW = 0.4, 0.7, 0.6


def status(model: MasteryModel, cid: str, now: float) -> str:
    state = model.states.get(cid)
    if state is None or state.n_obs == 0:
        return "unseen"
    m = model.mastery(cid, now)
    return "weak" if m < WEAK_BELOW else "developing" if m < STRONG_FROM else "strong"


def concept_rows(model: MasteryModel, now: float) -> list[dict]:
    rows = []
    for c in model.concepts.values():
        state = model.states.get(c.id)
        answers = state.n_obs if state else 0
        rows.append({"id": c.id, "concept": c.name, "topic": c.topic, "mastery": model.mastery(c.id, now),
                     "answers": answers, "status": status(model, c.id, now),
                     "due": answers > 0 and model.retention(c.id, now) < DUE_BELOW})
    return rows


def _escape(text: str) -> str:
    return text.replace('"', "'").replace("\\", "")


def concept_dot(concepts: list[Concept], model: MasteryModel, now: float) -> str:
    """Graphviz source: one box per concept, grouped by topic, arrows from prerequisite to concept."""
    lines = ['digraph course {', 'rankdir=LR; compound=true; fontname="Helvetica";',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
             'edge [color="#8A9A96", arrowsize=0.6];']
    topics = list(dict.fromkeys(c.topic for c in concepts))
    for i, topic in enumerate(topics):
        lines.append(f'subgraph cluster_{i} {{ label="{_escape(topic)}"; style=rounded; color="#B8C4C0"; fontsize=11;')
        for c in (c for c in concepts if c.topic == topic):
            colour = COLOURS[status(model, c.id, now)]
            percent = round(100 * model.mastery(c.id, now))
            lines.append(f'"{c.id}" [label="{_escape(c.name)}\\n{percent}%", fillcolor="{colour}"];')
        lines.append("}")
    known = {c.id for c in concepts}
    for c in concepts:
        lines += [f'"{p}" -> "{c.id}";' for p in c.prerequisites if p in known]
    lines.append("}")
    return "\n".join(lines)


def attempt_history(conn: sqlite3.Connection, learner_id: str) -> list[dict]:
    rows = conn.execute("SELECT concept_id, correct, session_id, ts FROM attempts WHERE learner_id = ? ORDER BY ts, id",
                        (learner_id,)).fetchall()
    return [dict(r) for r in rows]


def session_accuracy(history: list[dict]) -> list[tuple[int, float, int]]:
    """(sitting number, share correct, answers) for each sitting, oldest first."""
    order: list[str] = []
    tallies: dict[str, list[int]] = {}
    for h in history:
        sid = h["session_id"] or "unknown"
        if sid not in tallies:
            order.append(sid)
            tallies[sid] = [0, 0]
        tallies[sid][0] += h["correct"]
        tallies[sid][1] += 1
    return [(i + 1, tallies[sid][0] / tallies[sid][1], tallies[sid][1]) for i, sid in enumerate(order)]


def chat_counts(conn: sqlite3.Connection, learner_id: str) -> dict[str, int]:
    return {r["concept_id"]: r["n"] for r in conn.execute(
        "SELECT concept_id, COUNT(*) AS n FROM chat_events WHERE learner_id = ? GROUP BY concept_id", (learner_id,))}