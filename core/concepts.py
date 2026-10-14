"""Concept map of the course: topics, concepts, prerequisites, and a concept tag for every chunk.

Concepts are proposed by a model from each topic's slide titles, checked for a valid prerequisite
graph, and chunks are tagged by embedding similarity (slides and video only compete within their
own topic, so a SQL slide cannot be tagged with a normalization concept).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import numpy as np

from .jsonutil import parse_json_object
from .schema import Chunk, Concept, Modality

SYSTEM = """You design a concept map for one topic of a university course.
Reply with a single JSON object and nothing else:
{"concepts": [{"id": "snake_case_id", "name": "Short name", "description": "One sentence definition", "prerequisites": ["other_concept_id"]}]}
Rules: give 6 to 8 concepts that together cover the topic. Ids are lowercase snake_case and unique.
prerequisites are ids of concepts listed earlier in your answer or under "Concepts from earlier topics";
use [] when there are none. A concept never depends on itself."""

_NON_WORD = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    return _NON_WORD.sub("_", text.lower()).strip("_")[:40]


def slide_titles(chunks: list[Chunk], source_ids: set[str], limit: int = 45) -> list[str]:
    """First line of each slide in the given sources, in order, without repeats, spread evenly."""
    titles, seen = [], set()
    for c in chunks:
        if c.source_id in source_ids and c.modality is Modality.SLIDES:
            line = c.text.strip().split("\n")[0].strip(" \u2022\u00b7-")[:80]
            if line and line.lower() not in seen:
                seen.add(line.lower())
                titles.append(line)
    if len(titles) > limit:
        titles = [titles[int(i * len(titles) / limit)] for i in range(limit)]
    return titles


def generate_concepts(topic: str, titles: list[str], known: list[Concept],
                      llm: Callable, attempts: int = 2) -> list[Concept]:
    earlier = "\n".join(f"- {c.id}: {c.name}" for c in known) or "- (none)"
    prompt = (f"Topic: {topic}\n\nSlide titles from this topic:\n" + "\n".join(f"- {t}" for t in titles)
              + f"\n\nConcepts from earlier topics:\n{earlier}")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            raw = parse_json_object(llm("generator", messages, temperature=0.2, max_tokens=2500))["concepts"]
            return _sanitise(topic, raw, known)
        except (ValueError, KeyError, TypeError) as exc:
            last_error = exc
    raise ValueError(f"could not get a valid concept list for {topic!r}: {last_error}")


def _sanitise(topic: str, raw: list[dict], known: list[Concept]) -> list[Concept]:
    allowed = {c.id for c in known}
    out: list[Concept] = []
    for item in raw:
        cid = slugify(str(item.get("id") or item.get("name", "")))
        if not cid or cid in allowed or cid in {c.id for c in out}:
            continue
        prereqs = [slugify(str(p)) for p in item.get("prerequisites", [])]
        prereqs = [p for p in dict.fromkeys(prereqs) if (p in allowed or p in {c.id for c in out}) and p != cid]
        out.append(Concept(id=cid, name=str(item.get("name", cid)).strip(), topic=topic,
                           description=str(item.get("description", "")).strip(), prerequisites=prereqs))
    if len(out) < 3:
        raise ValueError(f"only {len(out)} usable concepts")
    return out


def validate_graph(concepts: list[Concept]) -> None:
    """Every prerequisite must exist and the graph must have no cycles."""
    by_id = {c.id: c for c in concepts}
    for c in concepts:
        for p in c.prerequisites:
            if p not in by_id:
                raise ValueError(f"{c.id} depends on unknown concept {p}")
    state: dict[str, int] = {}

    def visit(cid: str) -> None:
        if state.get(cid) == 1:
            raise ValueError(f"prerequisite cycle through {cid}")
        if state.get(cid) == 2:
            return
        state[cid] = 1
        for p in by_id[cid].prerequisites:
            visit(p)
        state[cid] = 2

    for c in concepts:
        visit(c.id)


def tag_chunks(chunks: list[Chunk], matrix: np.ndarray, concepts: list[Concept], embedder,
               chunk_topic: dict[str, str | None], min_score: float = 0.45, margin: float = 0.04,
               max_tags: int = 2) -> dict[str, list[str]]:
    """chunk id -> concept ids. Chunks whose source belongs to a topic only compete within it."""
    concept_vecs = embedder.embed_documents([f"{c.name}. {c.description}" for c in concepts])
    sims = matrix @ concept_vecs.T
    topics = np.array([c.topic for c in concepts])
    tags: dict[str, list[str]] = {}
    for i, chunk in enumerate(chunks):
        row = sims[i].copy()
        topic = chunk_topic.get(chunk.source_id)
        if topic is not None and (topics == topic).any():
            row[topics != topic] = -np.inf
        order = np.argsort(-row)
        best = row[order[0]]
        if best < min_score:
            continue
        picked = [concepts[order[0]].id]
        for j in order[1:max_tags]:
            if row[j] >= best - margin:
                picked.append(concepts[j].id)
        tags[chunk.id] = picked
    return tags


def save_concept_graph(path: Path, concepts: list[Concept], tags: dict[str, list[str]]) -> None:
    path.write_text(json.dumps({"concepts": [c.model_dump() for c in concepts], "tags": tags},
                               indent=1, ensure_ascii=False), encoding="utf-8")


def load_concept_graph(path: Path) -> tuple[list[Concept], dict[str, list[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Concept(**c) for c in data["concepts"]], data["tags"]