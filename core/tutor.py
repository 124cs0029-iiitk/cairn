"""Grounded tutor: answers only from retrieved course material, with citations.

Anything the material does not cover is declined, and any general-knowledge
remark is returned separately so the interface can label it as outside the course.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .citations import build_citation
from .config import get_settings
from .jsonutil import parse_json_object
from .llm import chat
from .retrieval import Hit, Index
from .schema import Citation, Source

SYSTEM = """You are a study tutor for one course. Answer ONLY from the numbered sources provided.
Reply with a single JSON object and nothing else:
{"covered": true or false, "answer": "...", "used": [1, 2], "outside": "..."}
- covered: true if the sources contain what is needed to answer the question, even partly.
- answer: concise markdown (at most about 150 words). Follow each statement taken from a source with its
  number in square brackets, for example [2]. If covered is false, use an empty string.
- used: the numbers of the sources you relied on. Never invent a number.
- outside: if the question is not covered, or only partly covered, a short general-knowledge reply
  (at most 80 words) for the missing part. Otherwise an empty string.
Do not put outside knowledge in "answer"."""

# [1], fullwidth brackets, and the number-plus-dagger-plus-line-range form some models write
_BRACKETED_NUMBER = re.compile(r"[\u3010\[\uff3b]\s*(\d+)(?:\s*\u2020[^\]\u3011\uff3d]*)?\s*[\u3011\]\uff3d]")

OUTSIDE_ONLY = ("The course material does not cover this. Give a short general-knowledge answer "
                "(at most 80 words) and do not mention sources.")


@dataclass
class TutorAnswer:
    covered: bool
    answer: str
    outside: str
    citations: list[Citation] = field(default_factory=list)
    citation_numbers: list[int] = field(default_factory=list)   # the [n] each citation carries in the answer
    top_score: float = 0.0
    hits: list[Hit] = field(default_factory=list)


def retrieval_query(question: str, history: list[tuple[str, str]]) -> str:
    """Short follow-ups such as 'why?' borrow the previous question for searching."""
    if history and len(question.split()) < 6:
        return f"{history[-1][0]} {question}"
    return question


def _context(hits: list[Hit], sources: dict[str, Source]) -> tuple[str, list[Citation]]:
    citations = [build_citation(h.chunk, sources[h.chunk.source_id]) for h in hits]
    blocks = [f"[{n}] ({c.label})\n{h.chunk.searchable_text[:1200]}"
              for n, (h, c) in enumerate(zip(hits, citations), start=1)]
    return "\n\n".join(blocks), citations


def answer_question(question: str, index: Index, sources: dict[str, Source],
                    history: list[tuple[str, str]] | None = None, session_id: str = "global",
                    k: int = 6, min_score: float | None = None,
                    llm: Callable = chat) -> TutorAnswer:
    history = history or []
    threshold = get_settings().refusal_threshold if min_score is None else min_score
    hits = index.search(retrieval_query(question, history), k=k)
    top = hits[0].dense if hits else 0.0

    if not hits or top < threshold:       # nothing close enough: do not even offer sources
        outside = llm("generator", [{"role": "system", "content": OUTSIDE_ONLY},
                                    {"role": "user", "content": question}],
                      temperature=0.2, session_id=session_id)
        return TutorAnswer(False, "", outside.strip(), [], [], top, hits)

    context, citations = _context(hits, sources)
    messages = [{"role": "system", "content": SYSTEM}]
    for q, a in history[-3:]:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"})
    data = parse_json_object(llm("generator", messages, temperature=0.1, session_id=session_id,
                                  max_tokens=1200))

    covered = bool(data.get("covered"))
    numbers = list(dict.fromkeys(n for n in data.get("used", []) if isinstance(n, int) and 1 <= n <= len(hits)))
    if covered and not numbers:
        numbers = list(range(1, min(3, len(hits)) + 1))
    answer = _BRACKETED_NUMBER.sub(r"[\1]", str(data.get("answer", "")).strip()) if covered else ""
    return TutorAnswer(covered, answer, str(data.get("outside", "")).strip(),
                       [citations[n - 1] for n in numbers] if covered else [],
                       numbers if covered else [], top, hits)