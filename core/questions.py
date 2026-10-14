"""Exam questions: generation from one passage, cross-model verification, duplicate removal, grading.

A question is written by the generator model from a single chunk, so it always has a source to cite.
A different model then checks the answer key, and questions that fail stay in the bank marked
unverified (with the reason) so rejection rates can be reported.
"""
from __future__ import annotations

import json
import random
import re
import zlib
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .jsonutil import parse_json_object
from .schema import Chunk, Concept, Modality

GENERATE_SYSTEM = """You write exam questions for a university course, using ONLY the passage provided.
Reply with a single JSON object and nothing else.
Multiple choice: {"skip": false, "stem": "...", "options": ["...", "...", "...", "..."], "answer": 0, "explanation": "..."}
  where answer is the index (0 to 3) of the single correct option.
Short answer: {"skip": false, "stem": "...", "answer": "model answer in one or two sentences", "explanation": "..."}
Rules: the question must be answerable from the passage alone and must test the named concept.
Exactly one option is correct; the others are plausible but clearly wrong.
Write a self-contained exam question. Never refer to "the passage", "the slide", "the text" or "described above".
  Good: "Which normal form removes transitive dependencies?"  Bad: "According to the passage, which normal form removes transitive dependencies?"
Difficulty: 1 = recall a definition, 3 = apply it to a small example, 5 = reason about a tricky case.
If the passage does not teach the named concept, reply {"skip": true}."""

VERIFY_MCQ_SYSTEM = """Answer the multiple-choice question using the passage.
Reply with a single JSON object: {"answer": <index of the option you choose>, "ambiguous": true or false}
Set ambiguous to true if more than one option could be argued correct, or if none is."""

VERIFY_SHORT_SYSTEM = """You check an exam answer key against a passage.
Reply with a single JSON object: {"supported": true or false, "reason": "one short sentence"}
supported is true only if the proposed answer is correct according to the passage."""

GRADE_SYSTEM = """You grade a student's short answer against a model answer and a passage from the course.
Reply with a single JSON object: {"correct": true or false, "feedback": "one or two sentences that say what is right or missing"}
Be lenient about wording, strict about meaning."""

_LEAKS = re.compile(r"\b(the passage|this passage|the slide|this slide|(?:in|from|by) the text|the lecture)\b", re.I)
_SOURCE = r"(?:the|this)\s+(?:passage|text|slide|lecture|material)"
_LEAD_IN = re.compile(rf"^(?:based on|according to|from|using|per)\s+{_SOURCE}\s*,?\s*", re.I)
_TRAILING = re.compile(rf",?\s+(?:as |that is |which is |that was )?(?:explicitly |directly )?"
                       rf"(?:described|mentioned|listed|cited|stated|presented|discussed|given|shown|defined|outlined)"
                       rf"\s+(?:in|by)\s+{_SOURCE}\b", re.I)
_ACCORDING = re.compile(rf",?\s+according to\s+{_SOURCE}\b", re.I)
_ABOUT_SOURCE = re.compile(r"\bnot\s+(?:explicitly\s+)?(?:mentioned|listed|stated|described|cited|included)\b", re.I)
_LETTERS = "ABCD"


class Question(BaseModel):
    id: str
    concept_id: str
    chunk_id: str
    qtype: str                                  # "mcq" or "short"
    difficulty: int = Field(ge=1, le=5)
    stem: str
    options: list[str] = Field(default_factory=list)
    answer_index: int | None = None             # mcq only
    answer: str                                 # mcq: the correct option's text; short: the model answer
    explanation: str = ""
    verified: bool = False
    reject_reason: str = ""

    @model_validator(mode="after")
    def _shape(self) -> "Question":
        if self.qtype not in ("mcq", "short"):
            raise ValueError("qtype must be mcq or short")
        if not self.stem.strip() or not self.answer.strip():
            raise ValueError("empty stem or answer")
        if self.qtype == "mcq":
            if len(self.options) != 4 or len({o.strip().lower() for o in self.options}) != 4:
                raise ValueError("an mcq needs four distinct options")
            if self.answer_index not in range(4) or self.options[self.answer_index] != self.answer:
                raise ValueError("answer_index does not point at the answer")
        return self


def _crc(text: str) -> int:
    return zlib.crc32(text.encode())


def clean_stem(stem: str) -> str:
    """Drop phrases that point at the source ("Based on the passage, ...", "as described in the slide").

    Questions that are really about what the source mentions ("which is NOT mentioned in the passage")
    cannot be fixed this way and are rejected.
    """
    if _ABOUT_SOURCE.search(stem) and _LEAKS.search(stem):
        raise ValueError("the question asks about the source itself")
    stem = _ACCORDING.sub("", _TRAILING.sub("", _LEAD_IN.sub("", stem.strip())))
    stem = " ".join(stem.split())
    return stem[:1].upper() + stem[1:]


def _answer_index(raw, options: list[str]) -> int:
    if isinstance(raw, bool):
        raise ValueError("bad answer")
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    if text.isdigit():
        return int(text)
    if len(text) == 1 and text.upper() in _LETTERS:
        return _LETTERS.index(text.upper())
    for i, option in enumerate(options):
        if option.strip().lower() == text.lower():
            return i
    raise ValueError("cannot tell which option is correct")


ANGLES = ["define the concept", "explain why it is used or what problem it solves",
          "apply it to a small concrete example", "tell it apart from a related concept or spot a common mistake"]


def generate_question(concept: Concept, chunk: Chunk, qtype: str, difficulty: int, llm: Callable,
                      qid: str, seed: int = 0, angle: str = "") -> Question | None:
    """One question from one chunk, or None when the generator says the passage does not teach the concept."""
    user = (f"Concept: {concept.name} - {concept.description}\nType: {qtype}\nDifficulty: {difficulty}\n"
            + (f"Question angle: {angle}\n" if angle else "") + f"\nPassage:\n{chunk.searchable_text[:1800]}")
    data = parse_json_object(llm("generator", [{"role": "system", "content": GENERATE_SYSTEM},
                                               {"role": "user", "content": user}],
                                 temperature=0.5, max_tokens=1200, use_cache=False))
    if data.get("skip"):
        return None
    stem = clean_stem(str(data["stem"]))
    explanation = str(data.get("explanation", "")).strip()
    if qtype == "short":
        return Question(id=qid, concept_id=concept.id, chunk_id=chunk.id, qtype="short",
                        difficulty=difficulty, stem=stem, answer=str(data["answer"]).strip(),
                        explanation=explanation)
    options = [str(o).strip() for o in data["options"]]
    correct = options[_answer_index(data["answer"], options)]
    random.Random(_crc(qid) + seed).shuffle(options)            # models favour the first positions
    return Question(id=qid, concept_id=concept.id, chunk_id=chunk.id, qtype="mcq", difficulty=difficulty,
                    stem=stem, options=options, answer_index=options.index(correct), answer=correct,
                    explanation=explanation)


def verify_question(q: Question, chunk: Chunk, llm: Callable) -> tuple[bool, str]:
    """A second model checks the key. Returns (accepted, reason)."""
    if _LEAKS.search(q.stem):
        return False, "stem refers to the source material"
    passage = chunk.searchable_text[:1800]
    if q.qtype == "mcq":
        options = "\n".join(f"{i}: {o}" for i, o in enumerate(q.options))
        data = parse_json_object(llm("verifier", [
            {"role": "system", "content": VERIFY_MCQ_SYSTEM},
            {"role": "user", "content": f"Passage:\n{passage}\n\nQuestion: {q.stem}\nOptions:\n{options}"}],
            temperature=0.0, max_tokens=400))
        if data.get("ambiguous"):
            return False, "verifier found the options ambiguous"
        if _answer_index(data.get("answer"), q.options) != q.answer_index:
            return False, "verifier chose a different option"
        return True, "verifier agrees"
    data = parse_json_object(llm("verifier", [
        {"role": "system", "content": VERIFY_SHORT_SYSTEM},
        {"role": "user", "content": f"Passage:\n{passage}\n\nQuestion: {q.stem}\nProposed answer: {q.answer}"}],
        temperature=0.0, max_tokens=400))
    return (True, "verifier agrees") if data.get("supported") else (False, str(data.get("reason", "not supported")))


def is_duplicate(vec: np.ndarray, existing: Iterable[np.ndarray], threshold: float = 0.88) -> bool:
    return any(float(vec @ other) >= threshold for other in existing)


class Grade(BaseModel):
    correct: bool
    feedback: str


def grade_answer(q: Question, response: str | int | None, chunk: Chunk, llm: Callable) -> Grade:
    if q.qtype == "mcq":
        ok = response == q.answer_index
        right = f"The correct answer is {_LETTERS[q.answer_index]}: {q.answer}."
        return Grade(correct=ok, feedback=(q.explanation if ok else f"{right} {q.explanation}").strip())
    text = str(response or "").strip()
    if not text:
        return Grade(correct=False, feedback=f"No answer given. Model answer: {q.answer}")
    data = parse_json_object(llm("verifier", [
        {"role": "system", "content": GRADE_SYSTEM},
        {"role": "user", "content": f"Passage:\n{chunk.searchable_text[:1500]}\n\nQuestion: {q.stem}\n"
                                    f"Model answer: {q.answer}\nStudent answer: {text}"}],
        temperature=0.0, max_tokens=400))
    correct = bool(data.get("correct"))
    feedback = str(data.get("feedback", "")).strip()
    return Grade(correct=correct, feedback=feedback if correct else f"{feedback} Model answer: {q.answer}".strip())


# ---- building and storing the bank --------------------------------------------------------------

PLAN = [("mcq", 1, Modality.SLIDES), ("mcq", 3, Modality.PDF), ("short", 3, Modality.SLIDES),
        ("mcq", 4, Modality.VIDEO)]
MIN_CHARS = {Modality.SLIDES: 250, Modality.PDF: 300, Modality.VIDEO: 350}


def pick_chunk(candidates: list[Chunk], modality: Modality, used: set[str], rng: random.Random) -> Chunk | None:
    pools = [[c for c in candidates if c.modality is m and len(c.text) >= MIN_CHARS[m] and c.id not in used]
             for m in (modality, Modality.SLIDES, Modality.PDF, Modality.VIDEO)]
    for pool in pools:
        if pool:
            return rng.choice(pool)
    return None


def build_bank(concepts: list[Concept], chunks: list[Chunk], tags: dict[str, list[str]], llm: Callable,
               embedder, existing: list[Question] | None = None, plan=PLAN, attempts: int = 2,
               on_progress: Callable[[str], None] = lambda message: None) -> list[Question]:
    """Fill the bank: for every concept and plan slot, generate, verify and de-duplicate."""
    bank = list(existing or [])
    by_id = {c.id: c for c in chunks}
    stem_vecs = [embedder.embed_documents([q.stem])[0] for q in bank if q.verified]
    for concept in concepts:
        candidates = [by_id[cid] for cid, ids in tags.items() if concept.id in ids and cid in by_id]
        used = {q.chunk_id for q in bank if q.concept_id == concept.id}
        rng = random.Random(_crc(concept.id))
        for slot, (qtype, difficulty, modality) in enumerate(plan):
            if any(q.id.startswith(f"{concept.id}:{slot}:") and q.verified for q in bank):
                continue                                           # done in an earlier run
            for attempt in range(attempts):
                chunk = pick_chunk(candidates, modality, used, rng)
                if chunk is None:
                    on_progress(f"{concept.id} slot {slot}: no usable passage")
                    break
                used.add(chunk.id)
                qid = f"{concept.id}:{slot}:{attempt}"
                try:
                    q = generate_question(concept, chunk, qtype, difficulty, llm, qid, angle=ANGLES[slot % len(ANGLES)])
                    if q is None:
                        on_progress(f"{qid}: skipped (passage does not teach the concept)")
                        continue
                    ok, reason = verify_question(q, chunk, llm)
                    if ok:
                        vec = embedder.embed_documents([q.stem])[0]
                        if is_duplicate(vec, stem_vecs):
                            ok, reason = False, "near-duplicate of an existing question"
                        else:
                            stem_vecs.append(vec)
                    q.verified, q.reject_reason = ok, "" if ok else reason
                    bank.append(q)
                    on_progress(f"{qid}: {'accepted' if ok else 'rejected - ' + reason}")
                    if ok:
                        break
                except (ValueError, KeyError, TypeError) as exc:
                    on_progress(f"{qid}: malformed ({exc})")
    return bank


def save_bank(path: Path, bank: list[Question]) -> None:
    path.write_text(json.dumps([q.model_dump() for q in bank], indent=1, ensure_ascii=False), encoding="utf-8")


def load_bank(path: Path) -> list[Question]:
    return [Question(**item) for item in json.loads(path.read_text(encoding="utf-8"))] if path.exists() else []