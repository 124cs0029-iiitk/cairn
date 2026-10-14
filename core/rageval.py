"""Benchmark of the tutor: retrieval quality, answer quality, citations and refusals.

The test set has questions written from known passages (so the source location is known) and questions the
course does not cover. A judge model that is different from the answer-writing model scores the results with
the standard definitions:
  context precision  average precision of the retrieved passages, judged for relevance to the question
  context recall     share of the reference answer's statements found in the retrieved passages
  faithfulness       share of the answer's claims supported by the retrieved passages
  answer relevancy   how directly the answer addresses the question (judge score, 1 to 5, rescaled to 0..1)
"""
from __future__ import annotations

import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .jsonutil import parse_json_object
from .questions import clean_stem
from .retrieval import Index
from .schema import Chunk, Modality, Source
from .tutor import answer_question

QUESTION_SYSTEM = """You write test questions for a study assistant.
Given a passage from a course, write ONE question a student might ask that the passage answers, and a short
reference answer (one or two sentences) taken only from the passage.
Reply with a single JSON object: {"skip": false, "question": "...", "reference_answer": "..."}.
The question must be self-contained and must not mention "the passage", "the slide" or "the text".
If the passage has no clear fact worth asking about, reply {"skip": true}."""

CONTEXT_SYSTEM = """You judge retrieved course passages for a question.
You get a question, a reference answer and numbered passages. Reply with one JSON object:
{"relevant": [1 or 0 for each passage in order, 1 if the passage helps answer the question],
 "statements": [{"text": "...", "supported": true or false}]}
statements are the separate facts in the reference answer; supported says whether the passages together contain that fact."""

ANSWER_SYSTEM = """You judge an answer written for a student, using only the numbered passages it was based on.
Reply with one JSON object: {"claims": [{"claim": "...", "supported": true or false}], "relevance": 1 to 5}
claims lists each factual claim in the answer and whether the passages support it;
relevance is how directly the answer addresses the question (5 = fully, 1 = not at all)."""

OFF_MATERIAL = [
    "Who won the cricket world cup in 2011?", "What is the capital of France?", "How do I train a neural network?",
    "Explain quantum entanglement.", "How do I bake sourdough bread?", "What is gradient descent?",
    "How does TCP congestion control work?", "What is a Kubernetes pod?", "Tell me about the French revolution.",
    "How do vaccines work?", "What is the speed of light?", "Explain how a transformer language model works.",
]


@dataclass
class EvalItem:
    id: str
    kind: str                         # "grounded" or "off_material"
    question: str
    gold_chunk_id: str | None = None
    reference_answer: str | None = None


def build_test_set(chunks: list[Chunk], llm: Callable, per_modality: int = 10, seed: int = 5,
                   on_progress: Callable[[str], None] = lambda message: None) -> list[EvalItem]:
    """Grounded questions from randomly chosen passages (even numbers per source type), plus off-material ones."""
    rng = random.Random(seed)
    items: list[EvalItem] = []
    for modality in (Modality.SLIDES, Modality.PDF, Modality.VIDEO):
        pool = [c for c in chunks if c.modality is modality and len(c.text) >= 300]
        rng.shuffle(pool)
        made = 0
        for chunk in pool:
            if made == per_modality:
                break
            try:
                data = parse_json_object(llm("generator", [
                    {"role": "system", "content": QUESTION_SYSTEM},
                    {"role": "user", "content": f"Passage:\n{chunk.searchable_text[:1500]}"}],
                    temperature=0.4, max_tokens=600, use_cache=False))
                if data.get("skip"):
                    continue
                question = clean_stem(str(data["question"]))
                answer = str(data["reference_answer"]).strip()
                if not question or not answer:
                    continue
            except (ValueError, KeyError, TypeError):
                continue
            made += 1
            items.append(EvalItem(f"g{len(items) + 1}", "grounded", question, chunk.id, answer))
            on_progress(f"{modality.value} {made}/{per_modality}: {question[:70]}")
    items += [EvalItem(f"o{i}", "off_material", q) for i, q in enumerate(OFF_MATERIAL, start=1)]
    return items


def average_precision(flags: list[int]) -> float:
    """Mean of precision@k taken at each relevant rank (0 when nothing relevant was retrieved)."""
    hits, total = 0, 0.0
    for k, flag in enumerate(flags, start=1):
        if flag:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


def _passages(chunks: list[Chunk], limit: int = 800) -> str:
    return "\n\n".join(f"[{i}] {c.searchable_text[:limit]}" for i, c in enumerate(chunks, start=1))


def _ratio(items: list[dict], key: str) -> float | None:
    return sum(bool(i.get(key)) for i in items) / len(items) if items else None


def evaluate_item(item: EvalItem, index: Index, sources: dict[str, Source], llm: Callable,
                  gold_modality: dict[str, str]) -> dict:
    ans = answer_question(item.question, index, sources, [], "eval", llm=llm)
    row = {"id": item.id, "kind": item.kind, "question": item.question, "covered": ans.covered,
           "top_score": round(ans.top_score, 3), "answer": ans.answer, "outside": ans.outside,
           "retrieved": [h.chunk.id for h in ans.hits], "cited": [c.chunk_id for c in ans.citations],
           "gold": item.gold_chunk_id, "modality": gold_modality.get(item.gold_chunk_id or "")}
    if item.kind == "off_material":
        return row
    ranked = row["retrieved"]
    row["gold_rank"] = ranked.index(item.gold_chunk_id) + 1 if item.gold_chunk_id in ranked else None
    contexts = [h.chunk for h in ans.hits]
    try:
        verdict = parse_json_object(llm("verifier", [
            {"role": "system", "content": CONTEXT_SYSTEM},
            {"role": "user", "content": f"Question: {item.question}\nReference answer: {item.reference_answer}\n\n"
                                        f"Passages:\n{_passages(contexts)}"}], temperature=0.0, max_tokens=1200))
        flags = [int(bool(f)) for f in verdict["relevant"]][:len(contexts)]
        flags += [0] * (len(contexts) - len(flags))
        statements = verdict.get("statements", [])
        row["relevant_flags"] = flags
        row["context_precision"] = average_precision(flags)
        row["context_recall"] = (sum(bool(s.get("supported")) for s in statements) / len(statements)) if statements else None
        cited_flags = [f for cid, f in zip(ranked, flags) if cid in row["cited"]]
        row["citation_precision"] = (sum(cited_flags) / len(cited_flags)) if cited_flags else None
    except (ValueError, KeyError, TypeError) as exc:
        row["judge_error"] = f"context judge: {exc}"
    if ans.covered:
        row["citation_hit"] = item.gold_chunk_id in row["cited"]
        try:
            verdict = parse_json_object(llm("verifier", [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": f"Question: {item.question}\n\nPassages:\n{_passages(contexts)}\n\n"
                                            f"Answer:\n{ans.answer}"}], temperature=0.0, max_tokens=1200))
            claims = verdict.get("claims", [])
            row["faithfulness"] = (sum(bool(c.get("supported")) for c in claims) / len(claims)) if claims else None
            row["answer_relevancy"] = (min(5, max(1, float(verdict["relevance"]))) - 1) / 4
        except (ValueError, KeyError, TypeError) as exc:
            row["judge_error"] = f"answer judge: {exc}"
    return row


def _mean(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.fmean(values) if values else None


def summarize(rows: list[dict], threshold: float) -> dict:
    grounded = [r for r in rows if r["kind"] == "grounded"]
    off = [r for r in rows if r["kind"] == "off_material"]
    answered = [r for r in grounded if r["covered"]]

    def retrieval(subset: list[dict]) -> dict:
        ranks = [r.get("gold_rank") for r in subset]
        return {"n": len(subset),
                "recall_at_1": sum(1 for k in ranks if k == 1) / len(subset) if subset else None,
                "recall_at_3": sum(1 for k in ranks if k and k <= 3) / len(subset) if subset else None,
                "recall_at_6": sum(1 for k in ranks if k) / len(subset) if subset else None,
                "mrr": statistics.fmean(1 / k if k else 0.0 for k in ranks) if subset else None,
                "context_precision": _mean(subset, "context_precision"),
                "context_recall": _mean(subset, "context_recall")}

    return {
        "counts": {"grounded": len(grounded), "off_material": len(off), "answered": len(answered),
                   "judge_errors": sum(1 for r in rows if r.get("judge_error"))},
        "retrieval": retrieval(grounded),
        "retrieval_by_source_type": {m: retrieval([r for r in grounded if r["modality"] == m])
                                     for m in sorted({r["modality"] for r in grounded if r["modality"]})},
        "generation": {"faithfulness": _mean(answered, "faithfulness"), "answer_relevancy": _mean(answered, "answer_relevancy")},
        "citations": {"gold_source_cited": _ratio(answered, "citation_hit"), "cited_sources_relevant": _mean(grounded, "citation_precision")},
        "refusal": {"correct_refusal_rate_off_material": (sum(not r["covered"] for r in off) / len(off)) if off else None,
                    "false_refusal_rate_grounded": (sum(not r["covered"] for r in grounded) / len(grounded)) if grounded else None,
                    "refused_by_score_gate": sum(1 for r in off if not r["covered"] and r["top_score"] < threshold),
                    "refused_by_model_check": sum(1 for r in off if not r["covered"] and r["top_score"] >= threshold)},
    }


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def load_test_set(path: Path) -> list[EvalItem]:
    return [EvalItem(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def items_to_json(items: list[EvalItem]) -> list[dict]:
    return [asdict(i) for i in items]