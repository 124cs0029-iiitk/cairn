"""Build the bank of verified exam questions.

Usage (from the project folder):
    python scripts/build_questions.py --concepts 3     # a quick trial on the first 3 concepts
    python scripts/build_questions.py                  # all concepts (resumes where it stopped)

Each question is written by the generator model from one passage and checked by a different model.
The bank is saved after every concept to data/kb/questions.json, so it can be interrupted safely.
"""
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CALLS_PER_SESSION_PER_HOUR", "5000")      # this is a batch job, not a student

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.concepts import load_concept_graph  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.llm import LLMError, chat  # noqa: E402
from core.questions import build_bank, load_bank, save_bank  # noqa: E402
from core.retrieval import FastEmbedder, load_chunks  # noqa: E402

TRANSIENT = ("rate", "429", "too many", "tokens per", "getaddrinfo", "connect", "timed out", "timeout",
             "502", "503", "504", "overloaded")


def patient_chat(role, messages, **kwargs):
    """chat() that waits and retries on rate limits and short network or server problems."""
    for attempt in range(7):
        try:
            return chat(role, messages, **kwargs)
        except LLMError as exc:
            text = str(exc).lower()
            if attempt < 6 and any(word in text for word in TRANSIENT):
                wait = 15 * (attempt + 1)
                print(f"    (temporary problem: {str(exc)[:70]}... waiting {wait}s)")
                time.sleep(wait)
            else:
                raise


def main() -> None:
    limit = int(sys.argv[sys.argv.index("--concepts") + 1]) if "--concepts" in sys.argv else None
    kb = get_settings().kb_dir
    concepts, tags = load_concept_graph(kb / "concepts.json")
    chunks = load_chunks(kb / "chunks.jsonl")
    path = kb / "questions.json"
    bank = load_bank(path)
    embedder = FastEmbedder()
    todo = concepts[:limit] if limit else concepts

    for number, concept in enumerate(todo, start=1):
        print(f"[{number}/{len(todo)}] {concept.name}")
        bank = build_bank([concept], chunks, tags, patient_chat, embedder, existing=bank,
                          on_progress=lambda message: print("   ", message))
        save_bank(path, bank)

    verified = [q for q in bank if q.verified]
    print(f"\n{len(bank)} questions generated, {len(verified)} verified, {len(bank) - len(verified)} rejected")
    print("verified by type:", dict(Counter(q.qtype for q in verified)),
          "| by difficulty:", dict(sorted(Counter(q.difficulty for q in verified).items())))
    print("rejection reasons:", dict(Counter(q.reject_reason for q in bank if not q.verified)))
    for q in verified[:2]:
        print(f"\nExample ({q.qtype}, difficulty {q.difficulty}, {q.concept_id}):\n  {q.stem}")
        for i, option in enumerate(q.options):
            print(f"   {'*' if i == q.answer_index else ' '} {option}")
        if q.qtype == "short":
            print("   answer:", q.answer)


if __name__ == "__main__":
    main()