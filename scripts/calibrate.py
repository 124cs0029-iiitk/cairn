"""Score in-course, borderline and off-course questions against the index.

Usage (from the project folder):  python scripts/calibrate.py
Shows the best similarity score for each question so the refusal threshold can be chosen from data.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.citations import build_citation  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.ingest import load_manifest  # noqa: E402
from core.retrieval import FastEmbedder, Index  # noqa: E402

IN_COURSE = [
    "What is a foreign key?", "Explain third normal form", "What is a primary key?",
    "How do you draw an ER diagram?", "What is a functional dependency?",
    "Write a SQL query with a join", "What is the relational model?",
    "What is a weak entity set?", "Explain BCNF", "What does DBMS stand for?",
]
BORDERLINE = ["What is a B+ tree?", "Explain ACID transactions", "What is MongoDB?"]
OFF_COURSE = [
    "How do I train a neural network?", "Who won the cricket world cup?", "What is the capital of France?",
    "Explain quantum entanglement", "How do I bake bread?", "What is gradient descent?",
    "Tell me about the French revolution", "What is the speed of light?",
]


def main(embedder=None) -> dict:
    index = Index.load(get_settings().kb_dir, embedder or FastEmbedder())
    _, sources = load_manifest(ROOT)
    by_id = {s.id: s for s in sources}
    result = {}
    for name, questions in (("IN COURSE", IN_COURSE), ("BORDERLINE", BORDERLINE), ("OFF COURSE", OFF_COURSE)):
        print(f"\n{name}")
        scores = []
        for q in questions:
            hit = index.search(q, k=1)[0]
            label = build_citation(hit.chunk, by_id[hit.chunk.source_id]).label
            scores.append(hit.dense)
            print(f"  {hit.dense:.3f}  {q:<40} -> {label}")
        result[name] = scores
    lowest_in, highest_off = min(result["IN COURSE"]), max(result["OFF COURSE"])
    print(f"\nlowest in-course score : {lowest_in:.3f}")
    print(f"highest off-course score: {highest_off:.3f}")
    if highest_off < lowest_in:
        print(f"a score threshold of {(lowest_in + highest_off) / 2:.3f} separates them cleanly")
    else:
        print("the two groups overlap, so the answer check by the model must do the refusing")
    return result


if __name__ == "__main__":
    main()