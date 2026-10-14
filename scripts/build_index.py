"""Embed every chunk and save the search index.

Usage (from the project folder):  python scripts/build_index.py
Needs data/kb/chunks.jsonl from build_chunks.py. Also prints a few test searches so you can
see how relevant and off-topic questions score.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import get_settings  # noqa: E402
from core.ingest import load_manifest  # noqa: E402
from core.citations import build_citation  # noqa: E402
from core.retrieval import FastEmbedder, Index, load_chunks  # noqa: E402

TESTS = [
    "What is a foreign key?",
    "Explain third normal form",
    "How is an entity relationship diagram drawn?",
    "Write a SELECT with a join",
    "How do I train a neural network?",
    "Who won the cricket world cup?",
]


def main() -> None:
    kb = get_settings().kb_dir
    chunks = load_chunks(kb / "chunks.jsonl")
    print(f"Embedding {len(chunks)} chunks (the first run downloads the model)...")
    start = time.time()
    index = Index.build(chunks, FastEmbedder())
    index.save(kb)
    print(f"Done in {time.time() - start:.0f}s, matrix {index.matrix.shape}")

    _, sources = load_manifest(ROOT)
    by_id = {s.id: s for s in sources}
    print(f"\n{'question':<46} {'top score':>9}  best source")
    for q in TESTS:
        hits = index.search(q, k=3)
        label = build_citation(hits[0].chunk, by_id[hits[0].chunk.source_id]).label
        print(f"{q:<46} {hits[0].dense:>9.3f}  {label}")


if __name__ == "__main__":
    main()