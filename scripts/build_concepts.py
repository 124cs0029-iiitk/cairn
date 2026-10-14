"""Build the concept map: topics, concepts, prerequisites and a concept tag for every chunk.

Usage (from the project folder):  python scripts/build_concepts.py
Needs data/kb/chunks.jsonl and data/kb/embeddings.npy. Writes data/kb/concepts.json.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.concepts import generate_concepts, save_concept_graph, slide_titles, tag_chunks, validate_graph  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.ingest import load_manifest  # noqa: E402
from core.llm import chat  # noqa: E402
from core.retrieval import FastEmbedder, load_chunks  # noqa: E402


def main() -> None:
    manifest, sources = load_manifest(ROOT)
    kb = get_settings().kb_dir
    chunks = load_chunks(kb / "chunks.jsonl")
    matrix = np.load(kb / "embeddings.npy")
    titles_of = {g["id"]: g["title"] for g in manifest["groups"]}
    group_of = {s["id"]: s.get("group") for s in manifest["sources"]}

    concepts = []
    for gid, topic in titles_of.items():
        ids = {sid for sid, g in group_of.items() if g == gid}
        new = generate_concepts(topic, slide_titles(chunks, ids), concepts, chat)
        concepts.extend(new)
        print(f"{topic}: {len(new)} concepts")
    validate_graph(concepts)

    chunk_topic = {sid: titles_of.get(g) for sid, g in group_of.items()}
    tags = tag_chunks(chunks, matrix, concepts, FastEmbedder(), chunk_topic)
    save_concept_graph(kb / "concepts.json", concepts, tags)

    mod_of = {c.id: c.modality.value for c in chunks}
    print(f"\n{len(concepts)} concepts, {len(tags)} of {len(chunks)} chunks tagged\n")
    for topic in titles_of.values():
        print(topic)
        for c in (c for c in concepts if c.topic == topic):
            counts = Counter(mod_of[cid] for cid, ids in tags.items() if c.id in ids)
            needs = ", ".join(c.prerequisites) or "-"
            print(f"  {c.id:<30} slides {counts['slides']:>3}  video {counts['video']:>3}  "
                  f"book {counts['pdf']:>3}   needs: {needs}")
    untagged = Counter(c.modality.value for c in chunks if c.id not in tags)
    print("\nuntagged chunks by type:", dict(untagged))


if __name__ == "__main__":
    main()