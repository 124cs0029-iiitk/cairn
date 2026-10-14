"""Build the cited chunk file for the course.

Usage (from the project folder):  python scripts/build_chunks.py
Writes data/kb/chunks.jsonl and data/kb/figure_pages.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.ingest import load_manifest, pdf_chunks, video_chunks  # noqa: E402
from core.schema import Modality  # noqa: E402


def main() -> None:
    _, sources = load_manifest(ROOT)
    all_chunks, figure_pages, problems = [], [], []

    print(f"{'source':<44} {'chunks':>6} {'pages':>6} {'empty':>6} {'figures':>8}")
    for src in sources:
        try:
            if src.modality is Modality.VIDEO:
                vtt = ROOT / "data" / "raw" / "captions" / f"{src.youtube_id}.vtt"
                chunks = video_chunks(src, vtt)
                print(f"{src.title[:43]:<44} {len(chunks):>6} {'-':>6} {'-':>6} {'-':>8}")
            else:
                chunks, figs, st = pdf_chunks(src, ROOT / "static" / src.static_path)
                figure_pages.extend(figs)
                print(f"{src.title[:43]:<44} {st['chunks']:>6} {st['pages']:>6} "
                      f"{st['empty_pages']:>6} {st['figure_pages']:>8}")
                for n, example in st["boilerplate"]:
                    print(f"    removed (on {n} pages): {example[:70]!r}")
            all_chunks.extend(chunks)
        except Exception as exc:
            problems.append(f"{src.title}: {exc}")
            print(f"{src.title[:43]:<44} FAILED: {str(exc)[:50]}")

    out = ROOT / "data" / "kb"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(c.model_dump_json() + "\n")
    (out / "figure_pages.json").write_text(json.dumps(figure_pages), encoding="utf-8")

    by_mod = {m.value: sum(c.modality is m for c in all_chunks) for m in Modality}
    print(f"\nTotal chunks: {len(all_chunks)}  {by_mod}")
    print(f"Pages with figures (for the vision step): {len(figure_pages)}")
    if problems:
        print("Problems:", *problems, sep="\n  ")

    def show(label, chunk):
        print(f"\n--- sample: {label} ({chunk.id}) ---\n{chunk.text[:350]}")

    for label, pred in (("slide", lambda c: c.source_id == "slides_ch7" and c.locator.page == 12),
                        ("textbook", lambda c: c.source_id == "textbook" and c.locator.page == 60),
                        ("video", lambda c: c.modality is Modality.VIDEO and c.locator.start_s > 600)):
        chunk = next((c for c in all_chunks if pred(c)), None)
        if chunk:
            show(label, chunk)


if __name__ == "__main__":
    main()