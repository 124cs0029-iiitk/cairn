"""Benchmark the tutor: retrieval, answer quality, citations and refusals.

Usage (from the project folder):
    python scripts/run_rag_eval.py            # build the test set the first time, then evaluate
    python scripts/run_rag_eval.py --rebuild  # write a new test set

Writes data/eval/test_set.json and data/eval/rag_eval.json. Takes roughly 10 to 25 minutes because of model
rate limits; it can be stopped and restarted (judge answers are cached).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("CALLS_PER_SESSION_PER_HOUR", "5000")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import get_settings  # noqa: E402
from core.ingest import load_manifest  # noqa: E402
from core.rageval import (build_test_set, evaluate_item, load_test_set, save_json, summarize,  # noqa: E402
                          items_to_json)
from core.retrieval import FastEmbedder, Index, load_chunks  # noqa: E402
from core.retry import patient_chat  # noqa: E402


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def main() -> None:
    settings = get_settings()
    kb, out = settings.kb_dir, ROOT / "data" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(kb / "chunks.jsonl")
    modality_of = {c.id: c.modality.value for c in chunks}
    _, source_list = load_manifest(ROOT)
    sources = {s.id: s for s in source_list}
    index = Index.load(kb, FastEmbedder())

    path = out / "test_set.json"
    if path.exists() and "--rebuild" not in sys.argv:
        items = load_test_set(path)
        print(f"Using the saved test set: {len(items)} questions")
    else:
        print("Writing test questions from randomly chosen passages...")
        items = build_test_set(chunks, patient_chat, on_progress=lambda m: print("  ", m))
        save_json(path, items_to_json(items))

    rows = []
    for number, item in enumerate(items, start=1):
        row = evaluate_item(item, index, sources, patient_chat, modality_of)
        rows.append(row)
        outcome = "answered" if row["covered"] else "declined"
        print(f"[{number}/{len(items)}] {item.kind:<12} {outcome:<8} {item.question[:62]}")
        save_json(out / "rag_eval_rows.json", rows)

    summary = summarize(rows, settings.refusal_threshold)
    save_json(out / "rag_eval.json", {"summary": summary, "rows": rows,
                                      "models": {"answers": settings.generator_model, "judge": settings.verifier_model},
                                      "refusal_threshold": settings.refusal_threshold})
    r, g, c, f = summary["retrieval"], summary["generation"], summary["citations"], summary["refusal"]
    print(f"\n{summary['counts']['grounded']} grounded and {summary['counts']['off_material']} off-material questions "
          f"({summary['counts']['judge_errors']} judge errors)")
    print(f"\nRetrieval:  recall@1 {pct(r['recall_at_1'])}  recall@3 {pct(r['recall_at_3'])}  recall@6 {pct(r['recall_at_6'])}"
          f"  MRR {r['mrr']:.2f}")
    print(f"            context precision {pct(r['context_precision'])}  context recall {pct(r['context_recall'])}")
    print(f"Answers:    faithfulness {pct(g['faithfulness'])}  answer relevancy {pct(g['answer_relevancy'])}")
    print(f"Citations:  the source the question came from was cited in {pct(c['gold_source_cited'])} of answers; "
          f"{pct(c['cited_sources_relevant'])} of cited sources were relevant")
    print(f"Refusals:   {pct(f['correct_refusal_rate_off_material'])} of off-material questions declined "
          f"({f['refused_by_score_gate']} by the score gate, {f['refused_by_model_check']} by the model's own check); "
          f"{pct(f['false_refusal_rate_grounded'])} of course questions wrongly declined")
    for m, v in summary["retrieval_by_source_type"].items():
        print(f"   {m:<7} n={v['n']:<3} recall@1 {pct(v['recall_at_1'])} recall@6 {pct(v['recall_at_6'])} "
              f"precision {pct(v['context_precision'])}")
    print("\nSaved data/eval/rag_eval.json")


if __name__ == "__main__":
    main()