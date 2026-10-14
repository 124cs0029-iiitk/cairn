"""Run the simulated-student evaluation on the real question bank.

Usage (from the project folder):  python scripts/run_simulation.py
Runs three forgetting scenarios (the speed at which simulated students forget is the biggest assumption)
and writes data/eval/simulation.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.concepts import load_concept_graph  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.questions import load_bank  # noqa: E402
from core.simulate import evaluate  # noqa: E402

SCENARIOS = {"slow forgetting": 0.08, "realistic forgetting": 0.25, "fast forgetting": 0.40}


def main() -> None:
    kb = get_settings().kb_dir
    concepts, _ = load_concept_graph(kb / "concepts.json")
    bank = [q for q in load_bank(kb / "questions.json") if q.verified]
    print(f"{len(concepts)} concepts, {len(bank)} verified questions")
    results = {}
    for name, forgetting in SCENARIOS.items():
        results[name] = evaluate(concepts, bank, students_per_profile=50, forget_probability=forgetting)
        c = results[name]["config"]
        print(f"\n=== {name}: {forgetting:.0%} of unpractised concepts forgotten per {c['gap_days']:.0f}-day gap; "
              f"{c['students']} students x {c['sessions']} sittings x {c['questions_per_session']} questions")
        print(f"{'policy':<20} {'mastery gain':>15} {'learned':>15} {'repeats':>8} {'on unknown':>11}   Brier first -> last")
        for policy, p in results[name]["policies"].items():
            gain, gse = p["mastery_gain"]
            learned, lse = p["unknown_concepts_learned"]
            print(f"{policy:<20} {gain:>8.3f} ±{gse:.3f} {learned:>8.3f} ±{lse:.3f} {p['repetition_rate']:>8.1%} "
                  f"{p['share_on_unknown_concepts']:>11.1%}   {p['brier_by_sitting'][0]:.3f} -> {p['brier_by_sitting'][-1]:.3f}")
        for diff, (mean, se) in results[name]["paired_gain_difference"].items():
            print(f"  paired {diff}: {mean:+.3f} ± {se:.3f}")
    out = ROOT / "data" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "simulation.json").write_text(json.dumps({"scenarios": results, "default": "realistic forgetting"}, indent=1),
                                         encoding="utf-8")
    print("\nSaved data/eval/simulation.json")


if __name__ == "__main__":
    main()