import json
from collections import Counter

import pandas as pd
import streamlit as st

from core.config import get_settings
from core.questions import load_bank
from core.ui import page_header

st.set_page_config(page_title="Evaluation - Cairn", page_icon="🪨", layout="wide")
page_header("Evaluation", "How well the tutor, the question bank and the personalization work, measured on this course.")

settings = get_settings()
kb, ev = settings.kb_dir, settings.root / "data" / "eval"


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


# ---- question bank ------------------------------------------------------------------------------
st.header("Question bank")
bank = load_bank(kb / "questions.json")
if bank:
    verified = [q for q in bank if q.verified]
    reasons = Counter("The second model disagreed with the answer key" if "verifier" in q.reject_reason or q.reject_reason.startswith(("The ", "the "))
                      else "Too similar to an accepted question" if "near-duplicate" in q.reject_reason
                      else "Referred to the source material" if "source" in q.reject_reason
                      else "Other" for q in bank if not q.verified)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Generated", len(bank))
    c2.metric("Verified and kept", f"{len(verified)} ({100 * len(verified) / len(bank):.0f}%)")
    c3.metric("Multiple choice / short answer", f"{sum(q.qtype == 'mcq' for q in verified)} / {sum(q.qtype == 'short' for q in verified)}")
    c4.metric("Concepts covered", len({q.concept_id for q in verified}))
    st.caption("Every question is written by one model from a single passage and checked by a different model, "
               "which must reach the same answer. Rejections: " + (", ".join(f"{n} {r.lower()}" for r, n in reasons.most_common()) or "none") + ".")
else:
    st.info("The question bank has not been built yet.")

# ---- tutor benchmark ----------------------------------------------------------------------------
st.header("Tutor: retrieval, answers, citations and refusals")
path = ev / "rag_eval.json"
if path.exists():
    data = json.loads(path.read_text(encoding="utf-8"))
    s, rows = data["summary"], data["rows"]
    r, g, c, f = s["retrieval"], s["generation"], s["citations"], s["refusal"]
    st.caption(f"{s['counts']['grounded']} questions written from known passages and {s['counts']['off_material']} questions the course does not cover. "
               f"Answers by {data['models']['answers']}, judged by a different model ({data['models']['judge']}).")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Faithfulness", pct(g["faithfulness"]), help="Share of the answer's claims supported by the retrieved passages")
    a2.metric("Answer relevancy", pct(g["answer_relevancy"]), help="How directly the answer addresses the question (judge score)")
    a3.metric("Context precision", pct(r["context_precision"]), help="Average precision of the retrieved passages")
    a4.metric("Context recall", pct(r["context_recall"]), help="Share of the reference answer's statements found in the retrieved passages")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Source found in top 1 / 6", f"{pct(r['recall_at_1'])} / {pct(r['recall_at_6'])}")
    b2.metric("Correct refusals", pct(f["correct_refusal_rate_off_material"]), help="Off-material questions the tutor declined")
    b3.metric("Wrong refusals", pct(f["false_refusal_rate_grounded"]), help="Course questions the tutor wrongly declined")
    b4.metric("Cited sources relevant", pct(c["cited_sources_relevant"]))
    st.caption(f"Of the off-material questions declined, {f['refused_by_score_gate']} were stopped by the similarity gate "
               f"and {f['refused_by_model_check']} by the model's own check that the passages really cover the question. "
               f"The source a question was written from was cited in {pct(c['gold_source_cited'])} of answers "
               "(a lower bound: another passage can answer equally well).")
    by_type = pd.DataFrame([{"Source type": m, "Questions": v["n"], "Found in top 1": pct(v["recall_at_1"]),
                             "Found in top 6": pct(v["recall_at_6"]), "Context precision": pct(v["context_precision"]),
                             "Context recall": pct(v["context_recall"])} for m, v in s["retrieval_by_source_type"].items()])
    st.dataframe(by_type, hide_index=True, width="stretch")
    problems = [row for row in rows if (row["kind"] == "grounded" and not row["covered"])
                or (row["kind"] == "off_material" and row["covered"]) or (row.get("faithfulness") is not None and row["faithfulness"] < 0.7)]
    with st.expander(f"Questions that went wrong ({len(problems)})"):
        for row in problems:
            kind = ("wrongly declined" if row["kind"] == "grounded" and not row["covered"]
                    else "answered although not in the course" if row["kind"] == "off_material" else "weakly supported answer")
            st.markdown(f"**{row['question']}**  \n{kind}; best similarity {row['top_score']}")
else:
    st.info("The tutor benchmark has not been run yet. Run `python scripts/run_rag_eval.py`.")

# ---- personalization ----------------------------------------------------------------------------
st.header("Personalization: simulated students")
path = ev / "simulation.json"
if path.exists():
    sim = json.loads(path.read_text(encoding="utf-8"))
    scenario = st.selectbox("How fast simulated students forget", list(sim["scenarios"]),
                            index=list(sim["scenarios"]).index(sim["default"]))
    res = sim["scenarios"][scenario]
    cfg = res["config"]
    st.caption(f"{cfg['students']} simulated students in four profiles, {cfg['sessions']} sittings of {cfg['questions_per_session']} questions, "
               f"{cfg['gap_days']:.0f} days apart, drawing on {cfg['bank_size']} verified questions. The same students take every policy.")
    names = {"adaptive": "Adaptive (ours)", "random": "Random, no repeats", "random_no_history": "Random, no memory of past quizzes"}
    table = pd.DataFrame([{"Policy": names[k], "Mastery gain": f"{p['mastery_gain'][0]:+.3f} ± {p['mastery_gain'][1]:.3f}",
                           "Repeated questions": f"{100 * p['repetition_rate']:.1f}%",
                           "Estimation error (Brier), first to last": f"{p['brier_by_sitting'][0]:.3f} to {p['brier_by_sitting'][-1]:.3f}"}
                          for k, p in res["policies"].items()])
    st.dataframe(table, hide_index=True, width="stretch")
    d = res["paired_gain_difference"]
    st.markdown("**Adaptive minus random, paired by student (mean ± standard error):**  "
                f"against random without repeats {d['adaptive_minus_random'][0]:+.3f} ± {d['adaptive_minus_random'][1]:.3f};  "
                f"against a quiz with no memory {d['adaptive_minus_random_no_history'][0]:+.3f} ± {d['adaptive_minus_random_no_history'][1]:.3f}.")
    left, right = st.columns(2)
    with left:
        st.markdown("**The learner model gets more accurate with data** (lower is better)")
        st.line_chart(pd.DataFrame({names[k]: p["brier_by_sitting"] for k, p in res["policies"].items()},
                                   index=[f"After {i}" for i in range(cfg["sessions"] + 1)]))
    with right:
        st.markdown("**Share of concepts the student actually knows**")
        st.line_chart(pd.DataFrame({names[k]: p["known_by_sitting"] for k, p in res["policies"].items()},
                                   index=[f"After {i}" for i in range(cfg["sessions"] + 1)]))
    st.caption("How to read this: the simulator is a sanity check, not a study of real learners. Estimation error and repeated "
               "questions do not depend on how the simulator makes students learn; the size of the gains does. In this run the "
               "adaptive policy is better than random by a small margin and clearly better than a quiz that forgets what it has asked, "
               "and its mastery estimates are about as accurate as those from random questions, not more. Absolute gains can be "
               "negative because the simulated students forget faster than a few quizzes teach.")
else:
    st.info("The simulation has not been run yet. Run `python scripts/run_simulation.py`.")