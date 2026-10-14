import time

import pandas as pd
import streamlit as st

from core import db
from core.citations import build_citation
from core.concepts import load_concept_graph
from core.config import get_settings
from core.dashboard import attempt_history, chat_counts, concept_dot, concept_rows, session_accuracy
from core.ingest import load_manifest
from core.learner import load_model
from core.retrieval import load_chunks
from core.schema import Modality
from core.ui import learner_sidebar, page_header, render_citation

st.set_page_config(page_title="Dashboard - Cairn", page_icon="🪨", layout="wide")
page_header("Dashboard", "What you know, what to work on next, and where to read about it.")

settings = get_settings()
kb = settings.kb_dir
if not (kb / "concepts.json").exists():
    st.warning("The concept map has not been built yet. Run `python scripts/build_concepts.py`.")
    st.stop()

learner_id = learner_sidebar()
if not learner_id:
    st.info("Enter your name in the sidebar to see your progress.")
    st.stop()


@st.cache_data(ttl=60, show_spinner=False)
def load_resources():
    concepts, tags = load_concept_graph(kb / "concepts.json")
    chunks = {c.id: c for c in load_chunks(kb / "chunks.jsonl")}
    _, sources = load_manifest(settings.root)
    return concepts, tags, chunks, {s.id: s for s in sources}


concepts, tags, chunks, sources = load_resources()
with db.session() as conn:
    db.get_or_create_learner(conn, learner_id, st.session_state["learner_name"])
    model = load_model(conn, learner_id, concepts)
    history = attempt_history(conn, learner_id)
    asked_in_chat = chat_counts(conn, learner_id)

now = time.time()
rows = concept_rows(model, now)
practised = [r for r in rows if r["answers"]]

if not history and not asked_in_chat:
    st.info("No activity yet. Take a quiz or ask the tutor a question, and your progress will appear here.")
    st.subheader("Course map")
    st.graphviz_chart(concept_dot(concepts, model, now), width="stretch")
    st.stop()

# ---- headline numbers ---------------------------------------------------------------------------
correct = sum(h["correct"] for h in history)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Concepts practised", f"{len(practised)} of {len(rows)}")
c2.metric("Answers", len(history))
c3.metric("Accuracy", f"{100 * correct / len(history):.0f}%" if history else "-")
c4.metric("Average mastery", f"{100 * sum(r['mastery'] for r in rows) / len(rows):.0f}%")

left, right = st.columns(2)
with left:
    st.subheader("Mastery by topic")
    topic_mastery = model.topic_mastery(now)
    st.bar_chart(pd.DataFrame({"Mastery (%)": {t: 100 * m for t, m in topic_mastery.items()}}), horizontal=True)
with right:
    st.subheader("Progress over time")
    sittings = session_accuracy(history)
    if len(sittings) >= 2:
        st.line_chart(pd.DataFrame({"Share correct (%)": [100 * a for _, a, _ in sittings]},
                                   index=[f"Sitting {n}" for n, _, _ in sittings]))
    else:
        st.caption("Take at least two quizzes to see a trend.")

# ---- weak spots and cited revision material -----------------------------------------------------
st.subheader("Work on these next")
weakest = sorted(practised or rows, key=lambda r: r["mastery"])[:3]
due = [r for r in rows if r["due"]]
for r in weakest:
    asked = f", asked the tutor about it {asked_in_chat[r['id']]} time(s)" if r["id"] in asked_in_chat else ""
    st.markdown(f"**{r['concept']}** ({r['topic']}): mastery {100 * r['mastery']:.0f}%{asked}")
    candidates = [chunks[cid] for cid, ids in tags.items() if r["id"] in ids and cid in chunks and len(chunks[cid].text) >= 250]
    candidates.sort(key=lambda c: (c.modality is not Modality.SLIDES, c.modality is not Modality.PDF))
    for chunk in candidates[:2]:
        render_citation(build_citation(chunk, sources[chunk.source_id]), sources[chunk.source_id])
if due:
    st.markdown("**Due for review (fading from memory):** " + ", ".join(r["concept"] for r in due))

# ---- every concept ------------------------------------------------------------------------------
st.subheader("All concepts")
table = pd.DataFrame([{"Topic": r["topic"], "Concept": r["concept"], "Mastery": round(100 * r["mastery"]),
                       "Answers": r["answers"], "Status": r["status"].capitalize(),
                       "Due for review": "Yes" if r["due"] else ""} for r in rows])
st.dataframe(table, hide_index=True, width="stretch",
             column_config={"Mastery": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%")})

st.subheader("Course map")
st.caption("Arrows point from a prerequisite to the concept that builds on it. Red: weak, amber: developing, "
           "green: strong, grey: not practised yet.")
st.graphviz_chart(concept_dot(concepts, model, now), width="stretch")
