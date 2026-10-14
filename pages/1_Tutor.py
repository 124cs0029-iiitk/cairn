import time
import uuid

import streamlit as st

from core import db
from core.concepts import load_concept_graph
from core.config import get_settings
from core.ingest import load_manifest
from core.learner import load_model, log_chat, save_model
from core.llm import LLMError
from core.retrieval import FastEmbedder, Index
from core.tutor import answer_question
from core.ui import learner_sidebar, page_header, render_citation
from core.uploads import combine, merged_sources

st.set_page_config(page_title="Tutor - Cairn", page_icon="🪨", layout="wide")
page_header("Tutor", "Ask a question. Answers come from your course material, with sources you can open.")

settings = get_settings()
if not (settings.kb_dir / "embeddings.npy").exists():
    st.warning("The search index has not been built yet. Run `python scripts/build_index.py`.")
    st.stop()


@st.cache_resource(show_spinner="Loading the course knowledge base...")
def load_kb():
    _, sources = load_manifest(settings.root)
    concepts, tags = load_concept_graph(settings.kb_dir / "concepts.json") \
        if (settings.kb_dir / "concepts.json").exists() else ([], {})
    return Index.load(settings.kb_dir, FastEmbedder()), {s.id: s for s in sources}, concepts, tags


index, sources, concepts, tags = load_kb()
uploads = st.session_state.get("uploads", [])
if uploads:                                              # course material plus what was uploaded this session
    upload_key = tuple(u.source.id for u in uploads)
    if st.session_state.get("combined_key") != upload_key:
        st.session_state["combined_index"], st.session_state["combined_key"] = combine(index, uploads), upload_key
    index, sources = st.session_state["combined_index"], merged_sources(sources, uploads)
    st.caption(f"Searching the course and {len(uploads)} uploaded file(s).")
learner_id = learner_sidebar()
st.session_state.setdefault("session_id", str(uuid.uuid4()))
st.session_state.setdefault("tutor_messages", [])
st.session_state.setdefault("pending_question", None)    # set while an answer is being prepared

if st.sidebar.button("Clear conversation"):
    st.session_state.tutor_messages = []
    st.session_state.pending_question = None
    st.rerun()
if not learner_id:
    st.sidebar.caption("Add your name so the tutor can keep track of what you ask about.")


def show_answer(ans) -> None:
    if ans.covered:
        st.markdown(ans.answer)
        if ans.outside:
            st.info(f"**Outside the course material (not cited):** {ans.outside}")
    else:
        st.warning("The course material does not cover this question.")
        if ans.outside:
            st.markdown(f"**General knowledge, not from your course:** {ans.outside}")
    for number, citation in zip(ans.citation_numbers, ans.citations):
        render_citation(citation, sources[citation.source_id], number)


def answered_pairs(messages) -> list[tuple[str, str]]:
    """(question, answer) pairs from earlier turns, used to resolve short follow-ups."""
    return [(a["content"], b["answer"].answer) for a, b in zip(messages, messages[1:])
            if a["role"] == "user" and b["role"] == "assistant" and "answer" in b]


def note_in_profile(ans) -> list[str]:
    """Record which concepts the answer was about, as weak evidence for the learner model."""
    if not learner_id or not ans.covered:
        return []
    names = {c.id: c.name for c in concepts}
    ids = list(dict.fromkeys(cid for citation in ans.citations for cid in tags.get(citation.chunk_id, [])))[:3]
    if not ids:
        return []
    now = time.time()
    with db.session() as conn:
        db.get_or_create_learner(conn, learner_id, st.session_state["learner_name"])
        model = load_model(conn, learner_id, concepts)
        model.observe_chat(ids, now)
        save_model(conn, learner_id, model)
        log_chat(conn, learner_id, ids, now)
    return [names[i] for i in ids if i in names]


for message in st.session_state.tutor_messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        elif "error" in message:
            st.error(message["error"])
        else:
            show_answer(message["answer"])
            if message.get("noted"):
                st.caption("Added to your learning profile: " + ", ".join(message["noted"]))

pending = st.session_state.pending_question
submitted = st.chat_input("Ask about the course material" if pending is None else "Preparing the answer...",
                          disabled=pending is not None)

if submitted and pending is None:
    st.session_state.tutor_messages.append({"role": "user", "content": submitted})
    st.session_state.pending_question = submitted
    st.rerun()                      # redraw with the input disabled, then answer

if pending is not None:
    history = answered_pairs(st.session_state.tutor_messages[:-1])
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the course material..."):
                ans = answer_question(pending, index, sources, history, st.session_state.session_id)
            st.session_state.tutor_messages.append({"role": "assistant", "answer": ans, "noted": note_in_profile(ans)})
        except (LLMError, ValueError) as exc:
            st.session_state.tutor_messages.append(
                {"role": "assistant", "error": f"Could not answer right now: {exc}"})
    st.session_state.pending_question = None
    st.rerun()