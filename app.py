"""Cairn: home page and system status."""
import streamlit as st

from core import db
from core.config import get_settings, role_status

st.set_page_config(page_title="Cairn", page_icon="🪨", layout="wide")

settings = get_settings()
with db.session() as conn:
    n_learners = conn.execute("SELECT COUNT(*) FROM learners").fetchone()[0]

st.title("Cairn")
st.write(
    "A study companion that answers from your own lectures, slides and textbooks, "
    "quizzes you on them, and keeps track of what you know."
)

left, right = st.columns([3, 2])

with left:
    st.subheader("Where to start")
    st.write("**Upload** course material, ask the **Tutor** a question and open its sources, "
             "take a **Quiz**, then check the **Dashboard** for weak topics.")
    st.info("Ingestion, tutor, quiz and dashboard are being built step by step. "
            "This build is the deployment skeleton.")

with right:
    st.subheader("System status")
    for role, (model, ready) in role_status(settings).items():
        st.write(f"{'✅' if ready else '⚠️'} **{role}**: `{model}`" +
                 ("" if ready else " (provider key missing)"))
    kb_ready = any(p.name != ".gitkeep" for p in settings.kb_dir.iterdir())
    st.write(f"{'✅' if kb_ready else '⚠️'} **Knowledge base**: " +
             ("loaded" if kb_ready else "not built yet"))
    st.write(f"**Learners on record**: {n_learners}")
