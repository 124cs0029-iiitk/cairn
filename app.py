"""Cairn: home page."""
import json
from collections import Counter

import streamlit as st

from core.config import get_settings, role_status

st.set_page_config(page_title="Cairn", page_icon="🪨", layout="wide")
settings = get_settings()


@st.cache_data(ttl=300, show_spinner=False)
def course_stats() -> dict:
    kb = settings.kb_dir
    stats = {"passages": Counter(), "concepts": 0, "questions": 0, "lectures": 0}
    if (kb / "chunks.jsonl").exists():
        with open(kb / "chunks.jsonl", encoding="utf-8") as fh:
            stats["passages"] = Counter(json.loads(line)["modality"] for line in fh if line.strip())
    if (kb / "concepts.json").exists():
        stats["concepts"] = len(json.loads((kb / "concepts.json").read_text(encoding="utf-8"))["concepts"])
    if (kb / "questions.json").exists():
        stats["questions"] = sum(1 for q in json.loads((kb / "questions.json").read_text(encoding="utf-8")) if q["verified"])
    manifest = settings.root / "data" / "course" / "dbms.json"
    if manifest.exists():
        stats["lectures"] = sum(s["modality"] == "video" for s in json.loads(manifest.read_text(encoding="utf-8"))["sources"])
    return stats


st.title("Cairn")
st.subheader("A study companion that answers from your own lectures, slides and textbooks.")
st.write("Every answer cites the exact slide, page or video moment it came from, and says so when the course "
         "material does not cover a question. Quizzes are checked by a second model, and your progress is tracked "
         "per concept, so practice goes where you need it.")

s = course_stats()
p = s["passages"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Searchable passages", sum(p.values()), help=f"{p['slides']} from slides, {p['pdf']} from the textbook, {p['video']} from lecture videos")
c2.metric("Lectures", s["lectures"], help="NPTEL Database Management Systems, IIT Kharagpur")
c3.metric("Concepts", s["concepts"], help="Grouped into five topics, with prerequisites")
c4.metric("Verified questions", s["questions"])

st.markdown("#### Where to start")
a, b, c = st.columns(3)
with a:
    st.markdown("**Ask the tutor**  \nGet an answer with sources you can open, or a clear note that the course does not cover it.")
    st.page_link("pages/1_Tutor.py", label="Open the tutor")
with b:
    st.markdown("**Take a quiz**  \nA diagnostic for new students, adaptive practice, or a mock exam, with a review of every answer.")
    st.page_link("pages/2_Quiz.py", label="Open the quiz")
with c:
    st.markdown("**See your progress**  \nMastery by concept, weak spots with reading to match, and a map of how concepts depend on each other.")
    st.page_link("pages/3_Dashboard.py", label="Open the dashboard")
d, e, f = st.columns(3)
with d:
    st.markdown("**Add your own material**  \nUpload slides or a textbook PDF, or paste a YouTube lecture link.")
    st.page_link("pages/4_Upload.py", label="Open upload")
with e:
    st.markdown("**How well does it work?**  \nBenchmarks for the tutor, the question bank and the personalization.")
    st.page_link("pages/5_Evaluation.py", label="Open the evaluation")

with st.expander("Sources and credits"):
    st.markdown(
        "- **Slides:** *Database System Concepts*, 7th edition, by Silberschatz, Korth and Sudarshan. "
        "Shown here for a hackathon demonstration; see [db-book.com](https://www.db-book.com) for the book and its terms of use.\n"
        "- **Textbook:** *Database Design*, 2nd edition, by Adrienne Watt and Nelson Eng, licensed CC BY 4.0.\n"
        "- **Lectures:** NPTEL *Database Management System*, IIT Kharagpur (Prof. Partha Pratim Das and Prof. Samiran Chattopadhyay), "
        "linked from YouTube; the app reads their captions and does not host the videos.")

with st.expander("System status"):
    for role, (model, ready) in role_status(settings).items():
        st.write(f"{'✅' if ready else '⚠️'} **{role}**: `{model}`" + ("" if ready else " (provider key missing)"))