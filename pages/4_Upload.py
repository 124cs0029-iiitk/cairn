import tempfile
from pathlib import Path

import streamlit as st

from core.config import get_settings
from core.llm import LLMError
from core.retrieval import FastEmbedder
from core.ui import page_header
from core.uploads import ingest_pdf, ingest_video

st.set_page_config(page_title="Upload - Cairn", page_icon="🪨", layout="wide")
page_header("Upload", "Add your own slides, textbook PDFs or a YouTube lecture. They go through the same pipeline as the "
                      "course, and the tutor can then answer from them with page or timestamp citations.")

settings = get_settings()
st.session_state.setdefault("uploads", [])
uploads = st.session_state["uploads"]


@st.cache_resource(show_spinner=False)
def embedder():
    return FastEmbedder()


def add(upload) -> None:
    uploads[:] = [u for u in uploads if u.source.id != upload.source.id] + [upload]


left, right = st.columns(2)
with left:
    st.subheader("A PDF")
    kind = st.radio("What is it?", ["slides", "textbook"], format_func=lambda k: "Lecture slides" if k == "slides" else "Textbook or notes",
                    horizontal=True)
    pdf = st.file_uploader("Choose a PDF", type=["pdf"], help=f"Up to {settings.max_upload_mb} MB. Text PDFs only; scanned pages are not read yet.")
    if pdf is not None and st.button("Add this PDF", type="primary"):
        if pdf.size > settings.max_upload_mb * 1_000_000:
            st.error(f"This file is larger than {settings.max_upload_mb} MB.")
        else:
            try:
                with st.spinner("Reading pages, removing headers and footers, embedding..."):
                    add(ingest_pdf(pdf.name, pdf.getvalue(), kind, embedder(), settings.static_dir))
                st.success("Added. Open the Tutor and ask about it.")
            except ValueError as exc:
                st.error(f"Could not add this PDF: {exc}")

with right:
    st.subheader("A YouTube lecture")
    url = st.text_input("Link", placeholder="https://www.youtube.com/watch?v=...",
                        help=f"Up to {settings.max_video_minutes} minutes, with English captions.")
    if st.button("Add this lecture", type="primary", disabled=not url.strip()):
        try:
            with st.spinner("Fetching captions and embedding..."):
                add(ingest_video(url, embedder(), Path(tempfile.gettempdir()) / "cairn_captions", settings.max_video_minutes))
            st.success("Added. Open the Tutor and ask about it.")
        except (ValueError, LLMError) as exc:
            st.error(f"Could not add this video: {exc}")
        except Exception as exc:                                  # network or YouTube problems
            st.error(f"Could not fetch this video right now: {str(exc)[:160]}")

st.subheader("In this session")
if not uploads:
    st.caption("Nothing uploaded yet. Uploads last for this browser session; the pre-built course is always available.")
for u in uploads:
    s = u.stats
    with st.container(border=True):
        cols = st.columns([5, 1])
        kind_label = {"slides": "Slides", "pdf": "Document", "video": "Video"}[u.source.modality.value]
        cols[0].markdown(f"**{u.source.title}**  \n{kind_label}, {len(u.chunks)} searchable passages")
        if u.source.modality.value == "video":
            note = "automatic captions" if s.get("automatic_captions") else "the speaker's own captions"
            cols[0].caption(f"{s['minutes']} minutes, read from {note}. Citations open the video at the right moment.")
        else:
            cols[0].caption(f"{s['pages']} pages read ({s['empty_pages']} without text), "
                            f"{len(s.get('boilerplate', []))} repeated header or footer lines removed, "
                            f"{s['figure_pages_found']} pages with figures noted.")
        if cols[1].button("Remove", key=f"rm_{u.source.id}"):
            uploads.remove(u)
            st.rerun()
if uploads:
    st.info("Uploaded material is searchable and cited by the tutor. Quizzes and the concept map cover the pre-built course.")