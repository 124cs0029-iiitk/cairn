"""Shared Streamlit components."""
from __future__ import annotations

import streamlit as st

from .citations import fmt_timestamp
from .quiz import learner_id_from
from .schema import Citation, Modality, Source


def render_citation(citation: Citation, source: Source | None = None, number: int | None = None) -> None:
    """A source the answer used. Documents open at the exact page; videos play from the cited moment."""
    with st.expander(f"[{number}] {citation.label}" if number else citation.label):
        st.caption(citation.excerpt)
        if citation.modality is Modality.VIDEO:
            start = int(citation.start_s or 0)
            if source and source.youtube_id:
                st.video(f"https://www.youtube.com/watch?v={source.youtube_id}", start_time=start)
            else:
                st.link_button(f"Open at {fmt_timestamp(start)}", citation.url)
        else:
            st.link_button(f"Open {citation.label}", citation.url)


def source_link(citation: Citation, key: str) -> None:
    """A plain link to the source, for places where an expander cannot be nested. `key` must be unique."""
    st.caption(f"Source: {citation.label}")
    st.link_button(f"Open {citation.label}", citation.url, key=key)


def learner_sidebar() -> str:
    """Ask for the student's name once; the same name is used on every page. Returns the learner id."""
    name = st.sidebar.text_input("Your name", value=st.session_state.get("learner_name", ""), placeholder="e.g. Asha")
    st.session_state["learner_name"] = name
    return learner_id_from(name)


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.write(subtitle)