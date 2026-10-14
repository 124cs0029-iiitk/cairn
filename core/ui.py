"""Shared Streamlit components."""
from __future__ import annotations

import streamlit as st

from .citations import fmt_timestamp
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


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.write(subtitle)