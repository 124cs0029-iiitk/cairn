"""Build citations whose links open the exact page, slide or video timestamp."""
from __future__ import annotations

from .schema import Chunk, Citation, Modality, Source

STATIC_URL_PREFIX = "/app/static"   # Streamlit static serving


def fmt_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def pdf_url(static_path: str, page: int) -> str:
    return f"{STATIC_URL_PREFIX}/{static_path.lstrip('/')}#page={page}"


def youtube_watch_url(video_id: str, seconds: float) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s"


def youtube_embed_url(video_id: str, seconds: float) -> str:
    return f"https://www.youtube.com/embed/{video_id}?start={int(seconds)}"


def build_citation(chunk: Chunk, source: Source, excerpt_chars: int = 280) -> Citation:
    if chunk.source_id != source.id:
        raise ValueError(f"Chunk {chunk.id} belongs to {chunk.source_id}, not {source.id}")

    text = " ".join(chunk.searchable_text.split())
    excerpt = text if len(text) <= excerpt_chars else text[: excerpt_chars - 1].rstrip() + "…"

    if chunk.modality is Modality.VIDEO:
        t = chunk.locator.start_s or 0.0
        if source.youtube_id:
            url = youtube_watch_url(source.youtube_id, t)
        else:
            url = f"{STATIC_URL_PREFIX}/{source.static_path}#t={int(t)}"
        label = f"{source.title} at {fmt_timestamp(t)}"
        return Citation(chunk_id=chunk.id, source_id=source.id, label=label, url=url,
                        modality=chunk.modality, excerpt=excerpt, start_s=t)

    page = chunk.locator.page or 1
    noun = "slide" if chunk.modality is Modality.SLIDES else "page"
    return Citation(chunk_id=chunk.id, source_id=source.id,
                    label=f"{source.title}, {noun} {page}",
                    url=pdf_url(source.static_path or "", page),
                    modality=chunk.modality, excerpt=excerpt, page=page)
