"""Adding material at run time: a PDF (slides or textbook) or a YouTube lecture.

Uploads go through the same pipeline as the pre-built course: boilerplate and template graphics are removed,
text is split into chunks that keep their page or timestamp, and the chunks are embedded so the tutor can
search and cite them next to the course material.
"""
from __future__ import annotations

import hashlib
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .captions import pick_track
from .ingest import pdf_chunks, video_chunks
from .retrieval import Index
from .schema import Chunk, Modality, Source

YOUTUBE_ID = re.compile(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})")


@dataclass
class Upload:
    source: Source
    chunks: list[Chunk]
    matrix: np.ndarray
    stats: dict = field(default_factory=dict)


def youtube_id(url: str) -> str | None:
    match = YOUTUBE_ID.search(url.strip())
    return match.group(1) if match else None


def _embed(chunks: list[Chunk], embedder, batch: int = 64) -> np.ndarray:
    parts = [embedder.embed_documents([c.searchable_text[:2000] for c in chunks[i:i + batch]])
             for i in range(0, len(chunks), batch)]
    return np.vstack(parts).astype(np.float32)


def ingest_pdf(filename: str, data: bytes, kind: str, embedder, static_dir: Path | None = None) -> Upload:
    """kind is "slides" or "textbook". The file is kept under static/uploads so citations can open it."""
    stem = re.sub(r"[^a-z0-9]+", "_", Path(filename).stem.lower()).strip("_")[:30] or "file"
    uid = f"up_{stem}_{hashlib.md5(data).hexdigest()[:6]}"
    modality = Modality.SLIDES if kind == "slides" else Modality.PDF
    source = Source(id=uid, title=Path(filename).stem, modality=modality, static_path=f"uploads/{uid}.pdf")
    folder = (static_dir or Path(tempfile.mkdtemp())) / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uid}.pdf"
    path.write_bytes(data)
    try:
        chunks, figure_pages, stats = pdf_chunks(source, path)
    except Exception as exc:                                    # unreadable or encrypted files
        raise ValueError(f"could not read this PDF: {exc}") from exc
    if not chunks:
        raise ValueError("no readable text was found; scanned PDFs are not supported yet")
    stats = {**stats, "figure_pages_found": len(figure_pages)}
    return Upload(source, chunks, _embed(chunks, embedder), stats)


def ingest_video(url: str, embedder, work_dir: Path, max_minutes: int = 20) -> Upload:
    """A YouTube lecture through its captions (the speaker's own, or YouTube's automatic ones)."""
    vid = youtube_id(url)
    if vid is None:
        raise ValueError("that does not look like a YouTube link")
    import yt_dlp

    base = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(base) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
    minutes = (info.get("duration") or 0) / 60
    if minutes > max_minutes:
        raise ValueError(f"this video is {minutes:.0f} minutes long; the limit is {max_minutes}")
    key, automatic = pick_track(info.get("subtitles") or {}), False
    if key is None:
        autos = info.get("automatic_captions") or {}
        key, automatic = next((k for k in autos if k == "en" or k.startswith("en")), None), True
    if key is None:
        raise ValueError("this video has no English captions, so it cannot be read yet")
    work_dir.mkdir(parents=True, exist_ok=True)
    opts = {**base, "writesubtitles": not automatic, "writeautomaticsub": automatic, "subtitleslangs": [key],
            "subtitlesformat": "vtt", "outtmpl": str(work_dir / "%(id)s")}
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={vid}"])
    files = sorted(work_dir.glob(f"{vid}.*.vtt"))
    if not files:
        raise ValueError("the captions could not be downloaded")
    source = Source(id=f"up_yt_{vid}", title=info.get("title") or vid, modality=Modality.VIDEO, youtube_id=vid)
    chunks = video_chunks(source, files[0])
    if not chunks:
        raise ValueError("the captions were empty")
    stats = {"minutes": round(minutes, 1), "caption_track": key, "automatic_captions": automatic, "chunks": len(chunks)}
    return Upload(source, chunks, _embed(chunks, embedder), stats)


def combine(base: Index, uploads: list[Upload]) -> Index:
    """The course index plus everything uploaded in this session."""
    if not uploads:
        return base
    chunks = base.chunks + [c for u in uploads for c in u.chunks]
    matrix = np.vstack([base.matrix] + [u.matrix for u in uploads])
    return Index(chunks, matrix, base.embedder)


def merged_sources(sources: dict[str, Source], uploads: list[Upload]) -> dict[str, Source]:
    return {**sources, **{u.source.id: u.source for u in uploads}}