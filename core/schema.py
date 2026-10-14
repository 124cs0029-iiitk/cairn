"""Data models shared by ingestion, retrieval, assessment and the UI.

Every content unit keeps its origin (page, slide number or video timestamp) so a
citation can always open the exact place it came from.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Modality(str, Enum):
    PDF = "pdf"          # textbook or notes
    SLIDES = "slides"    # slide deck (converted to PDF, so slide number == page)
    VIDEO = "video"


class Source(BaseModel):
    id: str
    title: str
    modality: Modality
    # File under static/ (relative), for PDF and slides, e.g. "courses/ml/lec3.pdf"
    static_path: Optional[str] = None
    # Hosted video: YouTube id. Uploaded video uses static_path instead.
    youtube_id: Optional[str] = None

    @model_validator(mode="after")
    def _has_location(self) -> "Source":
        if self.modality in (Modality.PDF, Modality.SLIDES) and not self.static_path:
            raise ValueError("PDF and slide sources need static_path")
        if self.modality is Modality.VIDEO and not (self.youtube_id or self.static_path):
            raise ValueError("Video sources need youtube_id or static_path")
        return self


class Locator(BaseModel):
    """Where inside a source a chunk lives."""
    page: Optional[int] = Field(default=None, ge=1)       # page or slide number, 1-based
    start_s: Optional[float] = Field(default=None, ge=0)  # video start, seconds
    end_s: Optional[float] = Field(default=None, ge=0)


class Chunk(BaseModel):
    id: str
    source_id: str
    modality: Modality
    locator: Locator
    text: str = ""
    figure_caption: str = ""            # VLM description of figures or on-screen content
    concept_ids: list[str] = Field(default_factory=list)
    has_figure: bool = False            # page holds images or drawings (for the vision pass)

    @model_validator(mode="after")
    def _locator_matches_modality(self) -> "Chunk":
        if self.modality in (Modality.PDF, Modality.SLIDES) and self.locator.page is None:
            raise ValueError("PDF and slide chunks need locator.page")
        if self.modality is Modality.VIDEO and self.locator.start_s is None:
            raise ValueError("Video chunks need locator.start_s")
        if not (self.text.strip() or self.figure_caption.strip()):
            raise ValueError("Chunk has neither text nor figure caption")
        return self

    @property
    def searchable_text(self) -> str:
        return "\n".join(p for p in (self.text.strip(), self.figure_caption.strip()) if p)


class Concept(BaseModel):
    id: str
    name: str
    topic: str                                   # parent topic or subtopic name
    description: str = ""
    prerequisites: list[str] = Field(default_factory=list)   # concept ids


class Citation(BaseModel):
    chunk_id: str
    source_id: str
    label: str            # e.g. "Lecture 3, slide 12" or "Lecture 3 at 12:34"
    url: str              # opens the exact page or timestamp
    modality: Modality
    excerpt: str
    page: Optional[int] = None
    start_s: Optional[float] = None
