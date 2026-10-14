"""Turn course sources (PDFs, slide decks, video captions) into cited chunks.

Every chunk carries its origin: page or slide number for documents, start time
for video. Nothing here is specific to one course or publisher: headers,
footers and template graphics are found by looking for what repeats across the
pages of the same document.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .captions import parse_vtt, window_cues
from .schema import Chunk, Locator, Modality, Source

MAX_CHARS = {Modality.SLIDES: 1800, Modality.PDF: 1200}
MIN_PAGES_FOR_REPEAT_DETECTION = 4
REPEAT_FRACTION = 0.25          # boilerplate repeats on at least this share of pages
BAND = 0.12                     # top and bottom share of the page that can hold headers and footers
MIN_FIGURE_SHAPES = 3           # leftover large shapes that suggest a diagram
MIN_SHAPE_SIZE = 30             # points; smaller shapes are bullets and decoration
MIN_IMAGE_AREA = 0.02           # an image smaller than this share of the page is ignored
BACKGROUND_AREA = 0.85
WIDE_SHAPE = 0.70          # a shape this large is a page background
BANNER_MAX_HEIGHT = 0.20        # image strips this short (share of page height) in the header or footer area are decoration

_PAGE_TOKEN = r"(?:\d+|[ivxlc]{1,6})"
_EDGE_NUMBER = re.compile(rf"^{_PAGE_TOKEN}\s*[•·|\-–—]\s*|\s*[•·|\-–—]\s*{_PAGE_TOKEN}$", re.I)
_LEGAL = re.compile(r"©|\bcopyright\b|all rights reserved|conditions on re-?use", re.I)
_BULLET_ONLY = re.compile(r"[•●▪◦○·\-–—*■□▸►]")
_PRIVATE_USE = re.compile("[\ue000-\uf8ff]")

@dataclass
class Block:
    text: str
    band: str                    # "top", "bottom" or "body"


@dataclass
class PageData:
    blocks: list[Block]
    figure_items: set = field(default_factory=set)

def tidy_symbols(text: str) -> str:
    """Symbol-font bullets come out as private-use characters; show them as a normal bullet."""
    return _PRIVATE_USE.sub("•", text)

def normalise(text: str) -> str:
    """Form of a line used to compare it across pages (page numbers ignored)."""
    t = _EDGE_NUMBER.sub("", text.strip().lower())
    return re.sub(r"\d+", "#", t)


def detect_boilerplate(pages: list[PageData]) -> dict[str, tuple[int, str]]:
    """Header and footer blocks that repeat across pages: {normalised: (pages, example)}."""
    if len(pages) < MIN_PAGES_FOR_REPEAT_DETECTION:
        return {}
    needed = max(3, math.ceil(REPEAT_FRACTION * len(pages)))
    counts: Counter = Counter()
    example: dict[str, str] = {}
    for page in pages:
        seen = {normalise(b.text) for b in page.blocks if b.band != "body"}
        for key in seen:
            counts[key] += 1
        for b in page.blocks:
            example.setdefault(normalise(b.text), b.text)
    return {k: (n, example[k]) for k, n in counts.items() if n >= needed and k}


def clean_blocks(blocks: list[Block], boilerplate: dict) -> list[str]:
    """Drop boilerplate and legal lines, and attach lone bullet marks to the next block."""
    out: list[str] = []
    pending = ""
    for b in blocks:
        if b.band != "body" and normalise(b.text) in boilerplate:
            continue
        if len(b.text) <= 200 and _LEGAL.search(b.text):
            continue
        if _BULLET_ONLY.fullmatch(b.text):
            pending = b.text + " "
            continue
        out.append(pending + b.text)
        pending = ""
    return out


def split_blocks(blocks: list[str], max_chars: int) -> list[str]:
    """Group consecutive blocks into chunks no longer than max_chars where possible."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for block in blocks:
        if current and size + len(block) + 1 > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(block)
        size += len(block) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def _figure_items(page) -> set:
    width, height = page.rect.width, page.rect.height
    area = width * height
    items: set = set()
    for info in page.get_image_info():
        x0, y0, x1, y1 = info["bbox"]
        if (x1 - x0) * (y1 - y0) < MIN_IMAGE_AREA * area:
            continue
        in_edge_band = y0 <= BAND * height or y1 >= (1 - BAND) * height
        if in_edge_band and (y1 - y0) <= BANNER_MAX_HEIGHT * height:
            continue                    # title banner or footer strip: decoration, not a figure
        # Identified by position and pixel size, not by internal id: converted decks often
        # store the same graphic once per page under different ids.
        items.add(("img", info.get("width", 0), info.get("height", 0),
                   round(x0), round(y0), round(x1), round(y1)))
    for d in page.get_drawings():
        r = d["rect"]
        if (r.width * r.height >= BACKGROUND_AREA * area or max(r.width, r.height) < MIN_SHAPE_SIZE or r.width >= WIDE_SHAPE * width):
            continue
        items.add(("shape", round(r.x0), round(r.y0), round(r.x1), round(r.y1)))
    return items


def figure_flags(pages: list[PageData]) -> list[bool]:
    """Per page: does it hold images or diagrams beyond the repeated template graphics?"""
    template: set = set()
    if len(pages) >= MIN_PAGES_FOR_REPEAT_DETECTION:
        needed = max(3, math.ceil(REPEAT_FRACTION * len(pages)))
        counts = Counter(item for p in pages for item in p.figure_items)
        template = {item for item, n in counts.items() if n >= needed}
    flags = []
    for p in pages:
        kept = p.figure_items - template
        images = sum(1 for i in kept if i[0] == "img")
        shapes = sum(1 for i in kept if i[0] == "shape")
        flags.append(images >= 1 or shapes >= MIN_FIGURE_SHAPES)
    return flags


def read_pages(doc) -> list[PageData]:
    pages = []
    for page in doc:
        height = page.rect.height
        blocks = []
        for b in page.get_text("blocks", sort=True):
            if b[6] != 0:
                continue
            text = tidy_symbols(" ".join(b[4].split()))
            if not text:
                continue
            band = "top" if b[3] <= BAND * height else "bottom" if b[1] >= (1 - BAND) * height else "body"
            blocks.append(Block(text, band))
        pages.append(PageData(blocks, _figure_items(page)))
    return pages


def pdf_chunks(source: Source, pdf_path: Path) -> tuple[list[Chunk], list[dict], dict]:
    """Return (chunks, figure_pages, stats) for a PDF or slide source."""
    import pymupdf

    limit = MAX_CHARS[source.modality]
    with pymupdf.open(pdf_path) as doc:
        pages = read_pages(doc)

    boilerplate = detect_boilerplate(pages)
    flags = figure_flags(pages)
    chunks: list[Chunk] = []
    figure_pages: list[dict] = []
    stats = {"pages": len(pages), "chunks": 0, "empty_pages": 0, "figure_pages": 0,
             "boilerplate": sorted(((n, ex) for n, ex in boilerplate.values()), reverse=True)[:3]}

    for index, (page, has_figure) in enumerate(zip(pages, flags), start=1):
        lines = clean_blocks(page.blocks, boilerplate)
        text = "\n".join(lines)
        if has_figure:
            stats["figure_pages"] += 1
            figure_pages.append({"source_id": source.id, "page": index, "text_chars": len(text)})
        if len(text) < 15:
            stats["empty_pages"] += 1
            continue
        for k, part in enumerate(split_blocks(lines, limit)):
            chunks.append(Chunk(id=f"{source.id}:p{index}:{k}", source_id=source.id,
                                modality=source.modality, locator=Locator(page=index),
                                text=part, has_figure=has_figure))
    stats["chunks"] = len(chunks)
    return chunks, figure_pages, stats


def video_chunks(source: Source, vtt_path: Path) -> list[Chunk]:
    segments = window_cues(parse_vtt(vtt_path.read_text(encoding="utf-8")))
    return [Chunk(id=f"{source.id}:t{int(seg.start)}", source_id=source.id, modality=Modality.VIDEO,
                  locator=Locator(start_s=seg.start, end_s=seg.end), text=seg.text)
            for seg in segments]


def load_manifest(root: Path, course: str = "dbms") -> tuple[dict, list[Source]]:
    manifest = json.loads((root / "data" / "course" / f"{course}.json").read_text(encoding="utf-8"))
    sources = [Source(**{k: v for k, v in s.items() if k in Source.model_fields})
               for s in manifest["sources"]]
    return manifest, sources