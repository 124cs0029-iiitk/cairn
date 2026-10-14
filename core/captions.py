"""Parse WebVTT captions and group them into passages that can be cited.

Each passage keeps the start time of its first cue, so a citation can open the
video at the right moment.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass

_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})")
_CUE = re.compile(r"^(?P<a>\S+)\s+-->\s+(?P<b>\S+)")
_TAG = re.compile(r"<[^>]+>")
_SENTENCE_END = (".", "?", "!")


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


def parse_timestamp(value: str) -> float:
    m = _TS.fullmatch(value.strip())
    if not m:
        raise ValueError(f"Bad timestamp: {value!r}")
    h, mnt, sec, ms = m.groups()
    return int(h or 0) * 3600 + int(mnt) * 60 + int(sec) + int(ms) / 1000


def clean_text(raw: str) -> str:
    text = _TAG.sub("", html.unescape(raw)).replace("\xa0", " ")
    return " ".join(text.split())


def parse_vtt(content: str) -> list[Cue]:
    lines = content.replace("\r\n", "\n").split("\n")
    cues: list[Cue] = []
    i = 0
    while i < len(lines):
        m = _CUE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        try:
            start, end = parse_timestamp(m["a"]), parse_timestamp(m["b"])
        except ValueError:
            i += 1
            continue
        i += 1
        buf: list[str] = []
        while i < len(lines) and lines[i].strip():
            buf.append(lines[i])
            i += 1
        text = clean_text(" ".join(buf))
        if text and not (cues and cues[-1].text == text):
            cues.append(Cue(start, end, text))
    return cues


def window_cues(cues: list[Cue], target_s: float = 45.0, max_s: float = 75.0,
                min_s: float = 15.0) -> list[Segment]:
    """Group cues into passages of about target_s seconds, ending on a sentence."""
    segments: list[Segment] = []
    current: list[Cue] = []

    def flush() -> None:
        if current:
            segments.append(Segment(current[0].start, current[-1].end,
                                    " ".join(c.text for c in current)))
            current.clear()

    for cue in cues:
        current.append(cue)
        span = current[-1].end - current[0].start
        if span >= max_s or (span >= target_s and cue.text.rstrip().endswith(_SENTENCE_END)):
            flush()
    flush()

    # Fold a very short final passage into the one before it
    if len(segments) >= 2 and segments[-1].end - segments[-1].start < min_s:
        last, prev = segments.pop(), segments.pop()
        segments.append(Segment(prev.start, last.end, prev.text + " " + last.text))
    return segments


def pick_track(subtitles: dict) -> str | None:
    """Choose a manually made English caption track from yt-dlp's `subtitles` dict.

    Prefers tracks named "Verified" (NPTEL's reviewed captions), then plain 'en',
    then any other English track. Returns None when there is none, which means
    the lecture needs speech-to-text instead.
    """
    english = [k for k in subtitles if k == "en" or k.startswith("en-") or k.startswith("en_")]
    for key in english:
        if any("verified" in str(fmt.get("name", "")).lower() for fmt in subtitles[key]):
            return key
    if "en" in english:
        return "en"
    return english[0] if english else None
