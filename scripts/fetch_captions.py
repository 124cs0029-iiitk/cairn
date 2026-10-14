"""Download NPTEL's English captions for every lecture in data/course/dbms.json.

Usage (from the project folder):  python scripts/fetch_captions.py
Videos without a manual English track are reported so they can be transcribed instead.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.captions import parse_vtt, pick_track, window_cues  # noqa: E402

OUT = ROOT / "data" / "raw" / "captions"


def fetch(yt_dlp, video_id: str) -> str | None:
    url = f"https://www.youtube.com/watch?v={video_id}"
    base = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(base) as ydl:
        info = ydl.extract_info(url, download=False)
    key = pick_track(info.get("subtitles") or {})
    if key is None:
        return None
    opts = {**base, "writesubtitles": True, "subtitleslangs": [key],
            "subtitlesformat": "vtt", "outtmpl": str(OUT / "%(id)s")}
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    produced = sorted(OUT.glob(f"{video_id}.*.vtt"))
    if not produced:
        raise RuntimeError("caption file was not written")
    produced[0].replace(OUT / f"{video_id}.vtt")
    return key


def main() -> None:
    import yt_dlp

    manifest = json.loads((ROOT / "data" / "course" / "dbms.json").read_text(encoding="utf-8"))
    videos = [s for s in manifest["sources"] if s["modality"] == "video"]
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    print(f"{'lecture':<42} {'track':<16} {'cues':>5} {'passages':>8} {'minutes':>8}")
    missing = []
    for v in videos:
        vid = v["youtube_id"]
        target = OUT / f"{vid}.vtt"
        try:
            if not target.exists():
                key = fetch(yt_dlp, vid)
                index[vid] = key
                time.sleep(2)  # be gentle with YouTube
            if not target.exists():
                missing.append(v["title"])
                print(f"{v['title']:<42} {'NONE':<16}   needs speech-to-text")
                continue
            cues = parse_vtt(target.read_text(encoding="utf-8"))
            segs = window_cues(cues)
            minutes = (cues[-1].end if cues else 0) / 60
            print(f"{v['title']:<42} {str(index.get(vid, '?'))[:16]:<16} "
                  f"{len(cues):>5} {len(segs):>8} {minutes:>8.1f}")
        except Exception as exc:  # keep going; report at the end
            missing.append(v["title"])
            print(f"{v['title']:<42} FAILED: {str(exc)[:60]}")
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\n{len(videos) - len(missing)} of {len(videos)} lectures have captions.")
    if missing:
        print("Need another source:", ", ".join(missing))


if __name__ == "__main__":
    main()
