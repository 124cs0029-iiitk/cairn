import json
from pathlib import Path

from core.captions import Cue, clean_text, parse_timestamp, parse_vtt, pick_track, window_cues
from core.schema import Source

SAMPLE = """WEBVTT
Kind: captions
Language: en

00:00:10.800 --> 00:00:22.350
Welcome to module two of database&nbsp;
management systems. In this module,&nbsp;&nbsp;

00:00:22.350 --> 00:00:33.900
we will talk primarily about the introduction&nbsp;
to the DBMS. This discussion will span two&nbsp;&nbsp;

00:00:33.900 --> 00:00:39.240
modules that is the current one module&nbsp;
2 and the next one that is module 3.&nbsp;&nbsp;

00:00:39.240 --> 00:00:49.800 align:start
Just to quickly recap in the last module we&nbsp;
have discussed about why we need databases,&nbsp;&nbsp;

00:00:49.800 --> 00:00:53.580
and we have introduced you to&nbsp;
different aspects of the course.
"""


def test_parse_timestamp_forms():
    assert parse_timestamp("00:00:10.800") == 10.8
    assert parse_timestamp("01:02:03.500") == 3723.5
    assert parse_timestamp("02:03.250") == 123.25


def test_clean_text_removes_entities_and_tags():
    assert clean_text("a&nbsp;\nb <c>x</c>&nbsp;&nbsp;") == "a b x"


def test_parse_vtt_cues():
    cues = parse_vtt(SAMPLE)
    assert len(cues) == 5
    assert cues[0].start == 10.8 and cues[0].end == 22.35
    assert cues[0].text == "Welcome to module two of database management systems. In this module,"
    assert "\xa0" not in cues[1].text and "&nbsp;" not in cues[1].text


def test_parse_vtt_drops_repeated_cue_and_bad_timestamps():
    content = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n\n00:00:02.000 --> 00:00:03.000\nhi\n\nxx --> yy\nbad\n"
    assert [c.text for c in parse_vtt(content)] == ["hi"]


def test_windows_end_on_sentence_and_keep_start_time():
    segs = window_cues(parse_vtt(SAMPLE), target_s=30, max_s=60, min_s=5)
    assert segs[0].start == 10.8
    assert segs[0].text.rstrip().endswith(".")
    assert segs[0].end - segs[0].start >= 30
    assert all(s.text for s in segs)


def test_short_last_window_is_merged():
    cues = [Cue(0, 50, "One long thought."), Cue(50, 55, "Tail.")]
    segs = window_cues(cues, target_s=45, max_s=75, min_s=15)
    assert len(segs) == 1 and segs[0].text == "One long thought. Tail."


def test_pick_track_prefers_verified_english():
    subs = {
        "bn": [{"name": "Bangla"}],
        "en-6UJrWS5jR_I": [{"name": "English - NPTEL Verified"}],
        "en-PVQ7OO5RyjI": [{"name": "English - NPTEL-Testing"}],
        "hi": [{"name": "Hindi"}],
    }
    assert pick_track(subs) == "en-6UJrWS5jR_I"


def test_pick_track_fallbacks():
    assert pick_track({"en": [{"name": "English"}], "hi": [{"name": "Hindi"}]}) == "en"
    assert pick_track({"hi": [{"name": "Hindi"}]}) is None
    assert pick_track({}) is None


def test_course_manifest_is_valid():
    manifest = json.loads((Path(__file__).resolve().parent.parent / "data/course/dbms.json").read_text())
    ids = set()
    groups = {g["id"] for g in manifest["groups"]}
    for s in manifest["sources"]:
        src = Source(**{k: v for k, v in s.items() if k in Source.model_fields})
        assert src.id not in ids
        ids.add(src.id)
        assert s["group"] is None or s["group"] in groups
    assert sum(s["modality"] == "video" for s in manifest["sources"]) == 16
    assert sum(s["modality"] == "slides" for s in manifest["sources"]) == 5
