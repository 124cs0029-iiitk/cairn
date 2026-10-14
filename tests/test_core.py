import pytest

from core import citations, db
from core.config import get_settings, role_status
from core.llm import RateLimiter, RateLimitExceeded, cache_key
from core.schema import Chunk, Locator, Modality, Source


def make_sources():
    pdf = Source(id="book", title="ISLR ch.3", modality=Modality.PDF, static_path="courses/ml/islr3.pdf")
    slides = Source(id="lec3", title="Lecture 3", modality=Modality.SLIDES, static_path="courses/ml/lec3.pdf")
    video = Source(id="v3", title="Lecture 3 video", modality=Modality.VIDEO, youtube_id="abc123")
    return pdf, slides, video


def test_pdf_and_slide_sources_require_path():
    with pytest.raises(ValueError):
        Source(id="x", title="x", modality=Modality.PDF)
    with pytest.raises(ValueError):
        Source(id="x", title="x", modality=Modality.VIDEO)


def test_chunk_requires_matching_locator_and_content():
    with pytest.raises(ValueError):
        Chunk(id="c", source_id="s", modality=Modality.SLIDES, locator=Locator(), text="hi")
    with pytest.raises(ValueError):
        Chunk(id="c", source_id="s", modality=Modality.VIDEO, locator=Locator(page=2), text="hi")
    with pytest.raises(ValueError):
        Chunk(id="c", source_id="s", modality=Modality.PDF, locator=Locator(page=1), text="  ")


def test_figure_caption_alone_is_valid_content():
    c = Chunk(id="c", source_id="s", modality=Modality.PDF, locator=Locator(page=4),
              figure_caption="Plot of loss versus epochs")
    assert "loss" in c.searchable_text


def test_slide_citation_deep_link():
    _, slides, _ = make_sources()
    chunk = Chunk(id="c1", source_id="lec3", modality=Modality.SLIDES,
                  locator=Locator(page=12), text="Gradient descent updates weights.")
    cit = citations.build_citation(chunk, slides)
    assert cit.url == "/app/static/courses/ml/lec3.pdf#page=12"
    assert cit.label == "Lecture 3, slide 12"


def test_video_citation_deep_link_and_timestamp():
    _, _, video = make_sources()
    chunk = Chunk(id="c2", source_id="v3", modality=Modality.VIDEO,
                  locator=Locator(start_s=754.8), text="Now the learning rate.")
    cit = citations.build_citation(chunk, video)
    assert cit.url == "https://www.youtube.com/watch?v=abc123&t=754s"
    assert cit.label == "Lecture 3 video at 12:34"


def test_citation_rejects_wrong_source_and_truncates_excerpt():
    pdf, slides, _ = make_sources()
    chunk = Chunk(id="c3", source_id="book", modality=Modality.PDF,
                  locator=Locator(page=1), text="word " * 200)
    with pytest.raises(ValueError):
        citations.build_citation(chunk, slides)
    assert len(citations.build_citation(chunk, pdf).excerpt) <= 280


def test_fmt_timestamp():
    assert citations.fmt_timestamp(59) == "0:59"
    assert citations.fmt_timestamp(3725) == "1:02:05"


def test_db_roundtrip(tmp_path):
    with db.session(tmp_path / "t.db") as conn:
        first = db.get_or_create_learner(conn, "u1", "Asha")
        again = db.get_or_create_learner(conn, "u1")
        assert first["id"] == again["id"] == "u1"
        assert again["name"] == "Asha"
        conn.execute(
            "INSERT INTO questions (id, concept_id, chunk_id, qtype, difficulty, payload, created_at)"
            " VALUES ('q1','c','k','mcq',2,'{}',0)")
        conn.execute(
            "INSERT INTO attempts (learner_id, question_id, concept_id, correct, ts)"
            " VALUES ('u1','q1','c',1,0)")
        with pytest.raises(Exception):
            conn.execute("INSERT INTO questions (id, concept_id, chunk_id, qtype, difficulty, payload, created_at)"
                         " VALUES ('q2','c','k','essay',2,'{}',0)")


def test_rate_limiter_blocks_then_recovers():
    rl = RateLimiter(max_calls=2, window_s=10)
    rl.check("a", now=0)
    rl.check("a", now=1)
    with pytest.raises(RateLimitExceeded):
        rl.check("a", now=2)
    rl.check("b", now=2)
    rl.check("a", now=12)


def test_cache_key_depends_on_inputs():
    m = [{"role": "user", "content": "hi"}]
    assert cache_key("x", m, 0.0, False) == cache_key("x", m, 0.0, False)
    assert cache_key("x", m, 0.0, False) != cache_key("y", m, 0.0, False)
    assert cache_key("x", m, 0.0, False) != cache_key("x", m, 0.7, False)


def test_settings_and_role_status():
    s = get_settings()
    assert s.model_for("generator") and s.model_for("verifier")
    with pytest.raises(ValueError):
        s.model_for("nonsense")
    assert set(role_status()) == {"generator", "verifier", "judge", "vlm", "stt"}
