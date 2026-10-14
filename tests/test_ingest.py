import pymupdf

from core.ingest import (Block, PageData, clean_blocks, detect_boilerplate, normalise, pdf_chunks, split_blocks, video_chunks)
from core.schema import Modality, Source
from tests.test_captions import SAMPLE


def make_deck(path, n_pages, figure_pages=(), footer=True, template_shape=True):
    """A synthetic deck: repeated footer, page number and logo shape on every page."""
    doc = pymupdf.open()
    for i in range(1, n_pages + 1):
        page = doc.new_page()
        page.insert_text((72, 100), f"Topic {i}", fontsize=16)
        page.insert_text((72, 160), f"Body sentence for slide {i} about keys.", fontsize=12)
        if i == 1:
            page.insert_text((72, 60), "All rights reserved by Acme", fontsize=10)
        if i == 2:
            page.insert_text((72, 220), "\u2022", fontsize=12)
            page.insert_text((110, 260), "Item text here", fontsize=12)
        if footer:
            page.insert_text((72, 810), "Acme Course Notes", fontsize=9)
            page.insert_text((400, 825), f"{i}.{i}", fontsize=9)
        if template_shape:
            page.draw_rect(pymupdf.Rect(10, 10, 60, 30))
        if i in figure_pages:
            for k in range(10):
                page.draw_rect(pymupdf.Rect(300, 300 + k * 20, 340, 312 + k * 20))
    doc.save(path)
    doc.close()


def source(modality=Modality.SLIDES):
    return Source(id="s", title="Deck", modality=modality, static_path="courses/x.pdf")


def test_normalise_ignores_page_numbers():
    assert normalise("vi \u2022 DATABASE DESIGN - 2ND EDITION") == normalise("14 \u2022 DATABASE DESIGN - 2ND EDITION")
    assert normalise("7.12") == normalise("3.4") == "#.#"
    assert normalise("DATABASE DESIGN - 2ND EDITION \u2022 14") == normalise("DATABASE DESIGN - 2ND EDITION \u2022 viii")


def test_detect_boilerplate_needs_repetition_in_header_or_footer_bands():
    words = ["Keys", "Joins", "Views", "Indexes", "Triggers", "Cursors", "Roles", "Backups"]

    def page(i):
        return PageData([Block(f"All about {words[i - 1]}", "top"), Block("Body line", "body"),
                         Block(f"My Course \u2022 {i}", "bottom"), Block(f"Header {i}", "top")])
    pages = [page(i) for i in range(1, 9)]
    found = detect_boilerplate(pages)
    assert normalise("My Course \u2022 1") in found
    assert normalise("Header 1") in found                       # differs only by a number
    assert normalise("Body line") not in found                  # body text never counts
    assert all("all about" not in k for k in found)             # titles with different words
    assert detect_boilerplate(pages[:3]) == {}                  # too few pages to judge


def test_clean_blocks_legal_lines_and_bullets():
    blocks = [Block("Copyright 2020 Someone", "body"), Block("\u2022", "body"),
              Block("Keys are unique", "body"), Block("Plain text", "body")]
    assert clean_blocks(blocks, {}) == ["\u2022 Keys are unique", "Plain text"]

def test_symbol_font_bullets_become_normal_bullets():
    assert tidy_symbols("closure of F. \uf0a7 We denote") == "closure of F. \u2022 We denote"
    blocks = [Block(tidy_symbols("\uf0a7"), "body"), Block("Keys are unique", "body")]
    assert clean_blocks(blocks, {}) == ["\u2022 Keys are unique"]
def test_split_blocks_respects_limit():
    blocks = ["a" * 50, "b" * 50, "c" * 50]
    assert split_blocks(blocks, 120) == ["a" * 50 + "\n" + "b" * 50, "c" * 50]
    assert split_blocks(["x" * 500], 100) == ["x" * 500]


def test_pdf_chunks_remove_template_text_keep_content_and_pages(tmp_path):
    pdf = tmp_path / "deck.pdf"
    make_deck(pdf, 8)
    chunks, _, stats = pdf_chunks(source(), pdf)
    assert [c.locator.page for c in chunks] == list(range(1, 9))
    joined = " ".join(c.text for c in chunks)
    assert "Acme Course Notes" not in joined and "All rights reserved" not in joined
    assert "3.3" not in joined
    assert "Topic 3" in chunks[2].text and "Body sentence for slide 3" in chunks[2].text
    assert any(line.endswith(" Item text here") and line[0] in "\u2022\u00b7"
               for line in chunks[1].text.split("\n"))
    assert stats["boilerplate"] and stats["empty_pages"] == 0


def test_figure_flag_ignores_template_graphics(tmp_path):
    pdf = tmp_path / "deck.pdf"
    make_deck(pdf, 8, figure_pages={5})
    chunks, figs, stats = pdf_chunks(source(), pdf)
    assert [f["page"] for f in figs] == [5]
    assert [c.has_figure for c in chunks] == [i == 5 for i in range(1, 9)]
    assert stats["figure_pages"] == 1


def test_small_shapes_such_as_bullets_are_not_figures(tmp_path):
    pdf = tmp_path / "bullets.pdf"
    doc = pymupdf.open()
    for i in range(1, 7):
        page = doc.new_page()
        page.insert_text((72, 100), f"Slide {i} with bullets", fontsize=14)
        for k in range(8):                       # a different bullet layout on every page
            page.draw_circle((60 + i, 150 + k * 30 + i), 3)
    doc.save(pdf)
    doc.close()
    _, figs, _ = pdf_chunks(source(), pdf)
    assert figs == []


def test_repeated_logo_with_different_ids_is_template_but_real_image_is_figure(tmp_path):
    pdf = tmp_path / "logos.pdf"
    doc = pymupdf.open()
    for i in range(1, 9):
        page = doc.new_page()
        page.insert_text((72, 200), f"Slide {i} text", fontsize=14)
        # same size and place on every page, different pixel content, so a different internal id
        logo = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
        logo.set_rect(logo.irect, (i * 20, 10, 10))
        page.insert_image(pymupdf.Rect(20, 300, 120, 400), pixmap=logo)
        if i == 5:
            photo = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 40))
            photo.set_rect(photo.irect, (200, 30, 30))
            page.insert_image(pymupdf.Rect(150, 450, 400, 650), pixmap=photo)
    doc.save(pdf)
    doc.close()
    _, figs, _ = pdf_chunks(source(), pdf)
    assert [f["page"] for f in figs] == [5]


def test_title_banner_images_of_varying_size_are_not_figures(tmp_path):
    pdf = tmp_path / "banners.pdf"
    doc = pymupdf.open()
    for i in range(1, 9):
        page = doc.new_page()
        page.insert_text((72, 120), f"Slide {i} text", fontsize=14)
        banner = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 30 + i, 10))
        banner.set_rect(banner.irect, (10, 10 * i, 10))
        page.insert_image(pymupdf.Rect(20, 10, 100 + i * 40, 70), pixmap=banner)   # wider every slide
        if i == 5:
            photo = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 40))
            photo.set_rect(photo.irect, (200, 30, 30))
            page.insert_image(pymupdf.Rect(150, 300, 400, 500), pixmap=photo)
    doc.save(pdf)
    doc.close()
    _, figs, _ = pdf_chunks(source(), pdf)
    assert [f["page"] for f in figs] == [5]


def test_full_width_boxes_and_rules_are_not_figures(tmp_path):
    pdf = tmp_path / "boxes.pdf"
    doc = pymupdf.open()
    for i in range(1, 9):
        page = doc.new_page()
        page.insert_text((72, 100), f"Page {i} text", fontsize=12)
        for k in range(4):                       # code boxes and table rules as wide as the text column
            page.draw_rect(pymupdf.Rect(57, 200 + k * 60 + i, 555, 240 + k * 60 + i))
        if i == 5:
            for k in range(4):                   # a real diagram: narrow boxes
                page.draw_rect(pymupdf.Rect(100 + k * 90, 600, 160 + k * 90, 650))
    doc.save(pdf)
    doc.close()
    _, figs, _ = pdf_chunks(source(), pdf)
    assert [f["page"] for f in figs] == [5]

def test_short_documents_are_not_stripped_by_repetition(tmp_path):
    pdf = tmp_path / "short.pdf"
    make_deck(pdf, 3)
    chunks, _, _ = pdf_chunks(source(), pdf)
    assert "Acme Course Notes" in chunks[0].text       # too short to tell footers from content


def test_video_chunks_use_start_time(tmp_path):
    vtt = tmp_path / "v.vtt"
    vtt.write_text(SAMPLE, encoding="utf-8")
    src = Source(id="video_abc", title="L1", modality=Modality.VIDEO, youtube_id="abc")
    chunks = video_chunks(src, vtt)
    assert chunks and chunks[0].locator.start_s == 10.8
    assert chunks[0].id == "video_abc:t10"
    assert all(c.modality is Modality.VIDEO for c in chunks)