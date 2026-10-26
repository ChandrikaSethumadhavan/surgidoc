"""t verifies:
text extraction works for plain text, PDFs, transcripts, and images
chunking respects size limits
ingesting many files behaves correctly
safe downloading and Kaggle dataset fetch logic work"""

import pytest

from surgidoc.ingest import chunk_text, extract_text, ingest_paths, download_public_texts, fetch_cholecseg8k

def make_pdf(text):
    stream = f"BT /F1 12 Tf 10 50 Td ({text}) Tj ET".encode()
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 100]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length %d>>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 6\n0000000000 65535 f \n" + b"".join(b"%010d 00000 n \n" % o for o in offs)
    return out + b"trailer<</Root 1 0 R/Size 6>>\nstartxref\n%d\n%%%%EOF" % xref


def test_extract_plain_text(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("Colonoscopy guideline")
    assert extract_text(p) == "Colonoscopy guideline"


def test_extract_pdf(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(make_pdf("Gastroscopy report"))
    assert "Gastroscopy report" in extract_text(p)


def test_extract_transcript_strips_timestamps(tmp_path):
    p = tmp_path / "v.vtt"
    p.write_text("WEBVTT\n\n1\n00:00:01.000 --> 00:00:03.000\nInsert scope\n\n00:00:04.000 --> 00:00:05.000\nInspect mucosa\n")
    assert extract_text(p) == "Insert scope Inspect mucosa"


def test_extract_image_uses_injected_ocr(tmp_path):
    p = tmp_path / "scan.png"
    p.write_bytes(b"x")
    assert extract_text(p, ocr=lambda path: "ocr text") == "ocr text"


def test_unsupported_type_raises(tmp_path):
    p = tmp_path / "a.exe"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported"):
        extract_text(p)


def test_chunks_respect_max_size_and_paragraphs():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 30 for i in range(10))
    chunks = chunk_text(text, max_chars=400)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)
    assert "Paragraph 0." in chunks[0]


def test_chunking_keeps_all_content():
    text = "A one. A two.\n\nB one. B two."
    assert " ".join(chunk_text(text, max_chars=14)).replace("\n", " ").count("one") == 2


def test_oversized_sentence_is_hard_split():
    chunks = chunk_text("x" * 1000, max_chars=300)
    assert all(len(c) <= 300 for c in chunks) and sum(map(len, chunks)) == 1000


def test_empty_text_yields_no_chunks():
    assert chunk_text("  \n ") == []


def test_ingest_paths_tags_source_and_index(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("one\n\ntwo")
    out = ingest_paths([p], max_chars=100)
    assert [c.source for c in out] == [str(p)] * len(out)
    assert [c.index for c in out] == list(range(len(out)))


def test_ingest_skips_unreadable_but_reports(tmp_path):
    bad = tmp_path / "a.exe"
    bad.write_bytes(b"x")
    good = tmp_path / "b.txt"
    good.write_text("ok")
    errors = []
    out = ingest_paths([bad, good], on_error=lambda p, e: errors.append(p))
    assert len(out) == 1 and errors == [bad]


def test_download_public_texts_uses_injected_fetch(tmp_path):
    saved = download_public_texts({"a.txt": "http://x/a"}, tmp_path, fetch=lambda url: b"hello " + url.encode())
    assert saved[0].read_text() == "hello http://x/a"


def test_download_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        download_public_texts({"../evil.txt": "http://x"}, tmp_path, fetch=lambda u: b"")


def test_fetch_cholecseg8k_calls_kagglehub():
    class FakeHub:
        def dataset_download(self, slug):
            assert slug == "newslab/cholecseg8k"
            return "/data/path"

    assert fetch_cholecseg8k(hub=FakeHub()) == "/data/path"
