"""t verifies:
text extraction works for plain text, PDFs, transcripts, and images
chunking respects size limits
ingesting many files behaves correctly
safe downloading and Kaggle dataset fetch logic work"""

import pytest

from surgidoc.ingest import chunk_text, extract_text, ingest_paths, download_public_texts, fetch_cholecseg8k

def make_pdf(text):
    """This is a helper that builds a minimal fake PDF in bytes.
It creates a valid PDF structure with:
catalog
pages
page content stream
font definition
It embeds the input text into the PDF stream so the test can assert that PDF extraction returns that text.
Why this exists:
real PDFs are annoying to generate, so this helper creates a tiny valid PDF for testing."""

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
    """verifies plain text files are read normally"""
    p = tmp_path / "a.txt"
    p.write_text("Colonoscopy guideline")
    assert extract_text(p) == "Colonoscopy guideline"


def test_extract_pdf(tmp_path):
    """Creates a fake PDF using make_pdf("Gastroscopy report")
Calls extract_text(p)
Asserts the extracted text contains “Gastroscopy report”
Purpose: verifies PDF parsing works and extracts visible text"""
    p = tmp_path / "a.pdf"
    p.write_bytes(make_pdf("Gastroscopy report"))
    assert "Gastroscopy report" in extract_text(p)


def test_extract_transcript_strips_timestamps(tmp_path):
    """Purpose: verifies transcript cleaning removes subtitle timing lines and keeps actual spoken content"""
    
    p = tmp_path / "v.vtt"
    p.write_text("WEBVTT\n\n1\n00:00:01.000 --> 00:00:03.000\nInsert scope\n\n00:00:04.000 --> 00:00:05.000\nInspect mucosa\n")
    assert extract_text(p) == "Insert scope Inspect mucosa"


def test_extract_image_uses_injected_ocr(tmp_path):
    """Purpose: verifies image extraction delegates to OCR logic and allows dependency injection for tests"""
    
    p = tmp_path / "scan.png"
    p.write_bytes(b"x")
    assert extract_text(p, ocr=lambda path: "ocr text") == "ocr text"


def test_unsupported_type_raises(tmp_path):
    p = tmp_path / "a.exe"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported"):
        extract_text(p)


def test_chunks_respect_max_size_and_paragraphs():
    """ verifies chunking respects max size while preserving natural text order"""
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 30 for i in range(10))
    chunks = chunk_text(text, max_chars=400)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)
    assert "Paragraph 0." in chunks[0]


def test_chunking_keeps_all_content():
    """Checks that all occurrences of “one” are still present in the combined chunk output"""
    text = "A one. A two.\n\nB one. B two."
    assert " ".join(chunk_text(text, max_chars=14)).replace("\n", " ").count("one") == 2


def test_oversized_sentence_is_hard_split():
    """Purpose: verifies the function can split a single oversized sentence without dropping or duplicating content"""
    
    chunks = chunk_text("x" * 1000, max_chars=300)
    assert all(len(c) <= 300 for c in chunks) and sum(map(len, chunks)) == 1000


def test_empty_text_yields_no_chunks():
    assert chunk_text("  \n ") == []


def test_ingest_paths_tags_source_and_index(tmp_path):
    """Purpose: validates that bulk ingestion labels chunks with their origin and sequence number"""
    
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


def test_extract_jats_xml_keeps_titles_and_paragraphs(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text("<article><front><article-title>ESGE guideline</article-title></front>"
                 "<body><sec><title>Methods</title><p>Use <italic>cold</italic> snare.</p></sec></body></article>")
    text = extract_text(p)
    assert "ESGE guideline" in text and "Use cold snare." in text
    assert "<" not in text


def test_ingest_root_makes_sources_relative_and_portable(tmp_path):
    (tmp_path / "kb").mkdir()
    p = tmp_path / "kb" / "a.txt"
    p.write_text("hello")
    out = ingest_paths([p], root=tmp_path)
    assert out[0].source == "kb/a.txt" and str(tmp_path) not in out[0].source


def test_ingest_root_rejects_paths_outside_root(tmp_path):
    root = tmp_path / "kb"
    root.mkdir()
    other = tmp_path / "x.txt"
    other.write_text("hi")
    with pytest.raises(ValueError, match="outside"):
        ingest_paths([other], root=root)


def test_ingest_without_root_keeps_full_path(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello")
    assert ingest_paths([p])[0].source == str(p)


# --- Kaggle credentials from keys.txt (T14) ----------------------------------------------------------------------
import logging
import os

from surgidoc.config import load_keys
from surgidoc.ingest import KaggleAuthError, kaggle_login

TOKEN = "KGAT_" + "a" * 32


class FakeAuth:
    def __init__(self, calls, whoami_error=None, username="alice"):
        self.calls, self.whoami_error, self.username = calls, whoami_error, username

    def set_kaggle_api_token(self, token):
        self.calls.append(("token", token))

    def set_kaggle_credentials(self, username, key):
        self.calls.append(("legacy", username, key))

    def whoami(self, verbose=True):
        self.calls.append(("whoami", verbose))
        if self.whoami_error:
            raise self.whoami_error
        return {"username": self.username}


class FakeKaggle:
    def __init__(self, **kw):
        self.calls = []
        self.auth = FakeAuth(self.calls, **kw)

    def dataset_download(self, slug):
        self.calls.append(("download", slug))
        return "/data/cholecseg8k"


def kaggle_keys(tmp_path, text):
    p = tmp_path / "keys.txt"
    p.write_text(text)
    return load_keys(p)


def test_a_kgat_value_in_kaggle_key_is_used_as_an_api_token(tmp_path):
    hub = FakeKaggle()
    path = fetch_cholecseg8k(kaggle_keys(tmp_path, f"KAGGLE_USERNAME=alice\nKAGGLE_KEY={TOKEN}\n"), hub=hub)
    assert path == "/data/cholecseg8k"
    assert hub.calls == [("token", TOKEN), ("whoami", False), ("download", "newslab/cholecseg8k")]  # verified before downloading


def test_kaggle_api_token_takes_precedence(tmp_path):
    hub = FakeKaggle()
    kaggle_login(kaggle_keys(tmp_path, f"KAGGLE_API_TOKEN=KGAT_{'b' * 32}\nKAGGLE_USERNAME=alice\nKAGGLE_KEY={TOKEN}\n"), hub=hub)
    assert hub.calls[0] == ("token", f"KGAT_{'b' * 32}")


def test_legacy_username_and_key_pair_is_supported(tmp_path):
    hub = FakeKaggle()
    kaggle_login(kaggle_keys(tmp_path, "KAGGLE_USERNAME=alice\nKAGGLE_KEY=0123456789abcdef\n"), hub=hub)
    assert hub.calls[0] == ("legacy", "alice", "0123456789abcdef")


def test_login_returns_the_authenticated_username(tmp_path):
    assert kaggle_login(kaggle_keys(tmp_path, f"KAGGLE_KEY={TOKEN}\n"), hub=FakeKaggle(username="bob")) == "bob"


def test_values_are_stripped_and_empty_values_count_as_missing(tmp_path):
    hub = FakeKaggle()
    kaggle_login(kaggle_keys(tmp_path, f"KAGGLE_API_TOKEN=\nKAGGLE_KEY=  {TOKEN}  \n"), hub=hub)
    assert hub.calls[0] == ("token", TOKEN)


@pytest.mark.parametrize("text", ["X=1\n", "KAGGLE_USERNAME=alice\n", "KAGGLE_KEY=0123456789abcdef\n"])
def test_missing_or_incomplete_credentials_fail_before_any_download(tmp_path, text):
    hub = FakeKaggle()
    with pytest.raises(KaggleAuthError, match="KAGGLE_API_TOKEN"):
        fetch_cholecseg8k(kaggle_keys(tmp_path, text), hub=hub)
    assert ("download", "newslab/cholecseg8k") not in hub.calls


def test_rejected_credentials_stop_the_download_and_never_leak_the_token(tmp_path):
    hub = FakeKaggle(whoami_error=RuntimeError(f"401 for token {TOKEN}"))
    with pytest.raises(KaggleAuthError) as e:
        fetch_cholecseg8k(kaggle_keys(tmp_path, f"KAGGLE_KEY={TOKEN}\n"), hub=hub)
    assert TOKEN not in str(e.value) and e.value.__cause__ is None and e.value.__suppress_context__
    assert ("download", "newslab/cholecseg8k") not in hub.calls


def test_without_keys_nothing_is_authenticated_for_backward_compatibility():
    hub = FakeKaggle()
    assert fetch_cholecseg8k(hub=hub) == "/data/cholecseg8k"
    assert hub.calls == [("download", "newslab/cholecseg8k")]


def test_the_token_is_never_logged_printed_or_written_anywhere(tmp_path, monkeypatch, caplog, capsys):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    before = dict(os.environ)
    with caplog.at_level(logging.DEBUG):
        fetch_cholecseg8k(kaggle_keys(tmp_path, f"KAGGLE_KEY={TOKEN}\n"), hub=FakeKaggle())
    out = capsys.readouterr()
    assert TOKEN not in caplog.text and TOKEN not in out.out + out.err
    assert dict(os.environ) == before and not any(k.startswith("KAGGLE") for k in os.environ if k not in before)
    assert list(home.iterdir()) == []  # nothing like ~/.kaggle/access_token was created


def test_download_errors_are_not_swallowed(tmp_path):
    class Broken(FakeKaggle):
        def dataset_download(self, slug):
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        fetch_cholecseg8k(kaggle_keys(tmp_path, f"KAGGLE_KEY={TOKEN}\n"), hub=Broken())


def test_missing_kagglehub_gives_an_actionable_message(tmp_path, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "kagglehub", None)  # makes `import kagglehub` raise ImportError
    with pytest.raises(ImportError, match=r"surgidoc\[kaggle\]"):
        fetch_cholecseg8k(kaggle_keys(tmp_path, f"KAGGLE_KEY={TOKEN}\n"))
