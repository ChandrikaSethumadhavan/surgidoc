"""FR-3, FR-4: loading the guideline PDFs into the knowledge base: page-aware chunks, resumable embedding, CLI."""
import urllib.error

import pytest
from fpdf import FPDF

from surgidoc import kb
from surgidoc.ingest import Chunk, clean_pages, guess_lang, guess_title, is_pdf, looks_like_references, pdf_chunks
from surgidoc.retrieval import MemoryStore, PgStore

def para(i, times=1):  # distinct text per page: identical lines on many pages would (rightly) count as a running header
    return (f"Propofol sedation number {i} should be monitored with capnography during gastrointestinal endoscopy. "
            f"Patients in group {i} must be assessed before the procedure and observed until they have recovered. ") * times


PARA = para(0)


def write_pdf(path, pages, header=None):
    pdf = FPDF()
    pdf.set_font("Helvetica", size=10)
    for i, text in enumerate(pages, 1):
        pdf.add_page()
        if header:
            pdf.cell(0, 6, header, new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 5, text)
        pdf.cell(0, 6, str(i))  # a bare page number
    pdf.output(str(path))
    return path


@pytest.fixture
def folder(tmp_path):
    write_pdf(tmp_path / "sedation.pdf", ["Sedation for endoscopy: European Guideline Authors A. B. " + para(1, 3), para(2, 4), para(3, 2), para(4, 2), para(5, 2)],
              header="Endoscopy 2026; 58")
    write_pdf(tmp_path / "leitlinie", ["Qualitätsanforderungen in der Endoskopie Autoren X. " + "Die Patienten werden vor dem Eingriff aufgeklärt und der Eingriff wird dokumentiert. " * 4])
    (tmp_path / "sedation.pdf:Zone.Identifier").write_text("[ZoneTransfer]\nZoneId=3\n")
    (tmp_path / "notes.txt").write_text("not a pdf")
    return tmp_path


class Emb:
    def __init__(self, fail=()):
        self.calls, self.fail = [], list(fail)

    def embed_documents(self, texts):
        self.calls.append(len(texts))
        if self.fail:
            code = self.fail.pop(0)
            if code:
                raise urllib.error.HTTPError("u", code, "x", {}, None)
        return [[1.0, float(len(t) % 7)] for t in texts]


def run(folder, store=None, emb=None, **kw):
    slept = []
    store = store or MemoryStore()
    out = kb.load(folder, store, emb or Emb(), sleep=slept.append, log=lambda *_: None, **kw)
    return out, store, slept


# --- reading PDFs -----------------------------------------------------------------------------------------
def test_pdfs_are_found_by_content_and_sidecars_are_ignored(folder):
    assert [p.name for p in kb.discover(folder)] == ["leitlinie", "sedation.pdf"]
    assert is_pdf(folder / "leitlinie") and not is_pdf(folder / "notes.txt") and not is_pdf(folder / "sedation.pdf:Zone.Identifier")


def test_chunks_carry_page_title_and_language(folder):
    en = pdf_chunks(folder / "sedation.pdf", "sedation.pdf", max_chars=300)
    assert {c.page for c in en} == {1, 2, 3, 4, 5} and [c.index for c in en] == list(range(len(en)))
    assert en[0].title == "Sedation for endoscopy: European Guideline" and en[0].lang == "en"
    de = pdf_chunks(folder / "leitlinie", "leitlinie")
    assert de[0].lang == "de" and de[0].title.startswith("Qualitätsanforderungen") and de[0].source == "leitlinie"


def test_running_headers_and_page_numbers_are_removed():
    pages = [f"Endoscopy 2026; 58\n{i}\nText of page {i} continues\nover line number {i}." for i in range(1, 8)]
    out = clean_pages(pages)
    assert all("Endoscopy 2026" not in p and not p.startswith(str(i)) for i, p in enumerate(out, 1))
    assert out[2] == "Text of page 3 continues over line number 3."


def test_hyphenated_line_breaks_are_joined_and_blank_lines_kept():
    assert clean_pages(["seda-\ntion is used.\n\nNew paragraph."]) == ["sedation is used.\n\nNew paragraph."]


def test_reference_lists_are_not_stored():
    refs = "Smith J. Endoscopy 2019; 51: 1-9. Doe A. Gut 2020; 69: 12-20. doi:10.1055/x PMID 123"
    assert looks_like_references(refs) and not looks_like_references(PARA)


def test_title_and_language_guessing():
    assert guess_title(["Electronic reprint for personal use Advanced imaging: ESGE Review Authors J. East"], "f") == "Advanced imaging: ESGE Review"
    assert guess_title([""], "fallback") == "fallback" and guess_lang(["die und der nicht mit die"]) == "de" and guess_lang(["the and of is"]) == "en"


# --- loading ----------------------------------------------------------------------------------------------
def test_load_stores_every_chunk_with_its_metadata(folder):
    out, store, _ = run(folder)
    assert out["stopped"] is None and out["remaining"] == 0 and out["stored_now"] == out["chunks"] > 2
    assert {s for s, _, _ in store.counts()} == {"leitlinie", "sedation.pdf"}
    assert all(c.title and c.page and c.lang for c in store.chunks.values())


def test_a_second_run_embeds_nothing(folder):
    _, store, _ = run(folder)
    emb = Emb()
    out, _, _ = run(folder, store, emb)
    assert emb.calls == [] and out["stored_now"] == 0 and out["stored_before"] == out["chunks"]


def test_a_quota_stop_keeps_progress_and_the_next_run_finishes(folder):
    store = MemoryStore()
    out, _, slept = run(folder, store, Emb(fail=[None, 429, 429, 429]), batch=2)
    assert out["stopped"] and out["remaining"] > 0 and 0 < out["stored_now"] < out["chunks"] and slept.count(65) == 2  # per-minute retries first
    first = len(store.chunks)
    out2, _, _ = run(folder, store, Emb(), batch=2)
    assert out2["stopped"] is None and out2["remaining"] == 0 and out2["stored_before"] == first
    assert len(store.chunks) == out["chunks"]  # no duplicates, nothing missing


def test_a_per_minute_limit_is_waited_out(folder):
    out, _, slept = run(folder, emb=Emb(fail=[429, None, None, None, None]))
    assert out["stopped"] is None and slept[0] == 65


def test_other_errors_are_not_swallowed(folder):
    with pytest.raises(urllib.error.HTTPError):
        run(folder, emb=Emb(fail=[500]))


def test_max_chunks_limits_one_run(folder):
    out, store, _ = run(folder, max_chunks=3, batch=2)
    assert out["stored_now"] == 3 == len(store.chunks) and out["stopped"] and out["remaining"] == out["chunks"] - 3


def test_replace_reloads_a_document(folder):
    _, store, _ = run(folder)
    store.chunks["leitlinie#0"] = Chunk("stale", "leitlinie", 0)
    run(folder, store, replace=True)
    assert store.chunks["leitlinie#0"].text != "stale"


def test_embedding_is_throttled_to_the_token_budget(folder):
    _, _, slept = run(folder, tpm=100)
    assert slept and max(slept) > 1.0


def test_an_unreadable_pdf_is_skipped_not_fatal(folder):
    (folder / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    out, _, _ = run(folder)
    assert out["documents"] == 2


# --- Postgres store contract -----------------------------------------------------------------------------
class Cur:
    def __init__(self, rows=()):
        self.log, self.rows = [], list(rows)

    def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class Conn:
    def __init__(self, rows=()):
        self.cur, self.commits = Cur(rows), 0

    def cursor(self):
        return self.cur

    def commit(self):
        self.commits += 1


def test_pg_store_keeps_page_title_and_language():
    conn = Conn()
    PgStore(conn, dim=2).add([Chunk("t", "doc.pdf", 4, 12, "A Title", "de")], [[0.5, 0.5]])
    sql, params = conn.cur.log[-1]
    assert "title, page, lang" in sql and params[:3] == ("doc.pdf#4", "doc.pdf", 4) and params[-3:] == ("A Title", 12, "de")
    PgStore(conn, dim=2).setup()
    assert any("ADD COLUMN IF NOT EXISTS title text" in s and "page int" in s for s, _ in conn.cur.log)
    conn.cur.rows = [("a#0", "txt", "a", 0, 7, "T", "en")]
    (c,) = PgStore(conn, dim=2).get(["a#0"])
    assert (c.page, c.title, c.lang) == (7, "T", "en")


def test_pg_store_lists_deletes_and_counts_by_document():
    conn = Conn([("a#0",), ("a#1",)])
    s = PgStore(conn, dim=2)
    assert s.ids_for("a") == {"a#0", "a#1"} and conn.cur.log[-1][1] == ("a",)
    s.delete_source("a")
    assert conn.cur.log[-1][0].startswith("DELETE FROM chunks WHERE source = %s") and conn.commits == 1
    conn.cur.rows = [("a", 2, 9)]
    assert s.counts() == [("a", 2, 9)]


# --- command line ---------------------------------------------------------------------------------------
def test_dry_run_needs_no_keys_and_no_network(folder, tmp_path):
    lines = []
    total = kb.main(["load", str(folder), "--dry-run", "--keys", str(tmp_path / "missing.txt")], log=lines.append)
    assert total > 2 and any("leitlinie" in l and "de" in l for l in lines) and "embedding requests" in lines[-1]


def test_load_and_status_commands(folder, tmp_path):
    keys = tmp_path / "keys.txt"
    keys.write_text("GOOGLE_AI_STUDIO_API_KEY=k\n")
    store = MemoryStore()

    class Embedder(Emb):
        @classmethod
        def from_keys(cls, k):
            return cls()

    class Fake(PgStore):  # same interface, memory behind it
        def __init__(self, conn):
            pass

        def __getattribute__(self, name):
            return getattr(store, name) if name in ("add", "ids_for", "delete_source", "counts") else (lambda: None) if name == "setup" else object.__getattribute__(self, name)

    lines = []
    import surgidoc.kb as k
    old = k.PgStore
    k.PgStore = Fake
    try:
        res = kb.main(["load", str(folder), "--keys", str(keys)], connect=lambda keys: None, embedder_cls=Embedder, log=lines.append)
        assert res["remaining"] == 0 and "stored now" in lines[-1]
        rows = kb.main(["status", "--keys", str(keys)], connect=lambda keys: None, log=lines.append)
        assert {r[0] for r in rows} == {"leitlinie", "sedation.pdf"} and "documents" in lines[-1]
    finally:
        k.PgStore = old


def test_load_without_the_database_key_names_the_missing_entry(folder, tmp_path, capsys):
    keys = tmp_path / "keys.txt"
    keys.write_text("GOOGLE_AI_STUDIO_API_KEY=k\n")
    with pytest.raises(SystemExit):
        kb.main(["load", str(folder), "--keys", str(keys)])
    assert "TIGERCLOUD_SERVICE_URL" in capsys.readouterr().err


# --- retrieval evaluation (the proof that RAG finds the right guideline) ---------------------------------
def doc_store():
    s = MemoryStore()
    s.add([Chunk("Informed consent must describe risks benefits and alternatives of endoscopy.", "consent.pdf", 0, 3, "Consent", "en"),
           Chunk("Reprocessing of flexible endoscopes requires manual cleaning and disinfection.", "reproc.pdf", 0, 9, "Reprocessing", "en")],
          [[1.0, 0.0], [0.0, 1.0]])
    return s


def keyword_search(store):
    from surgidoc.retrieval import term_overlap_rerank
    return lambda q: term_overlap_rerank(q, store.get(store.bm25(q, 30)))[:5]


def test_evaluate_scores_rank_hit_rate_and_mrr_and_skips_unloaded_documents():
    s = doc_store()
    qs = [("What must informed consent describe?", {"consent.pdf"}),
          ("How are flexible endoscopes reprocessed?", {"reproc.pdf"}),
          ("Totally unrelated question about sailing", {"consent.pdf"}),
          ("Question about a document that is not loaded", {"missing.pdf"})]
    res = kb.evaluate(qs, keyword_search(s), {"consent.pdf", "reproc.pdf"})
    assert [r.get("rank") for r in res["rows"]] == [1, 1, None, None] and res["rows"][3]["skipped"]
    assert res["asked"] == 3 and res["skipped"] == 1 and res["hit_at_5"] == pytest.approx(2 / 3) and res["mrr"] == pytest.approx(2 / 3)
    assert res["rows"][0]["top"] == {"source": "consent.pdf", "page": 3}


def test_evaluate_with_nothing_loaded_asks_nothing():
    res = kb.evaluate(kb.EVAL_SET, lambda q: [], set())
    assert res["asked"] == 0 and res["skipped"] == len(kb.EVAL_SET) and res["hit_at_5"] == 0.0


def test_the_eval_set_names_documents_that_exist_in_the_dataset_listing():
    expected = {d for _, docs in kb.EVAL_SET for d in docs}
    assert "2025_a-2652-8941.pdf" in expected and all(d.endswith(".pdf") or d == "s-0041-109520" for d in expected)


def test_eval_command_writes_a_result_file(tmp_path):
    keys = tmp_path / "keys.txt"
    keys.write_text("X=1\n")
    store, lines = doc_store(), []
    import surgidoc.kb as k
    old = k.PgStore
    k.PgStore = lambda conn: type("S", (), {"setup": lambda self: None, "counts": lambda self: store.counts(),
                                             "bm25": lambda self, q, n: store.bm25(q, n), "get": lambda self, ids: store.get(ids)})()
    try:
        res = kb.main(["eval", "--keyword-only", "--keys", str(keys), "--out", str(tmp_path / "out" / "eval.json")],
                      connect=lambda keys: None, log=lines.append)
    finally:
        k.PgStore = old
    saved = __import__("json").loads((tmp_path / "out" / "eval.json").read_text())
    assert saved["mode"] == "keyword-only" and saved["asked"] == res["asked"] == 0 and "hit@5" in lines[-1]
