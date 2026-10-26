import math

import pytest

from surgidoc.config import MissingKey, load_keys
from surgidoc.ingest import Chunk
from surgidoc.retrieval import (GeminiEmbedder, MemoryStore, PgStore, connect_tiger, normalize,
                                retrieve, rrf, term_overlap_rerank)


def keys(tmp_path, text):
    p = tmp_path / "keys.txt"
    p.write_text(text)
    return load_keys(p)


# --- rrf / normalize ---------------------------------------------------------
def test_rrf_rewards_agreement_between_lists():
    fused = rrf([["a", "b", "c"], ["c", "a", "d"]])
    assert [i for i, _ in fused][:2] == ["a", "c"]
    assert dict(fused)["a"] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_respects_top_and_handles_empty():
    assert len(rrf([[str(i) for i in range(50)]], top=30)) == 30
    assert rrf([[], []]) == []


def test_normalize_unit_length_and_zero_vector_safe():
    v = normalize([3.0, 4.0])
    assert math.isclose(sum(x * x for x in v), 1.0)
    assert normalize([0.0, 0.0]) == [0.0, 0.0]


# --- embedder ----------------------------------------------------------------
def test_embedder_request_shape_and_normalization():
    calls = []

    def post(url, headers, body):
        calls.append((url, headers, body))
        return {"embeddings": [{"values": [3.0, 4.0]} for _ in body["requests"]]}

    e = GeminiEmbedder("SECRET", dim=2, post=post)
    out = e.embed_documents(["a", "b"])
    url, headers, body = calls[0]
    assert url.endswith("/models/gemini-embedding-001:batchEmbedContents")
    assert "SECRET" not in url and headers["x-goog-api-key"] == "SECRET"
    r = body["requests"][0]
    assert r["taskType"] == "RETRIEVAL_DOCUMENT" and r["outputDimensionality"] == 2
    assert r["model"] == "models/gemini-embedding-001"
    assert out == [[0.6, 0.8], [0.6, 0.8]]


def test_embedder_query_uses_query_task_and_batches_large_input():
    sizes = []

    def post(url, headers, body):
        sizes.append(len(body["requests"]))
        return {"embeddings": [{"values": [1.0, 0.0]} for _ in body["requests"]]}

    e = GeminiEmbedder("k", dim=2, post=post, batch=100)
    e.embed_documents(["t"] * 250)
    assert sizes == [100, 100, 50]
    e.embed_query("q")
    assert sizes[-1] == 1


def test_embedder_rejects_wrong_count_or_dim():
    e = GeminiEmbedder("k", dim=3, post=lambda u, h, b: {"embeddings": [{"values": [1.0]}]})
    with pytest.raises(ValueError):
        e.embed_documents(["a"])
    e2 = GeminiEmbedder("k", dim=3, post=lambda u, h, b: {"embeddings": []})
    with pytest.raises(ValueError):
        e2.embed_documents(["a"])


def test_embedder_from_keys_requires_google_key(tmp_path):
    with pytest.raises(MissingKey, match="GOOGLE_AI_STUDIO_API_KEY"):
        GeminiEmbedder.from_keys(keys(tmp_path, "X=1\n"))


# --- memory store + retrieve -------------------------------------------------
def corpus():
    c = [Chunk("cold snare polypectomy for small colorectal polyps", "a", 0),
         Chunk("variceal bleeding band ligation", "a", 1),
         Chunk("oesophageal stent placement for stenosis", "b", 0),
         Chunk("bowel preparation before colonoscopy", "b", 1)]
    vecs = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.7, 0, 0.7]]
    return c, [normalize([float(x) for x in v]) for v in vecs]


class FakeEmbedder:
    def embed_query(self, text):
        return normalize([1.0, 0.0, 0.0])


def test_memory_bm25_finds_keyword_match():
    s = MemoryStore()
    s.add(*corpus())
    assert s.bm25("variceal ligation", 2)[0] == "a#1"
    assert s.bm25("zzzz", 3) == []


def test_memory_vector_orders_by_cosine():
    s = MemoryStore()
    s.add(*corpus())
    assert s.vector(normalize([1.0, 0, 0]), 2) == ["a#0", "b#1"]


def test_retrieve_returns_top5_hybrid_and_deduped():
    s = MemoryStore()
    c, v = corpus()
    s.add(c * 2, v * 2)  # duplicate ids must not duplicate results
    out = retrieve("colorectal polyps snare", s, FakeEmbedder(), top=5)
    assert 1 <= len(out) <= 4 and len({x.text for x in out}) == len(out)
    assert out[0].text.startswith("cold snare")


def test_retrieve_caps_at_top():
    s = MemoryStore()
    chunks = [Chunk(f"polyp text {i}", "s", i) for i in range(40)]
    s.add(chunks, [normalize([1.0, i, 0.0]) for i in range(40)])
    assert len(retrieve("polyp", s, FakeEmbedder(), top=5)) == 5


def test_term_overlap_rerank_prefers_query_coverage():
    a, b = Chunk("snare snare snare", "x", 0), Chunk("cold snare polypectomy", "x", 1)
    assert term_overlap_rerank("cold snare polypectomy", [a, b])[0] is b


def test_empty_store_returns_empty():
    assert retrieve("q", MemoryStore(), FakeEmbedder()) == []


# --- Postgres (Tiger Cloud) store, SQL contract -------------------------------
class FakeCur:
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


class FakeConn:
    def __init__(self, rows=()):
        self.cur = FakeCur(rows)
        self.commits = 0

    def cursor(self):
        return self.cur

    def commit(self):
        self.commits += 1


def sqls(conn):
    return [s for s, _ in conn.cur.log]


def test_pg_setup_creates_extensions_and_indexes():
    conn = FakeConn()
    PgStore(conn, dim=768).setup()
    joined = " | ".join(sqls(conn))
    for needle in ["CREATE EXTENSION IF NOT EXISTS pg_textsearch", "CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE",
                   "vector(768)", "USING bm25", "text_config = 'english'", "USING diskann", "vector_cosine_ops"]:
        assert needle in joined
    assert conn.commits == 1


def test_pg_dim_must_be_int():
    with pytest.raises(ValueError):
        PgStore(FakeConn(), dim="768; DROP TABLE x")


def test_pg_queries_are_parameterized():
    conn = FakeConn(rows=[("a#0",)])
    s = PgStore(conn, dim=3)
    assert s.bm25("x'; DROP TABLE chunks;--", 5) == ["a#0"]
    sql, params = conn.cur.log[-1]
    assert "content <@> to_bm25query(%s, 'chunks_bm25_idx')" in sql  # bare 'col <@> text' fails with bound params (live)
    assert "DROP" not in sql and params[0] == "x'; DROP TABLE chunks;--"
    s.vector([0.1, 0.2, 0.3], 5)
    sql, params = conn.cur.log[-1]
    assert "<=>" in sql and params[0] == "[0.1,0.2,0.3]"


def test_pg_add_upserts_and_commits():
    conn = FakeConn()
    PgStore(conn, dim=2).add([Chunk("t", "s", 0)], [[0.5, 0.5]])
    sql, params = conn.cur.log[-1]
    assert "INSERT INTO chunks" in sql and "ON CONFLICT (id)" in sql
    assert params[:3] == ("s#0", "s", 0) and conn.commits == 1


def test_pg_get_preserves_order():
    conn = FakeConn(rows=[("b#0", "tb", "b", 0), ("a#0", "ta", "a", 0)])
    out = PgStore(conn, dim=2).get(["a#0", "b#0"])
    assert [c.text for c in out] == ["ta", "tb"]


# --- connection from keys.txt -------------------------------------------------
def test_connect_requires_tiger_keys(tmp_path):
    with pytest.raises(MissingKey, match="TIGERCLOUD_SERVICE_URL"):
        connect_tiger(keys(tmp_path, "X=1\n"), connect=lambda dsn: dsn)


def test_connect_injects_password_url_encoded(tmp_path):
    k = keys(tmp_path, "TIGERCLOUD_SERVICE_URL=postgres://tsdbadmin@h.tsdb.cloud.timescale.com:5432/tsdb?sslmode=require\n"
                       "TIGERCLOUD_DB_PASSWORD=p@ss/w:rd\n")
    dsn = connect_tiger(k, connect=lambda dsn: dsn)
    assert dsn.startswith("postgres://tsdbadmin:p%40ss%2Fw%3Ard@h.tsdb.cloud.timescale.com:5432/tsdb")
    assert "sslmode=require" in dsn


def test_connect_missing_password_when_url_has_none(tmp_path):
    k = keys(tmp_path, "TIGERCLOUD_SERVICE_URL=postgres://u@h:5432/tsdb\n")
    with pytest.raises(MissingKey, match="TIGERCLOUD_DB_PASSWORD"):
        connect_tiger(k, connect=lambda dsn: dsn)


def test_pg_custom_table_names_every_object_consistently():
    conn = FakeConn()
    store = PgStore(conn, dim=3, table="smoke_ab12")
    store.setup()
    store.bm25("q", 1)
    joined = " | ".join(sqls(conn))
    assert "CREATE TABLE IF NOT EXISTS smoke_ab12" in joined and "ON smoke_ab12 USING bm25" in joined
    assert "to_bm25query(%s, 'smoke_ab12_bm25_idx')" in sqls(conn)[-1]
    assert "chunks" not in joined


@pytest.mark.parametrize("bad", ["x; DROP TABLE y", "Chunks", "1abc", "a-b", "", "a" * 60])
def test_pg_table_name_is_validated(bad):
    with pytest.raises(ValueError):
        PgStore(FakeConn(), dim=3, table=bad)


def test_pg_drop_removes_only_its_table():
    conn = FakeConn()
    PgStore(conn, dim=3, table="smoke_ab12").drop()
    assert sqls(conn)[-1] == "DROP TABLE IF EXISTS smoke_ab12" and conn.commits == 1
