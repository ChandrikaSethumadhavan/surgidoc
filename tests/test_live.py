"""Opt-in live smoke tests against the real services using keys.txt. Run: SURGIDOC_LIVE=1 pytest tests/test_live.py
Skipped by default so the normal suite stays offline. Sends only a trivial public prompt; never prints secrets."""
import os
from pathlib import Path

import pytest

from surgidoc.config import load_keys

KEYS = Path(__file__).resolve().parent.parent / "keys.txt"
pytestmark = pytest.mark.skipif(not os.environ.get("SURGIDOC_LIVE"), reason="set SURGIDOC_LIVE=1 to run live checks")


@pytest.fixture(scope="module")
def keys():
    return load_keys(KEYS)


def models():
    from surgidoc.gateway import ROUTES

    return sorted({m for ms in ROUTES.values() for m in ms})


@pytest.mark.parametrize("model", models())
def test_every_default_route_model_is_accepted_by_the_api(keys, model):
    from surgidoc.gateway import Gateway

    g = Gateway.from_keys(keys, routes={"t": [model]}, default_limits={"rpm": 5, "rpd": 50})
    assert g.generate("t", "Reply with the single word: ok").text.strip()


def test_embedding_has_requested_dimension(keys):
    from surgidoc.retrieval import GeminiEmbedder

    assert len(GeminiEmbedder.from_keys(keys, dim=768).embed_query("cold snare polypectomy")) == 768


def test_redis_roundtrip(keys):
    from surgidoc.cache import Cache

    c = Cache.from_keys(keys, ttl=60)
    c.set("live-smoke", {"ok": True})
    assert c.get("live-smoke") == {"ok": True}


def test_tiger_cloud_connects_with_vector_extension_available(keys):
    from surgidoc.retrieval import connect_tiger

    conn = connect_tiger(keys)
    try:
        with conn.cursor() as cur:
            cur.execute("select name from pg_available_extensions where name in ('vector','vectorscale','pg_textsearch')")
            assert {r[0] for r in cur.fetchall()} >= {"vector", "vectorscale"}
    finally:
        conn.close()


def test_hybrid_retrieval_end_to_end_on_a_throwaway_table(keys):
    import uuid

    from surgidoc.ingest import Chunk
    from surgidoc.retrieval import GeminiEmbedder, PgStore, connect_tiger, retrieve

    texts = ["Cold snare polypectomy is recommended for colorectal polyps smaller than 10 mm.",
             "Band ligation is the preferred endoscopic treatment for acute oesophageal variceal bleeding.",
             "Self-expanding metal stents are used for malignant oesophageal stenosis."]
    chunks = [Chunk(t, "esge/test.xml", i) for i, t in enumerate(texts)]
    emb = GeminiEmbedder.from_keys(keys, dim=768)
    conn = connect_tiger(keys)
    store = PgStore(conn, dim=768, table="smoke_" + uuid.uuid4().hex[:8])
    try:
        store.setup()
        store.add(chunks, emb.embed_documents(texts))
        assert store.bm25("variceal bleeding band ligation", 3)[0] == "esge/test.xml#1"
        assert store.vector(emb.embed_query("removing small colon polyps"), 3)[0] == "esge/test.xml#0"
        top = retrieve("how are bleeding oesophageal varices treated", store, emb, top=2)
        assert top[0].text == texts[1] and len(top) == 2
    finally:
        store.drop()
        conn.close()
