import pytest

from surgidoc.cache import Cache, CachedEmbedder, cached_retrieve
from surgidoc.config import MissingKey, load_keys
from surgidoc.ingest import Chunk


class FakeRedis:
    def __init__(self):
        self.data, self.ttls, self.down = {}, {}, False

    def get(self, k):
        if self.down:
            raise ConnectionError("redis down")
        return self.data.get(k)

    def set(self, k, v, ex=None):
        if self.down:
            raise ConnectionError("redis down")
        self.data[k] = v
        self.ttls[k] = ex


class CountingEmbedder:
    dim, model = 3, "m"

    def __init__(self):
        self.doc_calls, self.query_calls = [], []

    def embed_documents(self, texts):
        self.doc_calls.append(list(texts))
        return [[float(len(t)), 0.0, 1.0] for t in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return [float(len(text)), 1.0, 0.0]


def test_json_roundtrip_with_ttl():
    r = FakeRedis()
    c = Cache(r, ttl=60)
    c.set("k", {"a": [1, 2]})
    assert c.get("k") == {"a": [1, 2]}
    assert list(r.ttls.values()) == [60]


def test_miss_returns_none_and_keys_are_namespaced_and_hashed():
    r = FakeRedis()
    c = Cache(r)
    assert c.get("nope") is None
    c.set("patient colonoscopy text", 1)
    (key,) = r.data
    assert key.startswith("surgidoc:") and "patient" not in key


def test_corrupt_value_is_a_miss():
    r = FakeRedis()
    c = Cache(r)
    c.set("k", 1)
    r.data[next(iter(r.data))] = b"{not json"
    assert c.get("k") is None


def test_redis_outage_fails_open_and_is_counted():
    r = FakeRedis()
    r.down = True
    c = Cache(r)
    assert c.get("k") is None
    c.set("k", 1)  # must not raise
    assert c.errors == 2


def test_memoize_serves_repeat_from_cache():  # AC-5
    c = Cache(FakeRedis())
    calls = []

    def compute():
        calls.append(1)
        return {"answer": 42}

    assert c.memoize("q", ("what is x", 5), compute) == {"answer": 42}
    assert c.memoize("q", ("what is x", 5), compute) == {"answer": 42}
    assert len(calls) == 1
    c.memoize("q", ("what is y", 5), compute)
    assert len(calls) == 2


def test_memoize_computes_when_redis_down():
    r = FakeRedis()
    r.down = True
    assert Cache(r).memoize("q", ("x",), lambda: 7) == 7


def test_session_state_save_load():
    c = Cache(FakeRedis(), session_ttl=3600)
    assert c.load_session("s1") is None
    c.save_session("s1", {"step": "judge"})
    assert c.load_session("s1") == {"step": "judge"}
    assert c.load_session("s2") is None


def test_session_uses_its_own_ttl():
    r = FakeRedis()
    c = Cache(r, ttl=10, session_ttl=99)
    c.save_session("s", {})
    assert list(r.ttls.values()) == [99]


def test_ttl_must_be_positive():
    with pytest.raises(ValueError):
        Cache(FakeRedis(), ttl=0)


def test_cached_embedder_hits_misses_and_keeps_order():
    inner, cache = CountingEmbedder(), Cache(FakeRedis())
    e = CachedEmbedder(inner, cache)
    first = e.embed_documents(["aa", "b"])
    second = e.embed_documents(["b", "ccc", "aa"])
    assert inner.doc_calls == [["aa", "b"], ["ccc"]]
    assert second == [first[1], [3.0, 0.0, 1.0], first[0]]


def test_cached_embedder_query_and_doc_do_not_collide():
    inner, cache = CountingEmbedder(), Cache(FakeRedis())
    e = CachedEmbedder(inner, cache)
    e.embed_documents(["same"])
    e.embed_query("same")
    assert inner.query_calls == ["same"]  # doc cache entry not reused for a query
    e.embed_query("same")
    assert inner.query_calls == ["same"]


def test_cached_embedder_key_depends_on_model():
    cache = Cache(FakeRedis())
    a, b = CountingEmbedder(), CountingEmbedder()
    b.model = "other"
    CachedEmbedder(a, cache).embed_documents(["t"])
    CachedEmbedder(b, cache).embed_documents(["t"])
    assert len(b.doc_calls) == 1


def test_cached_retrieve_roundtrips_chunks():
    cache = Cache(FakeRedis())
    calls = []

    def fn():
        calls.append(1)
        return [Chunk("t", "s", 0)]

    assert cached_retrieve(cache, "q", fn) == [Chunk("t", "s", 0)]
    assert cached_retrieve(cache, "q", fn) == [Chunk("t", "s", 0)]
    assert len(calls) == 1


def test_from_keys_requires_redis_url(tmp_path):
    p = tmp_path / "keys.txt"
    p.write_text("X=1\n")
    with pytest.raises(MissingKey, match="REDIS_URL"):
        Cache.from_keys(load_keys(p))


def test_from_keys_passes_url_to_connect(tmp_path):
    p = tmp_path / "keys.txt"
    p.write_text("REDIS_URL=rediss://default:pw@h:6379\n")
    seen = []
    Cache.from_keys(load_keys(p), connect=lambda url: seen.append(url) or FakeRedis())
    assert seen == ["rediss://default:pw@h:6379"]
