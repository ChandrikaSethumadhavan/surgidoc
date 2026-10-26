"""FR-11: Redis cache for embeddings, query results and session state. Fails open: a Redis outage is a miss."""
import hashlib
import json
import logging

from surgidoc.ingest import Chunk

log = logging.getLogger(__name__)


def _digest(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


class Cache:
    def __init__(self, client, ttl=86400, session_ttl=3600, prefix="surgidoc"):
        if ttl <= 0 or session_ttl <= 0:
            raise ValueError("ttl must be positive")
        self.client, self.ttl, self.session_ttl, self.prefix, self.errors = client, ttl, session_ttl, prefix, 0

    @classmethod
    def from_keys(cls, keys, connect=None, **kw):
        url = keys.require("REDIS_URL")
        if connect is None:
            import redis

            connect = redis.Redis.from_url
        return cls(connect(url), **kw)

    def _key(self, *parts):  # hashed so text (possibly clinical) never appears in a Redis key
        return f"{self.prefix}:{_digest(*parts)}"

    def _get(self, key):
        try:
            raw = self.client.get(key)
            return None if raw is None else json.loads(raw)
        except Exception as e:  # outage or corrupt value: treat as miss, never log the value
            self.errors += 1
            log.warning("cache read failed: %s", type(e).__name__)
            return None

    def _set(self, key, value, ttl):
        try:
            self.client.set(key, json.dumps(value), ex=ttl)
        except Exception as e:
            self.errors += 1
            log.warning("cache write failed: %s", type(e).__name__)

    def get(self, *parts):
        return self._get(self._key(*parts))

    def set(self, *args, ttl=None):
        *parts, value = args
        self._set(self._key(*parts), value, ttl or self.ttl)

    def memoize(self, namespace, parts, compute):
        key = self._key(namespace, *parts)
        hit = self._get(key)
        if hit is not None:
            return hit
        value = compute()
        self._set(key, value, self.ttl)
        return value

    def load_session(self, session_id):
        return self._get(self._key("session", session_id))

    def save_session(self, session_id, state):
        self._set(self._key("session", session_id), state, self.session_ttl)


class CachedEmbedder:
    """Wraps an embedder; only cache misses reach it, in one batch, order preserved."""

    def __init__(self, inner, cache):
        self.inner, self.cache = inner, cache

    def _key(self, kind, text):
        return ("emb", kind, getattr(self.inner, "model", ""), getattr(self.inner, "dim", 0), text)

    def embed_documents(self, texts):
        found = {t: self.cache.get(*self._key("doc", t)) for t in set(texts)}
        missing = [t for t in dict.fromkeys(texts) if found[t] is None]
        if missing:
            for t, v in zip(missing, self.inner.embed_documents(missing)):
                found[t] = v
                self.cache.set(*self._key("doc", t), v)
        return [found[t] for t in texts]

    def embed_query(self, text):
        hit = self.cache.get(*self._key("query", text))
        if hit is not None:
            return hit
        v = self.inner.embed_query(text)
        self.cache.set(*self._key("query", text), v)
        return v


def cached_retrieve(cache, query, fn, version="v1"):
    """fn() -> list[Chunk]. Bump version when the corpus changes so stale results are not served."""
    rows = cache.memoize("retrieve", (version, query), lambda: [[c.text, c.source, c.index] for c in fn()])
    return [Chunk(*r) for r in rows]
