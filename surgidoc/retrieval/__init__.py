"""FR-4: embed, store in Tiger Cloud (BM25 + vector), RRF-fuse to 30, rerank to 5."""
import json
import math
import re
import urllib.request
from urllib.parse import quote, urlsplit, urlunsplit

from surgidoc.ingest import Chunk

GEMINI = "https://generativelanguage.googleapis.com/v1beta"
_WORD = re.compile(r"\w+")


def chunk_id(c):
    return f"{c.source}#{c.index}"


def normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else list(v)


def rrf(rankings, k=60, top=30):
    scores = {}
    for ranking in rankings:
        for rank, id_ in enumerate(ranking, 1):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top]  # stable: ties keep first-seen order


# --- embeddings ---------------------------------------------------------------
def _post(url, headers, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


EMBED_MODEL = "gemini-embedding-2"  # its vectors are NOT comparable with gemini-embedding-001's: the store records which model made each


class GeminiEmbedder:
    """Google AI Studio embeddings. Key goes in a header, never the URL. Rate limits/backoff live in the gateway (T05).
    gemini-embedding-2 has no task type: the task goes into the text ("title: ... | text: ..." for documents,
    "task: search result | query: ..." for queries). gemini-embedding-001 uses taskType."""

    def __init__(self, api_key, model=EMBED_MODEL, dim=768, post=_post, batch=100):
        self.key, self.model, self.dim, self.post, self.batch = api_key, model, dim, post, batch

    @classmethod
    def from_keys(cls, keys, **kw):
        return cls(keys.require("GOOGLE_AI_STUDIO_API_KEY"), **kw)

    def _request(self, text, task, title=None):
        item = {"model": f"models/{self.model}", "outputDimensionality": self.dim}
        if self.model == "gemini-embedding-001":
            item["taskType"] = task
        elif task == "RETRIEVAL_DOCUMENT":
            text = f"title: {title or 'none'} | text: {text}"
        else:
            text = f"task: search result | query: {text}"
        item["content"] = {"parts": [{"text": text}]}
        return item

    def _embed(self, texts, task, titles=None):
        out = []
        titles = titles or [None] * len(texts)
        for i in range(0, len(texts), self.batch):
            part = texts[i:i + self.batch]
            body = {"requests": [self._request(t, task, ti) for t, ti in zip(part, titles[i:i + self.batch])]}
            resp = self.post(f"{GEMINI}/models/{self.model}:batchEmbedContents",
                             {"Content-Type": "application/json", "x-goog-api-key": self.key}, body)
            vecs = [e["values"] for e in resp.get("embeddings", [])]
            if len(vecs) != len(part) or any(len(v) != self.dim for v in vecs):
                raise ValueError("embedding response count or dimension mismatch")
            out += [normalize(v) for v in vecs]  # 001 only returns unit vectors at the default 3072 dims
        return out

    def embed_documents(self, texts, titles=None):
        return self._embed(texts, "RETRIEVAL_DOCUMENT", titles)

    def embed_query(self, text):
        return self._embed([text], "RETRIEVAL_QUERY")[0]


# --- stores -------------------------------------------------------------------
class MemoryStore:
    """Dev/test store with real BM25 + cosine; same interface as PgStore."""

    def __init__(self):
        self.chunks, self.vecs = {}, {}

    def add(self, chunks, vectors):
        for c, v in zip(chunks, vectors):
            self.chunks[chunk_id(c)], self.vecs[chunk_id(c)] = c, v

    def bm25(self, query, n, k1=1.5, b=0.75):
        docs = {i: _WORD.findall(c.text.lower()) for i, c in self.chunks.items()}
        if not docs:
            return []
        avg = sum(map(len, docs.values())) / len(docs) or 1
        scores = {}
        for t in set(_WORD.findall(query.lower())):
            df = sum(t in d for d in docs.values())
            idf = math.log((len(docs) - df + 0.5) / (df + 0.5) + 1)
            for i, d in docs.items():
                tf = d.count(t)
                if tf:
                    scores[i] = scores.get(i, 0) + idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg))
        return sorted(scores, key=lambda i: -scores[i])[:n]

    def vector(self, vec, n):
        q = normalize(vec)
        sims = {i: sum(a * b for a, b in zip(q, normalize(v))) for i, v in self.vecs.items()}
        return sorted(sims, key=lambda i: -sims[i])[:n]

    def get(self, ids):
        return [self.chunks[i] for i in ids if i in self.chunks]

    def ids_for(self, source):
        return {i for i, c in self.chunks.items() if c.source == source}

    def delete_source(self, source):
        for i in self.ids_for(source):
            del self.chunks[i], self.vecs[i]

    def counts(self, any_model=False):
        out = {}
        for c in self.chunks.values():
            n, pages = out.get(c.source, (0, 0))
            out[c.source] = (n + 1, max(pages, c.page or 0))
        return sorted((src, n, pages) for src, (n, pages) in out.items())


class PgStore:
    """Tiger Cloud: pg_textsearch BM25 (<@>) and pgvectorscale DiskANN (<=>); all values parameterized."""

    def __init__(self, conn, dim=768, table="chunks", embed_model=EMBED_MODEL):
        if not isinstance(dim, int):
            raise ValueError("dim must be an int")
        if not re.fullmatch(r"[a-z_][a-z0-9_]{0,40}", str(table)):  # table name is interpolated into DDL: whitelist it
            raise ValueError("invalid table name")
        self.conn, self.dim, self.table, self.embed_model = conn, dim, table, embed_model

    def _run(self, sql, params=None, fetch=False):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall() if fetch else None

    def setup(self):
        for sql in [
            "CREATE EXTENSION IF NOT EXISTS pg_textsearch",
            "CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE",
            f"CREATE TABLE IF NOT EXISTS {self.table} (id text PRIMARY KEY, source text NOT NULL, idx int NOT NULL, "
            f"content text NOT NULL, embedding vector({self.dim}) NOT NULL)",
            f"ALTER TABLE {self.table} ADD COLUMN IF NOT EXISTS title text, ADD COLUMN IF NOT EXISTS page int, "
            "ADD COLUMN IF NOT EXISTS lang text, ADD COLUMN IF NOT EXISTS embed_model text",
            f"CREATE INDEX IF NOT EXISTS {self.table}_bm25_idx ON {self.table} USING bm25(content) "
            "WITH (text_config = 'english')",
            f"CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx ON {self.table} "
            "USING diskann (embedding vector_cosine_ops)",
        ]:
            self._run(sql)
        self.conn.commit()

    def add(self, chunks, vectors):
        for c, v in zip(chunks, vectors):
            self._run(f"INSERT INTO {self.table} (id, source, idx, content, embedding, title, page, lang, embed_model) "
                      "VALUES (%s, %s, %s, %s, %s::vector, %s, %s, %s, %s) ON CONFLICT (id) "
                      "DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding, "
                      "title = EXCLUDED.title, page = EXCLUDED.page, lang = EXCLUDED.lang, embed_model = EXCLUDED.embed_model",
                      (chunk_id(c), c.source, c.index, c.text, _vec(v), c.title, c.page, c.lang, self.embed_model))
        self.conn.commit()

    def bm25(self, query, n):
        # a bare `content <@> %s` fails with bound params on Tiger Cloud (live test): name the BM25 index explicitly
        rows = self._run(f"SELECT id FROM {self.table} ORDER BY content <@> to_bm25query(%s, '{self.table}_bm25_idx') "
                         "LIMIT %s", (query, n), fetch=True)
        return [r[0] for r in rows]

    def vector(self, vec, n):
        # only vectors made by the current embedding model are comparable with the query vector
        rows = self._run(f"SELECT id FROM {self.table} WHERE embed_model = %s ORDER BY embedding <=> %s::vector LIMIT %s",
                         (self.embed_model, _vec(vec), n), fetch=True)
        return [r[0] for r in rows]

    def get(self, ids):
        if not ids:
            return []
        rows = self._run(f"SELECT id, content, source, idx, page, title, lang FROM {self.table} WHERE id = ANY(%s)",
                         (list(ids),), fetch=True)
        by_id = {r[0]: Chunk(r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def ids_for(self, source):
        """Ids already stored for the source WITH the current embedding model (older-model rows get re-embedded)."""
        return {r[0] for r in self._run(f"SELECT id FROM {self.table} WHERE source = %s AND embed_model = %s",
                                        (source, self.embed_model), fetch=True)}

    def delete_source(self, source):
        self._run(f"DELETE FROM {self.table} WHERE source = %s", (source,))
        self.conn.commit()

    def counts(self, any_model=False):
        """-> [(source, chunks, pages)] for the status command; by default only chunks embedded with the current model."""
        where, params = ("", None) if any_model else ("WHERE embed_model = %s ", (self.embed_model,))
        return [tuple(r) for r in self._run(f"SELECT source, count(*), coalesce(max(page), 0) FROM {self.table} {where}"
                                            "GROUP BY source ORDER BY source", params, fetch=True)]

    def drop(self):
        self._run(f"DROP TABLE IF EXISTS {self.table}")
        self.conn.commit()


def _vec(v):
    return "[" + ",".join(str(x) for x in v) + "]"


def connect_tiger(keys, connect=None):
    """Connect using keys.txt only: TIGERCLOUD_SERVICE_URL (+ TIGERCLOUD_DB_PASSWORD if the URL has none)."""
    parts = urlsplit(keys.require("TIGERCLOUD_SERVICE_URL"))
    if parts.password is None:
        pw = keys.require("TIGERCLOUD_DB_PASSWORD")
        netloc = f"{parts.username}:{quote(pw, safe='')}@{parts.hostname}" + (f":{parts.port}" if parts.port else "")
        parts = parts._replace(netloc=netloc)
    if connect is None:
        import psycopg

        connect = psycopg.connect
    return connect(urlunsplit(parts))


# --- retrieval ----------------------------------------------------------------
def term_overlap_rerank(query, chunks):
    # ponytail: lexical stand-in for the cross-encoder rerank; swap in Cohere/Gemini rerank when a free key is verified
    q = set(_WORD.findall(query.lower()))
    cover = lambda c: len(q & set(_WORD.findall(c.text.lower()))) / (len(q) or 1)
    return sorted(chunks, key=lambda c: -cover(c))


def retrieve(query, store, embedder, rerank=term_overlap_rerank, candidates=30, top=5):
    fused = rrf([store.bm25(query, candidates), store.vector(embedder.embed_query(query), candidates)], top=candidates)
    return rerank(query, store.get([i for i, _ in fused]))[:top]
