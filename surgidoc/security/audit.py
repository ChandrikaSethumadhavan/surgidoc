"""NFR-6: append-only, hash-chained audit log.

Each entry carries the sha256 of the previous entry, so edits, deletions and reordering are detectable. Cutting off the
tail is only detectable against an externally anchored head (verify(expected_head=...)). The Postgres sink also makes
the table reject UPDATE/DELETE/TRUNCATE; that guards against application bugs, not against the database owner.
"""
import copy
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

GENESIS = "0" * 64
_SECRET = re.compile(r"pass|secret|token|key|auth|credential", re.I)


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    index: int | None = None
    reason: str = ""


def entry_hash(entry):
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def verify_chain(entries, expected_head=None):
    prev = GENESIS
    for i, e in enumerate(entries):
        if e["seq"] != i + 1:
            return VerifyResult(False, i, "sequence gap or reorder")
        if e["prev_hash"] != prev:
            return VerifyResult(False, i, "broken link to previous entry")
        if e["hash"] != entry_hash(e):
            return VerifyResult(False, i, "entry content changed")
        prev = e["hash"]
    head = (len(entries), prev)
    if expected_head is not None and tuple(expected_head) != head:
        return VerifyResult(False, len(entries), "head does not match the anchored head (entries removed?)")
    return VerifyResult(True)


def _text(value, limit=200):
    return "".join(c if c.isprintable() else "?" for c in str(value))[:limit]


def _clean(value):
    if isinstance(value, dict):
        return {str(k): "***" if _SECRET.search(str(k)) else _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:500] + "…"
    return value if value is None or isinstance(value, (bool, int, float)) else _text(value, 500)


class MemorySink:
    def __init__(self):
        self.entries, self._lock = [], threading.Lock()

    def append(self, make):
        with self._lock:
            last = self.entries[-1] if self.entries else None
            e = make(last["seq"] if last else 0, last["hash"] if last else GENESIS)
            self.entries.append(e)
            return e

    def read_all(self):
        with self._lock:
            return copy.deepcopy(self.entries)


class PgSink:
    """Tiger Cloud Postgres. An advisory lock serialises writers so the chain cannot fork."""

    def __init__(self, conn, prefix="surgidoc"):
        import re as _re

        if not _re.fullmatch(r"[a-z_][a-z0-9_]{0,30}", str(prefix)):
            raise ValueError("invalid table prefix")
        self.conn, self.table = conn, f"{prefix}_audit"

    def setup(self):
        t = self.table
        with self.conn.cursor() as cur:
            cur.execute(f"CREATE TABLE IF NOT EXISTS {t} (seq bigint PRIMARY KEY, at text NOT NULL, actor text NOT NULL, "
                        "action text NOT NULL, resource text NOT NULL, outcome text NOT NULL, detail text NOT NULL, "
                        "prev_hash text NOT NULL, hash text NOT NULL)")
            cur.execute(f"CREATE OR REPLACE FUNCTION {t}_immutable() RETURNS trigger LANGUAGE plpgsql AS $$ "
                        "BEGIN RAISE EXCEPTION 'audit log is append-only'; END $$")
            cur.execute(f"CREATE OR REPLACE TRIGGER {t}_no_change BEFORE UPDATE OR DELETE ON {t} "
                        f"FOR EACH ROW EXECUTE FUNCTION {t}_immutable()")
            cur.execute(f"CREATE OR REPLACE TRIGGER {t}_no_truncate BEFORE TRUNCATE ON {t} "
                        f"FOR EACH STATEMENT EXECUTE FUNCTION {t}_immutable()")
        self.conn.commit()

    def append(self, make):
        t = self.table
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (t,))
                cur.execute(f"SELECT seq, hash FROM {t} ORDER BY seq DESC LIMIT 1")
                row = cur.fetchone()
                e = make(*(row if row else (0, GENESIS)))
                cur.execute(f"INSERT INTO {t} (seq, at, actor, action, resource, outcome, detail, prev_hash, hash) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                            (e["seq"], e["at"], e["actor"], e["action"], e["resource"], e["outcome"],
                             json.dumps(e["detail"], sort_keys=True, ensure_ascii=False), e["prev_hash"], e["hash"]))
            self.conn.commit()
            return e
        except Exception:
            self.conn.rollback()
            raise

    def read_all(self):
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT seq, at, actor, action, resource, outcome, detail, prev_hash, hash FROM {self.table} "
                        "ORDER BY seq")
            rows = cur.fetchall()
        keys = ("seq", "at", "actor", "action", "resource", "outcome", "detail", "prev_hash", "hash")
        return [{**dict(zip(keys, r)), "detail": json.loads(r[6])} for r in rows]

    def drop(self):
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table}")
            cur.execute(f"DROP FUNCTION IF EXISTS {self.table}_immutable()")
        self.conn.commit()


class AuditLog:
    def __init__(self, sink, clock=lambda: datetime.now(timezone.utc)):
        self.sink, self.clock = sink, clock

    def log(self, actor, action, resource="", outcome="ok", detail=None):
        actor, action = _text(actor).strip(), _text(action).strip()
        if not actor or not action:
            raise ValueError("actor and action are required")
        resource, outcome, detail = _text(resource), _text(outcome, 20), _clean(detail or {})

        def make(prev_seq, prev_hash):
            e = {"seq": prev_seq + 1, "at": self.clock().isoformat(), "actor": actor, "action": action,
                 "resource": resource, "outcome": outcome, "detail": detail, "prev_hash": prev_hash}
            e["hash"] = entry_hash(e)
            return e

        return self.sink.append(make)

    def entries(self):
        return self.sink.read_all()

    def head(self):
        es = self.entries()
        return (es[-1]["seq"], es[-1]["hash"]) if es else (0, GENESIS)

    def verify(self, expected_head=None):
        return verify_chain(self.entries(), expected_head)
