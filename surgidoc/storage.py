"""FR-10, FR-13: store approved reports (JSON + PDF), review events (hypertable) and daily rollups in Tiger Cloud Postgres.

With a Cipher, statements, sources and the PDF are stored as AES-GCM envelopes bound to the report id, and event
details are encrypted too (their text, e.g. a rejection reason, is clinician free text). The approver name, content hash,
event type/actor/time stay in the clear so records remain queryable and auditable. Every value is a bound parameter; the
table prefix is whitelisted because it is interpolated into DDL.
"""
import base64
import json
import re
from datetime import date

from surgidoc.approval import content_hash
from surgidoc.security import Cipher, DecryptionError


class StorageError(Exception):
    pass


def _json(obj):
    return json.dumps(obj, ensure_ascii=False).encode()


class ReportStore:
    def __init__(self, conn, prefix="surgidoc", cipher=None):
        if not re.fullmatch(r"[a-z_][a-z0-9_]{0,30}", str(prefix)):
            raise ValueError("invalid table prefix")
        if cipher is not None and not isinstance(cipher, Cipher):
            raise TypeError("cipher must be a surgidoc.security.Cipher")
        self.conn, self.prefix, self.cipher = conn, prefix, cipher
        self.reports, self.events, self.rollups = f"{prefix}_reports", f"{prefix}_events", f"{prefix}_rollups"

    def _run(self, sql, params=None, fetch=None):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone() if fetch == "one" else cur.fetchall() if fetch == "all" else None

    def setup(self):
        for sql in [
            f"CREATE TABLE IF NOT EXISTS {self.reports} (id text PRIMARY KEY, revision int NOT NULL, "
            "content_hash text NOT NULL, approver_name text NOT NULL, approved_at timestamptz NOT NULL, "
            "statements bytea NOT NULL, sources bytea NOT NULL, pdf bytea NOT NULL, encrypted boolean NOT NULL, "
            "created_at timestamptz DEFAULT now())",
            f"CREATE TABLE IF NOT EXISTS {self.events} (report_id text NOT NULL, type text NOT NULL, actor text NOT NULL, "
            "revision int NOT NULL, at timestamptz NOT NULL, detail jsonb NOT NULL DEFAULT '{}')",
            f"SELECT create_hypertable('{self.events}', 'at', if_not_exists => TRUE)",
            f"CREATE INDEX IF NOT EXISTS {self.events}_report_idx ON {self.events} (report_id, at DESC)",
            f"CREATE TABLE IF NOT EXISTS {self.rollups} (day date NOT NULL, metric text NOT NULL, value bigint NOT NULL, "
            "PRIMARY KEY (day, metric))",
        ]:
            self._run(sql)
        self.conn.commit()

    # --- sealing -----------------------------------------------------------------------------------
    def _seal(self, data, aad):
        return self.cipher.encrypt(data, aad) if self.cipher else data

    def _open(self, blob, aad, encrypted):
        if not encrypted:
            return bytes(blob)
        if self.cipher is None:
            raise StorageError("report is encrypted but no cipher is configured")
        try:
            return self.cipher.decrypt(bytes(blob), aad)
        except DecryptionError:
            raise StorageError("cannot decrypt report data (wrong key, wrong record, or modified)") from None

    # --- reports -----------------------------------------------------------------------------------
    def save_report(self, report, pdf):
        if report.get("status") != "APPROVED":
            raise StorageError("only APPROVED reports can be stored")
        if not isinstance(pdf, bytes) or not pdf.startswith(b"%PDF"):
            raise StorageError("pdf must be PDF bytes")
        if content_hash(report["statements"], report["sources"]) != report.get("content_hash"):
            raise StorageError("report content hash does not match its approval")
        rid = report["report_id"]
        self._run(f"INSERT INTO {self.reports} (id, revision, content_hash, approver_name, approved_at, statements, "
                  "sources, pdf, encrypted) VALUES (%s, %s, %s, %s, %s::timestamptz, %s, %s, %s, %s)",
                  (rid, report["revision"], report["content_hash"], report["approver_name"], report["approved_at"],
                   self._seal(_json(report["statements"]), f"{rid}:statements"),
                   self._seal(_json(report["sources"]), f"{rid}:sources"), self._seal(pdf, f"{rid}:pdf"),
                   self.cipher is not None))
        self.conn.commit()

    def get_report(self, report_id):
        row = self._run(f"SELECT id, revision, content_hash, approver_name, approved_at, statements, sources, encrypted "
                        f"FROM {self.reports} WHERE id = %s", (report_id,), fetch="one")
        if not row:
            return None
        rid, revision, digest, name, at, st, so, enc = row
        statements = json.loads(self._open(st, f"{rid}:statements", enc))
        sources = json.loads(self._open(so, f"{rid}:sources", enc))
        if content_hash(statements, sources) != digest:
            raise StorageError("stored report content does not match its content hash")
        return {"status": "APPROVED", "report_id": rid, "revision": revision, "content_hash": digest,
                "approver_name": name, "approved_at": at.isoformat() if hasattr(at, "isoformat") else at,
                "statements": statements, "sources": sources}

    def get_pdf(self, report_id):
        row = self._run(f"SELECT pdf, encrypted FROM {self.reports} WHERE id = %s", (report_id,), fetch="one")
        return self._open(row[0], f"{report_id}:pdf", row[1]) if row else None

    # --- events ------------------------------------------------------------------------------------
    @staticmethod
    def _event_aad(report_id, type_, revision):
        return f"{report_id}:event:{type_}:{revision}"

    def record_events(self, report_id, events):
        for e in events:
            detail = e.get("detail", {})
            if self.cipher:
                blob = self.cipher.encrypt_json(detail, self._event_aad(report_id, e["type"], e["revision"]))
                detail = {"encrypted": base64.urlsafe_b64encode(blob).decode()}
            self._run(f"INSERT INTO {self.events} (report_id, type, actor, revision, at, detail) "
                      "VALUES (%s, %s, %s, %s, %s::timestamptz, %s::jsonb)",
                      (report_id, e["type"], e["actor"], e["revision"], e["at"], json.dumps(detail, ensure_ascii=False)))
        self.conn.commit()

    def get_events(self, report_id):
        rows = self._run(f"SELECT type, actor, revision, at, detail FROM {self.events} WHERE report_id = %s ORDER BY at",
                         (report_id,), fetch="all") or []
        out = []
        for type_, actor, revision, at, detail in rows:
            detail = detail if isinstance(detail, dict) else json.loads(detail)
            if "encrypted" in detail:
                detail = json.loads(self._open(base64.urlsafe_b64decode(detail["encrypted"]),
                                               self._event_aad(report_id, type_, revision), True))
            out.append({"type": type_, "actor": actor, "revision": revision,
                        "at": at.isoformat() if hasattr(at, "isoformat") else at, "detail": detail})
        return out

    # --- housekeeping --------------------------------------------------------------------------------
    def rollup_day(self, day):
        day = date.fromisoformat(str(day)).isoformat()
        self._run(f"INSERT INTO {self.rollups} (day, metric, value) SELECT %s::date, type, count(*) "
                  f"FROM {self.events} WHERE at >= %s::date AND at < %s::date + 1 GROUP BY type "
                  "ON CONFLICT (day, metric) DO UPDATE SET value = EXCLUDED.value", (day, day, day))
        self.conn.commit()

    def drop(self):
        for t in (self.reports, self.events, self.rollups):
            self._run(f"DROP TABLE IF EXISTS {t}")
        self.conn.commit()
