"""FR-10: store approved reports (JSON + PDF), review events (hypertable) and daily rollups in Tiger Cloud Postgres.

Every value is a bound parameter; the table prefix is whitelisted because it is interpolated into DDL.
"""
import json
import re
from datetime import date

from surgidoc.approval import content_hash


class StorageError(Exception):
    pass


class ReportStore:
    def __init__(self, conn, prefix="surgidoc"):
        if not re.fullmatch(r"[a-z_][a-z0-9_]{0,30}", str(prefix)):
            raise ValueError("invalid table prefix")
        self.conn, self.prefix = conn, prefix
        self.reports, self.events, self.rollups = f"{prefix}_reports", f"{prefix}_events", f"{prefix}_rollups"

    def _run(self, sql, params=None, fetch=False):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone() if fetch else None

    def setup(self):
        for sql in [
            f"CREATE TABLE IF NOT EXISTS {self.reports} (id text PRIMARY KEY, revision int NOT NULL, "
            "content_hash text NOT NULL, approver_name text NOT NULL, approved_at timestamptz NOT NULL, "
            "statements jsonb NOT NULL, sources jsonb NOT NULL, pdf bytea NOT NULL, created_at timestamptz DEFAULT now())",
            f"CREATE TABLE IF NOT EXISTS {self.events} (report_id text NOT NULL, type text NOT NULL, actor text NOT NULL, "
            "revision int NOT NULL, at timestamptz NOT NULL, detail jsonb NOT NULL DEFAULT '{}')",
            f"SELECT create_hypertable('{self.events}', 'at', if_not_exists => TRUE)",
            f"CREATE INDEX IF NOT EXISTS {self.events}_report_idx ON {self.events} (report_id, at DESC)",
            f"CREATE TABLE IF NOT EXISTS {self.rollups} (day date NOT NULL, metric text NOT NULL, value bigint NOT NULL, "
            "PRIMARY KEY (day, metric))",
        ]:
            self._run(sql)
        self.conn.commit()

    def save_report(self, report, pdf):
        if report.get("status") != "APPROVED":
            raise StorageError("only APPROVED reports can be stored")
        if not isinstance(pdf, bytes) or not pdf.startswith(b"%PDF"):
            raise StorageError("pdf must be PDF bytes")
        if content_hash(report["statements"], report["sources"]) != report.get("content_hash"):
            raise StorageError("report content hash does not match its approval")
        self._run(f"INSERT INTO {self.reports} (id, revision, content_hash, approver_name, approved_at, statements, "
                  "sources, pdf) VALUES (%s, %s, %s, %s, %s::timestamptz, %s::jsonb, %s::jsonb, %s)",
                  (report["report_id"], report["revision"], report["content_hash"], report["approver_name"],
                   report["approved_at"], json.dumps(report["statements"], ensure_ascii=False),
                   json.dumps(report["sources"], ensure_ascii=False), pdf))
        self.conn.commit()

    def record_events(self, report_id, events):
        for e in events:
            self._run(f"INSERT INTO {self.events} (report_id, type, actor, revision, at, detail) "
                      "VALUES (%s, %s, %s, %s, %s::timestamptz, %s::jsonb)",
                      (report_id, e["type"], e["actor"], e["revision"], e["at"],
                       json.dumps(e.get("detail", {}), ensure_ascii=False)))
        self.conn.commit()

    def rollup_day(self, day):
        day = date.fromisoformat(str(day)).isoformat()
        self._run(f"INSERT INTO {self.rollups} (day, metric, value) SELECT %s::date, type, count(*) "
                  f"FROM {self.events} WHERE at >= %s::date AND at < %s::date + 1 GROUP BY type "
                  "ON CONFLICT (day, metric) DO UPDATE SET value = EXCLUDED.value", (day, day, day))
        self.conn.commit()

    def get_pdf(self, report_id):
        row = self._run(f"SELECT pdf FROM {self.reports} WHERE id = %s", (report_id,), fetch=True)
        return bytes(row[0]) if row else None

    def drop(self):
        for t in (self.reports, self.events, self.rollups):
            self._run(f"DROP TABLE IF EXISTS {t}")
        self.conn.commit()
