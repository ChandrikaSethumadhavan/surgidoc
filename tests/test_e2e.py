"""AC-1, AC-3, AC-7, AC-9, NFR-1: the whole pipeline with real logic and fake network edges (sockets are blocked)."""
import io
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pypdf import PdfReader

from surgidoc.agents import Statement
from surgidoc.approval import NotApproved, ReviewError
from surgidoc.cache import Cache
from surgidoc.config import load_keys
from surgidoc.gateway import GatewayBlocked, Result
from surgidoc.ingest import Chunk
from surgidoc.observability import Telemetry
from surgidoc.pipeline import Pipeline, build_pipeline
from surgidoc.security import AuditLog, Authorizer, Cipher, MemorySink, PermissionDenied, generate_key
from surgidoc.storage import ReportStore
from surgidoc.vision import Finding, Frame, VideoError

USERS = {"Chandrika": {"reviewer", "admin", "operator"}, "Olive": {"operator"}, "Rex": {"reviewer"}}
TEXTS = ["Small sessile polyp, about 6 mm, in the sigmoid colon.", "No active bleeding."]
GOOD = json.dumps({"statements": [{"text": "A small sessile polyp, about 6 mm, was seen in the sigmoid colon.", "sources": ["F0"]},
                                  {"text": "No active bleeding was observed.", "sources": ["F1"]}]})
APPROVE = json.dumps({"approved": True, "issues": [], "feedback": ""})
REJECT = json.dumps({"approved": False, "issues": [{"statement": 0, "problem": "x"}], "feedback": "unsupported"})


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted in an offline test")

    monkeypatch.setattr(socket.socket, "connect", refuse)


class Clock:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return datetime(2026, 10, 26, 12, 0, self.n % 60, tzinfo=timezone.utc)


class FakeGateway:
    def __init__(self, generator=(), judge=()):
        self.q, self.calls = {"generator": list(generator), "judge": list(judge)}, []

    def generate(self, role, prompt, **kw):
        self.calls.append(role)
        r = self.q[role].pop(0)
        if isinstance(r, Exception):
            raise r
        return Result(r, "m-" + role, 100, 20, 1)


class FakeRedis:
    def __init__(self):
        self.d = {}

    def get(self, k):
        return self.d.get(k)

    def set(self, k, v, ex=None):
        self.d[k] = v


class FakeCur:
    def __init__(self, rows=()):
        self.log, self.rows = [], list(rows)

    def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class FakeConn:
    def __init__(self):
        self.cur, self.commits = FakeCur(), 0

    def cursor(self):
        return self.cur

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


class Env:
    pass


def make(generator=(GOOD,), judge=(APPROVE,), texts=TEXTS, extract=None, cache=True, max_attempts=3, cipher=None):
    e = Env()
    e.spans, e.reader = InMemorySpanExporter(), InMemoryMetricReader()
    e.telemetry = Telemetry(span_exporter=e.spans, metric_reader=e.reader)
    e.audit = AuditLog(MemorySink(), clock=Clock())
    e.authz = Authorizer(USERS, audit=e.audit)
    e.conn = FakeConn()
    e.store = ReportStore(e.conn, prefix="e2e", cipher=cipher)
    e.gateway = FakeGateway(generator, judge)
    e.extracted, e.retrieved = [], []

    def extract_frames(video):
        e.extracted.append(video)
        return [Frame(i, i * 2.0, b"\xff\xd8x") for i in range(len(texts))]

    class Describer:
        def describe(self, frames):
            return [Finding(f.index, f.t, texts[f.index], "gemini-vision") for f in frames]

    def retrieve(q):
        e.retrieved.append(q)
        return [Chunk("Cold snare polypectomy is recommended for polyps under 10 mm.", "esge/poly.xml", 3)]

    e.cache = Cache(FakeRedis()) if cache else None
    e.pipe = Pipeline(extract_frames=extract or extract_frames, describer=Describer(), retrieve=retrieve, gateway=e.gateway,
                      store=e.store, audit=e.audit, authz=e.authz, telemetry=e.telemetry, cache=e.cache,
                      max_attempts=max_attempts, new_id=lambda: "rep-0001")
    return e


def inserted(e, table):
    return [p for s, p in e.conn.cur.log if s.startswith(f"INSERT INTO e2e_{table}")]


def pdf_text(data):
    return " ".join(p.extract_text() for p in PdfReader(io.BytesIO(data)).pages).replace("\n", " ")


VIDEO = "/data/case1.mp4"


# =============================== AC-1: the happy path ===========================================
def test_video_to_approved_pdf_end_to_end():
    e = make()
    run = e.pipe.run(VIDEO, "Olive")  # an operator runs the pipeline
    assert run.status == "pending_review" and run.report_id == "rep-0001" and run.frames == 2
    assert [s.sources for s in run.report.statements] == [["F0"], ["F1"]] and run.report.draft is True
    assert inserted(e, "reports") == []  # nothing stored before a human approves
    pub = e.pipe.approve(run.review, "Rex")  # a reviewer approves
    assert pub.pdf.startswith(b"%PDF") and pub.report_id == "rep-0001"
    text = pdf_text(pub.pdf)
    assert "APPROVED" in text and "Rex" in text and "sigmoid colon" in text
    (row,) = inserted(e, "reports")
    assert row[0] == "rep-0001" and "Rex" in row and pub.pdf in row
    assert len(inserted(e, "events")) == len(run.review.item.events) == 2  # created, approved


def test_audit_trail_covers_run_approval_and_publish_and_verifies():
    e = make()
    run = e.pipe.run(VIDEO, "Olive")
    e.pipe.approve(run.review, "Rex")
    log = e.audit.entries()
    assert [(x["actor"], x["action"], x["outcome"]) for x in log] == [
        ("Olive", "pipeline:run", "ok"), ("Rex", "report:approve", "ok"), ("Rex", "report:publish", "ok")]
    assert log[0]["resource"] == "case1.mp4" and "/data" not in json.dumps(log)  # file name only, never the path
    assert e.audit.verify().ok


def test_telemetry_covers_every_step_and_quality():
    e = make()
    run = e.pipe.run(VIDEO, "Olive")
    e.pipe.approve(run.review, "Rex")
    snap = e.telemetry.snapshot()
    for op in ["pipeline.run", "pipeline.frames", "pipeline.describe", "pipeline.retrieve", "pipeline.agents",
               "pipeline.publish", "gateway.generate"]:
        assert op in snap["operations"], op
    assert snap["usage"]["llm_requests"] == {"generator/m-generator": 1, "judge/m-judge": 1}
    q = snap["quality"]
    assert q["reports"] == {"approved": 1} and q["guardrail_block_rate"] == 0 and q["human_approval_rate"] == 1.0


def test_retrieval_is_cached_across_runs():
    e = make(generator=(GOOD, GOOD), judge=(APPROVE, APPROVE))
    e.pipe.run(VIDEO, "Olive")
    e.pipe.run(VIDEO, "Olive")
    assert len(e.retrieved) == 1
    assert e.telemetry.snapshot()["quality"]["cache_hit_rate"] == 0.5


# =============================== AC-3: no approval, no PDF ========================================
def test_publish_before_approval_is_refused_and_stores_nothing():
    e = make()
    run = e.pipe.run(VIDEO, "Olive")
    with pytest.raises(NotApproved):
        e.pipe.publish(run.review, "Rex")
    assert inserted(e, "reports") == []


def test_operators_cannot_approve_or_publish_and_denials_are_audited():  # AC-7
    e = make()
    run = e.pipe.run(VIDEO, "Olive")
    with pytest.raises(PermissionDenied):
        e.pipe.approve(run.review, "Olive")
    with pytest.raises(PermissionDenied):
        e.pipe.publish(run.review, "Olive")
    assert run.review.item.state == "pending_review" and inserted(e, "reports") == []
    assert [x["outcome"] for x in e.audit.entries()].count("denied") == 2 and e.audit.verify().ok


def test_strangers_cannot_run_the_pipeline_and_nothing_executes():
    e = make()
    with pytest.raises(PermissionDenied):
        e.pipe.run(VIDEO, "mallory")
    assert e.extracted == [] and e.gateway.calls == []
    assert e.audit.entries()[-1]["outcome"] == "denied"


# =============================== review flows ======================================================
def test_edit_goes_through_guardrails_then_approval_uses_the_edit():
    e = make()
    run = e.pipe.run(VIDEO, "Olive")
    with pytest.raises(ReviewError):
        run.review.edit("Rex", [Statement("The patient has colorectal cancer.", ["F0"])])
    run.review.edit("Rex", [Statement("A 6 mm sessile polyp was seen in the sigmoid colon.", ["F0"])])
    pub = e.pipe.approve(run.review, "Rex")
    text = pdf_text(pub.pdf)
    assert "A 6 mm sessile polyp was seen" in text and "No active bleeding" not in text  # edited-out finding is gone everywhere
    assert run.review.item.revision == 2


def test_reject_stores_events_but_no_report():
    e = make()
    run = e.pipe.run(VIDEO, "Olive")
    e.pipe.reject(run.review, "Rex", "Not accurate.")
    assert run.review.item.state == "rejected" and inserted(e, "reports") == []
    assert [p[1] for p in inserted(e, "events")] == ["created", "rejected"]
    assert e.telemetry.snapshot()["quality"]["human_approval_rate"] == 0.0


# =============================== things that must never reach a human ===============================
def test_guardrail_block_stops_before_human_review():
    bad = json.dumps({"statements": [{"text": "The patient has colorectal cancer.", "sources": ["F0"]}]})
    e = make(generator=(bad,), judge=(APPROVE,))
    run = e.pipe.run(VIDEO, "Olive")
    assert run.status == "blocked_by_guardrails" and run.review is None and "diagnosis" in run.violations
    assert e.telemetry.snapshot()["quality"]["guardrail_violations"] == {"diagnosis": 1}
    assert "colorectal cancer" not in json.dumps(e.audit.entries())  # rules only, never the text


def test_judge_rejection_after_all_attempts_is_not_reviewable():
    e = make(generator=(GOOD, GOOD), judge=(REJECT, REJECT), max_attempts=2)
    run = e.pipe.run(VIDEO, "Olive")
    assert run.status == "rejected" and run.review is None and run.report.attempts == 2
    assert inserted(e, "reports") == [] and e.audit.entries()[-1]["detail"]["status"] == "rejected"


def test_gemini_safety_block_is_reported_as_blocked():
    e = make(generator=(GatewayBlocked("b"),), judge=())
    run = e.pipe.run(VIDEO, "Olive")
    assert run.status == "blocked" and run.review is None


def test_unusable_frames_never_reach_the_agents():
    e = make(texts=["unclear", "Unclear."])
    run = e.pipe.run(VIDEO, "Olive")
    assert run.status == "no_usable_findings" and e.gateway.calls == [] and run.report is None


def test_unclear_frames_are_dropped_but_usable_ones_continue():
    e = make(texts=["unclear", "A 5 mm polyp in the rectum."], generator=(
        json.dumps({"statements": [{"text": "A 5 mm polyp was seen in the rectum.", "sources": ["F0"]}]}),), judge=(APPROVE,))
    run = e.pipe.run(VIDEO, "Olive")
    assert run.status == "pending_review" and "A 5 mm polyp" in run.review.view("Rex")["sources"]["F0"]["text"]


# =============================== failures ============================================================
def test_step_failure_is_audited_by_type_only_and_reraised():
    def boom(video):
        raise VideoError("cannot read /secret/path/video.mp4 password=hunter2")

    e = make(extract=boom)
    with pytest.raises(VideoError):
        e.pipe.run(VIDEO, "Olive")
    last = e.audit.entries()[-1]
    assert (last["outcome"], last["detail"]) == ("error", {"error": "VideoError"})
    assert "hunter2" not in json.dumps(e.audit.entries()) and e.audit.verify().ok
    assert e.telemetry.snapshot()["operations"]["pipeline.run"]["errors"] == 1


def test_storage_failure_after_approval_can_be_retried_with_publish():
    e = make()
    run = e.pipe.run(VIDEO, "Olive")
    real = e.store.save_report
    e.store.save_report = lambda *a: (_ for _ in ()).throw(RuntimeError("db down"))
    with pytest.raises(RuntimeError):
        e.pipe.approve(run.review, "Rex")
    assert run.review.item.state == "approved"  # the human decision is kept
    e.store.save_report = real
    assert e.pipe.publish(run.review, "Rex").pdf.startswith(b"%PDF")


# =============================== encryption at rest ====================================================
def test_with_a_cipher_the_stored_report_pdf_and_events_are_ciphertext():
    e = make(cipher=Cipher(generate_key()))
    run = e.pipe.run(VIDEO, "Olive")
    pub = e.pipe.approve(run.review, "Rex")
    (row,) = inserted(e, "reports")
    statements, sources, stored_pdf, flag = row[5], row[6], row[7], row[8]
    assert flag is True and all(b.startswith(b"SD1") for b in (statements, sources, stored_pdf))
    assert b"sigmoid" not in statements and not stored_pdf.startswith(b"%PDF") and pub.pdf not in row
    assert pub.pdf.startswith(b"%PDF")  # the caller still gets a normal PDF
    details = [p[5] for p in inserted(e, "events")]
    assert details and all('"encrypted"' in d and "content_hash" not in d for d in details)


def test_build_pipeline_requires_the_encryption_key(tmp_path):
    p = tmp_path / "keys.txt"
    p.write_text("GOOGLE_AI_STUDIO_API_KEY=k\nTIGERCLOUD_SERVICE_URL=postgres://u@h:5432/tsdb?sslmode=require\n"
                 "TIGERCLOUD_DB_PASSWORD=pw\nREDIS_URL=redis://h:6379\nSURGIDOC_USERS=Rex:reviewer\n")
    with pytest.raises(Exception, match="SURGIDOC_ENCRYPTION_KEY"):
        build_pipeline(load_keys(p), pg_connect=lambda d: FakeConn(), redis_connect=lambda u: FakeRedis())


# =============================== wiring from keys.txt ================================================
def test_build_pipeline_wires_real_components_from_keys(tmp_path):
    p = tmp_path / "keys.txt"
    p.write_text("GOOGLE_AI_STUDIO_API_KEY=k\nTIGERCLOUD_SERVICE_URL=postgres://u@h:5432/tsdb?sslmode=require\n"
                 f"TIGERCLOUD_DB_PASSWORD=pw\nREDIS_URL=redis://h:6379\nSURGIDOC_USERS=Rex:reviewer\nSURGIDOC_ENCRYPTION_KEY={generate_key()}\n")
    conn, seen = FakeConn(), {}
    pipe = build_pipeline(load_keys(p), pg_connect=lambda dsn: seen.setdefault("pg", dsn) and conn,
                          redis_connect=lambda url: seen.setdefault("redis", url) and FakeRedis())
    assert "pw" in seen["pg"] and seen["redis"] == "redis://h:6379"
    assert isinstance(pipe.store.cipher, Cipher)  # reports are encrypted at rest in the real wiring
    assert pipe.authz.can("Rex", "report:approve") and not pipe.authz.can("Olive", "pipeline:run")
    pipe.setup()
    sql = " | ".join(s for s, _ in conn.cur.log)
    for table in ["chunks", "surgidoc_reports", "surgidoc_events", "surgidoc_audit"]:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql


def test_build_pipeline_needs_the_required_keys(tmp_path):
    p = tmp_path / "keys.txt"
    p.write_text("X=1\n")
    with pytest.raises(Exception, match="GOOGLE_AI_STUDIO_API_KEY|TIGERCLOUD|REDIS|SURGIDOC"):
        build_pipeline(load_keys(p), pg_connect=lambda d: FakeConn(), redis_connect=lambda u: FakeRedis())


# =============================== NFR-1 / AC-9 ==========================================================
ROOT = Path(__file__).resolve().parent.parent


def test_every_component_has_automated_tests():
    tests = "\n".join(p.read_text() for p in (ROOT / "tests").glob("test_*.py"))
    components = sorted(p.stem if p.is_file() else p.name for p in (ROOT / "surgidoc").iterdir()
                        if (p.suffix == ".py" and p.stem != "__init__") or (p.is_dir() and (p / "__init__.py").exists()))
    assert len(components) >= 14
    untested = [c for c in components if f"surgidoc.{c}" not in tests]
    assert untested == []


def test_the_suite_is_configured_to_run_with_one_command_and_stay_offline_by_default():
    cfg = (ROOT / "pyproject.toml").read_text()
    assert 'testpaths = ["tests"]' in cfg
    live = (ROOT / "tests" / "test_live.py").read_text()
    assert "SURGIDOC_LIVE" in live and "skipif" in live  # live checks are opt-in
