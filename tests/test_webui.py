"""T16: the local web UI, exercised over real HTTP (loopback) with the real Pipeline and fake network edges."""
import http.client
import json
import re
import shutil
import socket
import subprocess
import sys
import threading
import time

import pytest

from surgidoc.gateway import GatewayBlocked, Result
from surgidoc.ingest import Chunk
from surgidoc.observability import Telemetry
from surgidoc.pipeline import Pipeline
from surgidoc.security import AuditLog, Authorizer, MemorySink
from surgidoc.storage import ReportStore
from surgidoc.vision import Finding, Frame, VideoError
from surgidoc.webui import WebApp, main, serve_ui

USERS = {"Olive": {"operator"}, "Rex": {"reviewer"}, "Ada": {"admin"}}
TEXTS = ["Small sessile polyp, about 6 mm, in the sigmoid colon.", "No active bleeding."]
GOOD = json.dumps({"statements": [{"text": "A small sessile polyp was seen in the sigmoid colon.", "sources": ["F0"]},
                                  {"text": "No active bleeding was observed.", "sources": ["F1"]}]})
APPROVE = json.dumps({"approved": True, "issues": [], "feedback": ""})
REJECT = json.dumps({"approved": False, "issues": [{"statement": 0, "problem": "x"}], "feedback": "unsupported"})
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200
AVI = b"RIFF\x00\x00\x00\x00AVI LIST" + b"\x00" * 200
MKV = b"\x1a\x45\xdf\xa3" + b"\x00" * 200


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    real = socket.socket.connect

    def only_loopback(self, addr, *a, **k):
        if isinstance(addr, tuple) and addr[0] not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"external network access attempted: {addr}")
        return real(self, addr, *a, **k)

    monkeypatch.setattr(socket.socket, "connect", only_loopback)


class FakeGateway:
    def __init__(self, generator, judge):
        self.q = {"generator": list(generator), "judge": list(judge)}

    def generate(self, role, prompt, **kw):
        r = self.q[role].pop(0)
        if isinstance(r, Exception):
            raise r
        return Result(r, "m-" + role, 10, 5, 1)


class FakeCur:
    def __init__(self):
        self.log = []

    def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class FakeConn:
    def __init__(self):
        self.cur = FakeCur()

    def cursor(self):
        return self.cur

    def commit(self):
        pass

    def rollback(self):
        pass


class Env:
    pass


def make_app(generator=(GOOD,), judge=(APPROVE,), texts=TEXTS, extract=None, max_upload=5000):
    e = Env()
    e.telemetry = Telemetry()
    e.audit = AuditLog(MemorySink())
    e.authz = Authorizer(USERS, audit=e.audit)
    e.conn, e.gate = FakeConn(), threading.Event()
    e.store = ReportStore(e.conn, prefix="web")
    e.videos = []

    def extract_frames(video):
        e.videos.append(video)
        return [Frame(i, i * 2.0, b"\xff\xd8x") for i in range(len(texts))]

    class Describer:
        def describe(self, frames):
            return [Finding(f.index, f.t, texts[f.index], "gemini-vision") for f in frames]

    e.pipe = Pipeline(extract_frames=extract or extract_frames, describer=Describer(),
                      retrieve=lambda q: [Chunk("Cold snare is recommended.", "esge/poly.xml", 3)],
                      gateway=FakeGateway(generator, judge), store=e.store, audit=e.audit, authz=e.authz,
                      telemetry=e.telemetry, new_id=lambda: "rep-web-1")
    e.app = WebApp(e.pipe, max_upload=max_upload)
    e.srv = serve_ui(e.app, port=0)
    e.handler_errors = []  # any exception escaping a request handler is a bug, even if the client saw a clean status
    e.srv.handle_error = lambda request, client_address: e.handler_errors.append(sys.exc_info()[1])
    e.client = Client(e.srv)
    return e


def close(e):
    e.srv.shutdown()
    e.srv.server_close()
    e.app.close()
    assert e.handler_errors == [], f"unhandled exception(s) in the request handler: {e.handler_errors!r}"


@pytest.fixture
def env():
    e = make_app()
    yield e
    close(e)


class Client:
    def __init__(self, srv):
        self.port = srv.server_address[1]
        self.host = f"127.0.0.1:{self.port}"
        self.token = re.search(r'data-token="([^"]+)"', self.request("GET", "/", token=False)[2].decode()).group(1)

    def request(self, method, path, body=None, headers=None, token=True, host=None, origin=None, ctype=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = {"Host": host or self.host}
        if token:
            h["X-CSRF-Token"] = self.token
        if origin:
            h["Origin"] = origin
        if ctype:
            h["Content-Type"] = ctype
        h.update(headers or {})
        if isinstance(body, (dict, list)):
            body, h["Content-Type"] = json.dumps(body).encode(), "application/json"
        c.request(method, path, body=body, headers=h)
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, dict(r.getheaders()), data

    def json(self, method, path, body=None, **kw):
        status, headers, data = self.request(method, path, body, **kw)
        return status, (json.loads(data) if data and "json" in headers.get("Content-Type", "") else data)

    def upload(self, data=MP4, ext="mp4", ctype="application/octet-stream", **kw):
        return self.json("POST", f"/api/upload?ext={ext}", data, ctype=ctype, **kw)

    def status(self):
        return self.json("GET", "/api/status")[1]

    def wait(self, phases, timeout=8.0):
        phases, deadline = ({phases} if isinstance(phases, str) else set(phases)), time.time() + timeout
        while time.time() < deadline:
            s = self.status()
            if s["phase"] in phases:
                return s
            time.sleep(0.02)
        raise AssertionError(f"timed out waiting for {phases}; last status: {self.status()['phase']}")

    def run(self, operator="Olive"):
        return self.json("POST", "/api/run", {"operator": operator})


def to_review(e):
    assert e.client.upload()[0] == 200 and e.client.run()[0] == 202
    return e.client.wait("pending_review")


# =============================== binding, page and headers =====================================================
def test_the_server_binds_loopback_only(env):
    assert env.srv.server_address[0] == "127.0.0.1"
    with pytest.raises(ValueError):
        serve_ui(env.app, host="0.0.0.0", port=0)


def test_the_page_is_served_with_a_nonce_csp_token_and_no_inline_handlers(env):
    status, headers, body = env.client.request("GET", "/", token=False)
    page = body.decode()
    csp = headers["Content-Security-Policy"]
    nonce = re.search(r"script-src 'nonce-([^']+)'", csp).group(1)
    assert status == 200 and "text/html" in headers["Content-Type"] and f'nonce="{nonce}"' in page
    for needed in ["default-src 'none'", "frame-ancestors 'none'", "base-uri 'none'", "form-action 'none'", "connect-src 'self'"]:
        assert needed in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert re.search(r"\son[a-z]+\s*=", page) is None and "innerHTML" not in page  # no inline handlers, no HTML injection sink
    assert headers["X-Content-Type-Options"] == "nosniff" and headers["Cache-Control"] == "no-store"
    assert headers["X-Frame-Options"] == "DENY" and headers["Referrer-Policy"] == "no-referrer"
    assert "DRAFT" in page and "clinician" in page.lower()


def test_every_page_load_gets_a_fresh_nonce_but_the_same_session_token(env):
    a = env.client.request("GET", "/", token=False)
    b = env.client.request("GET", "/", token=False)
    nonce = lambda h: re.search(r"nonce-([^']+)", h["Content-Security-Policy"]).group(1)  # noqa: E731
    assert nonce(a[1]) != nonce(b[1]) and env.client.token in a[2].decode() and env.client.token in b[2].decode()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_inline_javascript_is_syntactically_valid(env, tmp_path):
    page = env.client.request("GET", "/", token=False)[2].decode()
    script = re.search(r"<script nonce=\"[^\"]+\">(.*?)</script>", page, re.S).group(1)
    (tmp_path / "ui.js").write_text(script)
    done = subprocess.run(["node", "--check", str(tmp_path / "ui.js")], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_unknown_paths_and_wrong_methods(env):
    assert env.client.request("GET", "/etc/passwd")[0] == 404 and env.client.request("GET", "/api/nothing")[0] == 404
    assert env.client.request("DELETE", "/")[0] == 405 and env.client.request("PUT", "/api/run")[0] == 405
    assert env.client.request("GET", "/api/run")[0] == 405 and env.client.request("POST", "/api/status")[0] == 405


# =============================== DNS-rebinding, cross-site and CSRF protection ===================================
@pytest.mark.parametrize("path", ["/", "/api/status", "/dashboard", "/metrics.json"])
def test_requests_with_a_foreign_host_header_are_refused(env, path):
    assert env.client.request("GET", path, host="evil.example")[0] == 403
    assert env.client.request("GET", path, host=f"evil.example:{env.client.port}")[0] == 403


def test_localhost_names_are_accepted_as_host(env):
    assert env.client.request("GET", "/", host=f"localhost:{env.client.port}", token=False)[0] == 200


def test_posts_from_other_origins_are_refused(env):
    assert env.client.json("POST", "/api/run", {"operator": "Olive"}, origin="http://evil.example")[0] == 403
    assert env.client.json("POST", "/api/run", {"operator": "Olive"}, origin=f"http://127.0.0.1:{env.client.port}")[0] != 403


@pytest.mark.parametrize("path", ["/api/upload?ext=mp4", "/api/run", "/api/approve", "/api/reject", "/api/reset"])
def test_actions_need_the_csrf_token(env, path):
    assert env.client.request("POST", path, b"{}", token=False)[0] == 403
    assert env.client.request("POST", path, b"{}", headers={"X-CSRF-Token": "wrong"}, token=False)[0] == 403


def test_status_and_pdf_also_need_the_token(env):
    assert env.client.request("GET", "/api/status", token=False)[0] == 403
    assert env.client.request("GET", "/api/report.pdf", token=False)[0] == 403


# =============================== upload ===========================================================================
def test_a_valid_video_is_stored_under_a_random_name_and_reported(env):
    status, out = env.client.upload(MP4, ext="mp4")
    assert status == 200 and out == {"ok": True, "size": len(MP4)}
    files = list(env.app.tmpdir.iterdir())
    assert len(files) == 1 and files[0].suffix == ".mp4" and re.fullmatch(r"[0-9a-f]{32}\.mp4", files[0].name)
    assert files[0].read_bytes() == MP4 and env.client.status()["phase"] == "uploaded"


@pytest.mark.parametrize("data,ext", [(MP4, "mp4"), (MP4, "mov"), (AVI, "avi"), (MKV, "mkv")])
def test_known_video_containers_are_accepted(env, data, ext):
    assert env.client.upload(data, ext=ext)[0] == 200


@pytest.mark.parametrize("ext", ["exe", "../../etc/passwd", "mp4/../x", "", "MP4%00"])
def test_other_extensions_and_path_tricks_are_refused(env, ext):
    assert env.client.upload(MP4, ext=ext)[0] == 415
    assert list(env.app.tmpdir.iterdir()) == []


def test_content_that_is_not_a_video_is_refused_and_not_kept(env):
    assert env.client.upload(b"MZ\x90\x00 not a video at all" + b"\x00" * 100)[0] == 415
    assert env.client.upload(b"<html>hi</html>" * 20, ext="mp4")[0] == 415
    assert list(env.app.tmpdir.iterdir()) == [] and env.client.status()["phase"] == "idle"


def test_wrong_content_types_are_refused(env):
    assert env.client.upload(ctype="text/html")[0] == 415 and env.client.upload(ctype="video/mp4")[0] == 200


def test_oversized_uploads_are_refused_before_the_body_is_read(env):
    c = http.client.HTTPConnection("127.0.0.1", env.client.port, timeout=5)
    c.putrequest("POST", "/api/upload?ext=mp4", skip_host=True)
    for k, v in {"Host": env.client.host, "X-CSRF-Token": env.client.token, "Content-Type": "application/octet-stream",
                 "Content-Length": str(10 ** 12)}.items():
        c.putheader(k, v)
    c.endheaders()  # announce a terabyte, send nothing: the server must answer without waiting for it
    assert c.getresponse().status == 413


def test_the_size_limit_is_exact(env):
    ok = MP4 + b"\x00" * (5000 - len(MP4))
    assert env.client.upload(ok)[0] == 200
    assert env.client.upload(ok + b"\x00")[0] == 413


def test_uploads_without_a_content_length_are_refused(env):
    c = http.client.HTTPConnection("127.0.0.1", env.client.port, timeout=5)
    c.request("POST", "/api/upload?ext=mp4", body=iter([MP4]), headers={"Host": env.client.host, "X-CSRF-Token": env.client.token,
                                                                       "Content-Type": "application/octet-stream"})
    assert c.getresponse().status == 411


def test_a_second_upload_replaces_and_deletes_the_first(env):
    env.client.upload(MP4)
    first = list(env.app.tmpdir.iterdir())[0]
    env.client.upload(AVI, ext="avi")
    files = list(env.app.tmpdir.iterdir())
    assert len(files) == 1 and files[0] != first and not first.exists()


def test_uploads_are_refused_while_a_run_is_in_progress():
    gate = threading.Event()
    e = make_app(extract=lambda v: (gate.wait(5), [Frame(0, 0.0, b"x")])[1], texts=["a polyp"])
    try:
        e.client.upload()
        e.client.run()
        e.client.wait("running")
        assert e.client.upload()[0] == 409 and e.client.json("POST", "/api/reset")[0] == 409
        gate.set()
        e.client.wait({"pending_review", "judge_rejected", "guardrails_blocked", "failed"})
    finally:
        gate.set()
        close(e)


# =============================== running the pipeline ========================================================
def test_run_needs_an_uploaded_video(env):
    assert env.client.run()[0] == 409


def test_unregistered_or_unauthorised_operators_cannot_run_and_the_denial_is_audited(env):
    env.client.upload()
    assert env.client.run("mallory")[0] == 403
    assert env.client.status()["phase"] == "uploaded" and env.audit.entries()[-1]["outcome"] == "denied"
    assert env.client.run("Ada")[0] == 403  # an admin is not an operator
    assert env.videos == []  # nothing executed


def test_a_run_goes_through_the_steps_and_ends_pending_review(env):
    s = to_review(env)
    labels = {x["id"]: x["done"] for x in s["steps"]}
    assert all(labels[k] for k in ["frames", "describe", "retrieve", "agents", "guardrails"]) and not labels["review"]
    assert [x["text"][:12] for x in s["report"]["statements"]] == ["A small sess", "No active bl"]
    assert s["report"]["sources"]["F0"]["text"] == TEXTS[0] and s["report"]["statements"][0]["sources"] == ["F0"]
    assert s["pdf_ready"] is False and s["report"]["id"] == "rep-web-1"


def test_a_second_run_while_busy_is_refused_and_only_one_starts():
    gate = threading.Event()
    e = make_app(extract=lambda v: (gate.wait(5), [Frame(0, 0.0, b"x")])[1], texts=["a polyp"],
                 generator=(json.dumps({"statements": [{"text": "A polyp.", "sources": ["F0"]}]}),))
    try:
        e.client.upload()
        results = []
        ts = [threading.Thread(target=lambda: results.append(e.client.run()[0])) for _ in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert sorted(results) == [202, 409, 409, 409]
        gate.set()
        e.client.wait("pending_review")
    finally:
        gate.set()
        close(e)


def test_the_uploaded_file_is_deleted_after_the_run_and_never_exposed(env):
    s = to_review(env)
    assert list(env.app.tmpdir.iterdir()) == []
    assert str(env.app.tmpdir) not in json.dumps(s) and "/tmp" not in json.dumps(s)


def test_judge_rejection_blocked_and_guardrail_outcomes_are_reported_and_not_approvable():
    cases = [((GOOD, GOOD, GOOD), (REJECT, REJECT, REJECT), "judge_rejected"),
             ((GatewayBlocked("b"),), (), "blocked"),
             ((json.dumps({"statements": [{"text": "The patient has colorectal cancer.", "sources": ["F0"]}]}),), (APPROVE,),
              "guardrails_blocked")]
    for gen, judge, phase in cases:
        e = make_app(generator=gen, judge=judge)
        try:
            e.client.upload()
            e.client.run()
            s = e.client.wait(phase)
            assert s["report"] is None  # an unreviewable draft is never shown to the reviewer
            assert e.client.json("POST", "/api/approve", {"name": "Rex"})[0] == 409
            if phase == "guardrails_blocked":
                assert s["violations"] == ["diagnosis"] and "colorectal" not in json.dumps(s)
        finally:
            close(e)


def test_unusable_frames_are_reported():
    e = make_app(texts=["unclear", "Unclear."])
    try:
        e.client.upload()
        e.client.run()
        assert e.client.wait("no_usable_findings")["report"] is None
    finally:
        close(e)


def test_a_pipeline_failure_is_reported_generically_without_secrets_or_paths():
    def boom(video):
        raise VideoError("cannot read /secret/dir/video.mp4 password=hunter2")

    e = make_app(extract=boom)
    try:
        e.client.upload()
        e.client.run()
        s = e.client.wait("failed")
        assert "VideoError" in s["message"] and "hunter2" not in json.dumps(s) and "/secret" not in json.dumps(s)
        assert list(e.app.tmpdir.iterdir()) == []  # the upload is cleaned up even on failure
    finally:
        close(e)


# =============================== human review ===================================================================
def test_only_a_reviewer_can_approve_and_the_denial_is_audited(env):
    to_review(env)
    assert env.client.json("POST", "/api/approve", {"name": "Olive"})[0] == 403
    assert env.client.json("POST", "/api/approve", {"name": "stranger"})[0] == 403
    assert env.client.status()["phase"] == "pending_review" and env.client.status()["pdf_ready"] is False
    assert [e["outcome"] for e in env.audit.entries()].count("denied") == 2


def test_a_reviewer_approves_and_downloads_the_pdf(env):
    to_review(env)
    status, out = env.client.json("POST", "/api/approve", {"name": "Rex"})
    assert status == 200 and out["ok"] is True
    s = env.client.status()
    assert s["phase"] == "approved" and s["approver"] == "Rex" and s["pdf_ready"] is True
    status, headers, pdf = env.client.request("GET", f"/api/report.pdf?t={env.client.token}", token=False)
    assert status == 200 and pdf.startswith(b"%PDF") and headers["Content-Type"] == "application/pdf"
    assert "attachment" in headers["Content-Disposition"] and re.search(r'filename="[A-Za-z0-9._-]+\.pdf"', headers["Content-Disposition"])
    assert [p for sql, p in env.conn.cur.log if sql.startswith("INSERT INTO web_reports")]  # stored as well


def test_the_pdf_is_not_available_before_approval_or_after_rejection(env):
    to_review(env)
    assert env.client.request("GET", "/api/report.pdf")[0] == 409
    env.client.json("POST", "/api/reject", {"name": "Rex", "reason": "Not accurate."})
    assert env.client.status()["phase"] == "rejected" and env.client.request("GET", "/api/report.pdf")[0] == 409


def test_reject_requires_a_reviewer_name_and_a_reason(env):
    to_review(env)
    assert env.client.json("POST", "/api/reject", {"name": "Olive", "reason": "x"})[0] == 403
    assert env.client.json("POST", "/api/reject", {"name": "Rex", "reason": ""})[0] == 400
    assert env.client.json("POST", "/api/reject", {"name": "Rex", "reason": "No."})[0] == 200


def test_approval_needs_a_name_and_cannot_happen_twice(env):
    to_review(env)
    assert env.client.json("POST", "/api/approve", {})[0] == 400 and env.client.json("POST", "/api/approve", {"name": ""})[0] == 400
    assert env.client.json("POST", "/api/approve", {"name": "Rex"})[0] == 200
    assert env.client.json("POST", "/api/approve", {"name": "Rex"})[0] == 409


def test_a_storage_failure_after_approval_can_be_retried(env):
    to_review(env)
    real = env.store.save_report
    env.store.save_report = lambda *a: (_ for _ in ()).throw(RuntimeError("db down: postgres://u:SECRETPW@h"))
    status, out = env.client.json("POST", "/api/approve", {"name": "Rex"})
    assert status == 500 and "SECRETPW" not in json.dumps(out)
    assert env.client.status()["phase"] == "pending_review" and env.client.status()["pdf_ready"] is False
    env.store.save_report = real
    assert env.client.json("POST", "/api/approve", {"name": "Rex"})[0] == 200 and env.client.status()["pdf_ready"] is True


def test_reset_clears_the_state_for_the_next_video(env):
    to_review(env)
    env.client.json("POST", "/api/approve", {"name": "Rex"})
    assert env.client.json("POST", "/api/reset")[0] == 200
    s = env.client.status()
    assert s["phase"] == "idle" and s["report"] is None and s["pdf_ready"] is False
    assert env.client.request("GET", "/api/report.pdf")[0] == 409


def test_a_malformed_or_oversized_json_body_is_rejected(env):
    assert env.client.request("POST", "/api/run", b"{not json", ctype="application/json")[0] == 400
    assert env.client.request("POST", "/api/run", b'{"operator": "' + b"a" * 5000 + b'"}', ctype="application/json")[0] == 413
    assert env.client.request("POST", "/api/run", b"[1,2]", ctype="application/json")[0] == 400


def test_a_rejected_body_never_reaches_the_action_it_was_meant_for(env):
    calls = []
    env.app.run = lambda *a: calls.append("run") or (200, {})
    env.app.approve = lambda *a: calls.append("approve") or (200, {})
    env.app.reject = lambda *a: calls.append("reject") or (200, {})
    for path in ("/api/run", "/api/approve", "/api/reject"):
        assert env.client.request("POST", path, b"x" * 5000, ctype="application/json")[0] == 413  # too large
        assert env.client.request("POST", path, b"{broken", ctype="application/json")[0] == 400  # not JSON
        assert env.client.request("POST", path, b"[1]", ctype="application/json")[0] == 400  # not an object
        c = http.client.HTTPConnection("127.0.0.1", env.client.port, timeout=5)
        c.putrequest("POST", path, skip_host=True)
        for k, v in {"Host": env.client.host, "X-CSRF-Token": env.client.token, "Content-Length": "-5"}.items():
            c.putheader(k, v)
        c.endheaders()
        assert c.getresponse().status == 413  # negative length
    assert calls == []


def test_a_client_that_hangs_up_before_the_response_does_not_break_the_server(env):
    for _ in range(5):
        s = socket.create_connection(("127.0.0.1", env.client.port), timeout=5)
        s.sendall(f"GET /dashboard HTTP/1.0\r\nHost: {env.client.host}\r\n\r\n".encode())
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")  # close with RST
        s.close()
    assert env.client.status()["phase"] == "idle"  # still serving


def test_statement_text_is_returned_as_data_not_markup(env):
    e = make_app(generator=(json.dumps({"statements": [{"text": "A polyp <script>alert(1)</script> seen.", "sources": ["F0"]}]}),))
    try:
        # guardrails do not block markup, so the UI must treat it as plain text (textContent on the client)
        e.client.upload()
        e.client.run()
        s = e.client.wait({"pending_review", "guardrails_blocked"})
        page = e.client.request("GET", "/", token=False)[2].decode()
        assert "textContent" in page and "innerHTML" not in page and "<script>alert(1)</script>" not in page
        assert s["phase"] in {"pending_review", "guardrails_blocked"}
    finally:
        close(e)


# =============================== dashboard and cleanup ===========================================================
def test_the_dashboard_and_metrics_are_served_and_reflect_the_run(env):
    to_review(env)
    status, headers, body = env.client.request("GET", "/dashboard", token=False)
    assert status == 200 and "text/html" in headers["Content-Type"] and b"pipeline.run" in body
    assert "default-src 'none'" in headers["Content-Security-Policy"] and "script-src" not in headers["Content-Security-Policy"]
    m = json.loads(env.client.request("GET", "/metrics.json", token=False)[2])
    assert m["operations"]["pipeline.run"]["count"] == 1


def test_closing_removes_the_temporary_directory():
    e = make_app()
    e.client.upload()
    tmp = e.app.tmpdir
    assert tmp.exists()
    close(e)
    assert not tmp.exists()


def test_errors_never_leak_exception_text(env):
    env.client.upload()
    env.pipe.run = lambda *a: (_ for _ in ()).throw(RuntimeError("secret=abc123"))
    env.client.run()
    s = env.client.wait("failed")
    assert "abc123" not in json.dumps(s)


# =============================== command line =====================================================================
class FakePipe:
    def __init__(self):
        self.set_up, self.authz, self.telemetry = False, None, Telemetry()

    def setup(self):
        self.set_up = True


def test_main_builds_the_pipeline_sets_up_tables_and_serves_on_loopback(tmp_path, capsys):
    keys = tmp_path / "keys.txt"
    keys.write_text("X=1\n")
    built = []
    pipe = FakePipe()
    srv = main(["--port", "0", "--keys", str(keys)], build=lambda k: built.append(k) or pipe, block=False)
    try:
        assert pipe.set_up and len(built) == 1 and srv.server_address[0] == "127.0.0.1"
        out = capsys.readouterr().out
        assert f"http://127.0.0.1:{srv.server_address[1]}" in out
    finally:
        srv.shutdown()
        srv.server_close()
        srv.app.close()


@pytest.mark.parametrize("args", [["--port", "99999"], ["--port", "-1"], ["--port", "abc"], ["--host", "0.0.0.0"]])
def test_main_rejects_bad_ports_and_has_no_way_to_bind_elsewhere(tmp_path, args):
    keys = tmp_path / "keys.txt"
    keys.write_text("X=1\n")
    with pytest.raises(SystemExit):
        main(args + ["--keys", str(keys)], build=lambda k: FakePipe(), block=False)


def test_main_reports_a_missing_keys_file_cleanly(tmp_path):
    with pytest.raises(SystemExit):
        main(["--keys", str(tmp_path / "nope.txt")], build=lambda k: FakePipe(), block=False)
