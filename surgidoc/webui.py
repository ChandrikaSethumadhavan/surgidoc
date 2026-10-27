"""T16: a small local web UI (standard library only): upload a video, watch the pipeline steps, review the cited draft,
approve or reject as a registered reviewer, download the approved PDF, open the metrics dashboard.

Start:  python -m surgidoc.webui   then open http://127.0.0.1:8765

Single user, one run at a time, no login (the reviewer must be named in SURGIDOC_USERS and the existing role rules apply).
Hardening: loopback bind only; Host and Origin checks (DNS rebinding, cross-site posts); per-session CSRF token on every
API call; nonce-based CSP; upload size/type/magic checks with a random server-side file name; no exception text, path or
report text in errors or audit entries; the page writes data with textContent only.
"""
import argparse
import hmac
import html
import json
import secrets
import shutil
import tempfile
import threading
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from surgidoc.approval import NotApproved, ReviewError
from surgidoc.config import MissingKey, load_keys
from surgidoc.observability import render_dashboard_html
from surgidoc.pipeline import build_pipeline
from surgidoc.security import PermissionDenied

LOOPBACK = {"127.0.0.1", "::1", "localhost"}
ALLOWED_EXT = {"mp4", "mov", "avi", "mkv"}
MAX_JSON = 4096
STEPS = [("frames", "Extract frames", "pipeline.frames"), ("describe", "Describe frames", "pipeline.describe"),
         ("retrieve", "Retrieve knowledge", "pipeline.retrieve"), ("agents", "Generate and judge", "pipeline.agents")]
OUTCOMES = {"pending_review": ("pending_review", "Draft ready: a reviewer must approve or reject it."),
            "rejected": ("judge_rejected", "The judge rejected every attempt, so no draft can be reviewed."),
            "blocked": ("blocked", "The model's safety filter blocked this video's content."),
            "blocked_by_guardrails": ("guardrails_blocked", "The draft broke a safety rule and was stopped before review."),
            "no_usable_findings": ("no_usable_findings", "No frame could be described (all unclear).")}


def _looks_like_video(head):
    return (head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip")
            or (head[:4] == b"RIFF" and head[8:12] == b"AVI ") or head[:4] == b"\x1a\x45\xdf\xa3")


class WebApp:
    def __init__(self, pipeline, max_upload=200 * 1024 * 1024):
        self.pipeline, self.max_upload = pipeline, max_upload
        self.token = secrets.token_urlsafe(32)
        self.tmpdir = Path(tempfile.mkdtemp(prefix="surgidoc-ui-"))  # mode 0700
        self.lock, self.review_lock, self.upload_lock = threading.Lock(), threading.Lock(), threading.Lock()
        self._reset()

    def _reset(self):
        self.phase, self.message, self.upload = "idle", "", None  # upload = (path, size)
        self.review = self.pdf = self.approver = self.report_id = None
        self.violations, self.baseline = [], {}

    def close(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # --- upload ---------------------------------------------------------------------------------------------
    def begin_upload(self, ext):
        """-> (part path, final path) or None when busy. The client's file name is never used."""
        with self.lock:
            if self.phase in ("running", "pending_review") or not self.upload_lock.acquire(blocking=False):
                return None
        name = uuid.uuid4().hex
        return self.tmpdir / f"{name}.{ext}.part", self.tmpdir / f"{name}.{ext}"

    def abort_upload(self, part):
        part.unlink(missing_ok=True)
        self.upload_lock.release()

    def commit_upload(self, part, final, size):
        try:
            with self.lock:
                if self.upload:
                    self.upload[0].unlink(missing_ok=True)  # a new upload replaces the old one
                part.replace(final)
                self._reset()
                self.phase, self.upload = "uploaded", (final, size)
        finally:
            self.upload_lock.release()

    # --- run --------------------------------------------------------------------------------------------------
    def run(self, operator):
        if not isinstance(operator, str) or not operator.strip():
            return 400, {"error": "an operator name is required"}
        with self.lock:
            if self.phase != "uploaded":
                return 409, {"error": "upload a video first" if self.phase == "idle" else "busy: finish or reset first"}
            path = self.upload[0]
        try:
            self.pipeline.authz.require(operator, "pipeline:run", "upload")  # a denial is audited and nothing starts
        except PermissionDenied:
            return 403, {"error": "this user may not run the pipeline"}
        with self.lock:
            if self.phase != "uploaded":
                return 409, {"error": "busy"}
            self.phase, self.message = "running", "Running the pipeline."
            self.baseline = {k: v["count"] for k, v in self.pipeline.telemetry.snapshot()["operations"].items()}
        threading.Thread(target=self._worker, args=(path, operator), daemon=True).start()
        return 202, {"ok": True}

    def _worker(self, path, operator):
        try:
            res = self.pipeline.run(str(path), operator)
            with self.lock:
                self.phase, self.message = OUTCOMES[res.status]
                self.violations = list(res.violations)
                if res.status == "pending_review":
                    self.review, self.report_id = res.review, res.report_id
        except Exception as e:  # the exception text may hold paths or secrets: only its type is shown
            with self.lock:
                self.phase, self.message = "failed", f"The pipeline failed ({type(e).__name__}). Details are in the audit log."
        finally:
            path.unlink(missing_ok=True)
            with self.lock:
                self.upload = None

    # --- review -------------------------------------------------------------------------------------------------
    def _review_action(self, name, action, ok_phase):
        if not isinstance(name, str) or not name.strip():
            return 400, {"error": "your name is required"}
        if not self.review_lock.acquire(blocking=False):
            return 409, {"error": "another action is in progress"}
        try:
            with self.lock:
                if self.phase != "pending_review" or self.review is None:
                    return 409, {"error": "there is no draft waiting for review"}
                review = self.review
            try:
                result = action(review, name.strip())
            except PermissionDenied:
                return 403, {"error": "this user may not review reports"}
            except ReviewError as e:  # our own validation messages (e.g. a missing reason)
                return 400, {"error": str(e)}
            except NotApproved:
                return 409, {"error": "the report is not approved"}
            except Exception:
                return 500, {"error": "the action could not be completed; try again"}
            with self.lock:
                self.phase, self.approver = ok_phase, name.strip()
                if result is not None:
                    self.pdf = result.pdf
            return 200, {"ok": True}
        finally:
            self.review_lock.release()

    def approve(self, name):
        def act(review, who):  # if storage failed after the approval, a second click retries only the publish step
            return self.pipeline.publish(review, who) if review.item.state == "approved" else self.pipeline.approve(review, who)

        return self._review_action(name, act, "approved")

    def reject(self, name, reason):
        def act(review, who):
            self.pipeline.reject(review, who, reason if isinstance(reason, str) else "")

        return self._review_action(name, act, "rejected")

    def reset(self):
        with self.lock:
            if self.phase == "running":
                return 409, {"error": "a run is in progress"}
            if self.upload:
                self.upload[0].unlink(missing_ok=True)
            self._reset()
        return 200, {"ok": True}

    def get_pdf(self):
        with self.lock:
            return self.pdf if self.phase == "approved" else None

    # --- status ----------------------------------------------------------------------------------------------------
    def status(self):
        ops = self.pipeline.telemetry.snapshot()["operations"]
        with self.lock:
            ran = self.phase not in ("idle", "uploaded")
            steps = [{"id": i, "label": label, "done": ran and ops.get(op, {}).get("count", 0) > self.baseline.get(op, 0)}
                     for i, label, op in STEPS]
            steps.append({"id": "guardrails", "label": "Safety rules",
                          "done": self.phase in ("pending_review", "approved", "rejected", "guardrails_blocked")})
            steps.append({"id": "review", "label": "Human review", "done": self.phase in ("approved", "rejected")})
            report = None
            if self.review is not None and self.phase in ("pending_review", "approved", "rejected"):
                item = self.review.item
                statements = [{"text": s.text, "sources": list(s.sources)} for s in item.statements]
                cited = dict.fromkeys(c for s in statements for c in s["sources"])
                report = {"id": self.report_id, "revision": item.revision, "statements": statements,
                          "sources": {c: {"kind": item.sources[c].get("kind"), "text": str(item.sources[c].get("text", ""))[:300],
                                          **({"source": item.sources[c].get("source")} if item.sources[c].get("kind") == "knowledge" else {})}
                                      for c in cited if c in item.sources}}
            return {"phase": self.phase, "message": self.message, "steps": steps, "report": report,
                    "violations": list(self.violations), "approver": self.approver, "pdf_ready": self.pdf is not None and self.phase == "approved",
                    "upload": {"size": self.upload[1]} if self.upload else None}


# =============================== the page ================================================================================
CSS = ("body{font:15px system-ui,sans-serif;margin:0;background:#fff;color:#111}main{max-width:820px;margin:0 auto;padding:1rem 1.25rem 3rem}"
       "h1{font-size:1.3rem;margin:.2rem 0}h2{font-size:1rem;margin:1.4rem 0 .4rem}.banner{background:#fff4d6;border:1px solid #e0b84c;color:#5a4300;"
       "padding:.6rem .8rem;border-radius:6px;margin:.8rem 0}.row{display:flex;gap:.5rem;flex-wrap:wrap;align-items:center;margin:.4rem 0}"
       "button{padding:.4rem .9rem;border-radius:6px;border:1px solid #888;background:#f2f2f2;cursor:pointer}button:disabled{opacity:.45;cursor:default}"
       "input[type=text]{padding:.35rem .5rem;border:1px solid #888;border-radius:6px;min-width:12rem}progress{width:12rem}"
       "ul.steps{list-style:none;padding:0}ul.steps li{padding:.15rem 0}.done{color:#0a7a2f}.todo{color:#777}.bad{color:#b00020}.ok{color:#0a7a2f}"
       "ol.st li{margin:.35rem 0}.cite{font-size:.8rem;color:#555;margin-left:.4rem}.src{font-size:.85rem;color:#444;margin:.2rem 0}"
       "a.btn{display:inline-block;padding:.4rem .9rem;border:1px solid #0a7a2f;border-radius:6px;color:#0a7a2f;text-decoration:none}"
       "@media(prefers-color-scheme:dark){body{background:#121212;color:#eaeaea}.banner{background:#3a3000;border-color:#8a7000;color:#ffe9a0}"
       "button{background:#2a2a2a;color:#eee;border-color:#666}input[type=text]{background:#1e1e1e;color:#eee;border-color:#666}"
       ".done,.ok{color:#5fd68a}.bad{color:#ff8a9b}.todo,.cite,.src{color:#aaa}a.btn{color:#5fd68a;border-color:#5fd68a}}")

SCRIPT = r"""
(() => {
  const token = document.body.dataset.token;
  const $ = (id) => document.getElementById(id);
  const api = async (path, opts = {}) => {
    const r = await fetch(path, {...opts, headers: {"X-CSRF-Token": token, ...(opts.headers || {})}});
    let data = {};
    try { data = JSON.parse(await r.text()); } catch (e) { data = {}; }
    return {status: r.status, data};
  };
  const post = (path, body) => api(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})});
  const say = (text, bad) => { const el = $("msg"); el.textContent = text || ""; el.className = bad ? "bad" : "ok"; };
  const el = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; };
  const idle = ["idle", "uploaded", "approved", "rejected", "judge_rejected", "blocked", "guardrails_blocked", "no_usable_findings", "failed"];

  function render(s) {
    $("phase").textContent = s.phase.replace(/_/g, " ");
    $("status").textContent = s.message;
    const steps = $("steps"); steps.replaceChildren();
    s.steps.forEach((x) => steps.append(el("li", (x.done ? "✓ " : "○ ") + x.label, x.done ? "done" : "todo")));
    const list = $("statements"), sources = $("sources"); list.replaceChildren(); sources.replaceChildren();
    if (s.report) {
      s.report.statements.forEach((st) => { const li = el("li", st.text); li.append(el("span", "[" + st.sources.join(", ") + "]", "cite")); list.append(li); });
      Object.entries(s.report.sources).forEach(([id, src]) => {
        const where = src.kind === "knowledge" ? src.source : "video frame finding";
        sources.append(el("div", "[" + id + "] " + where + ": " + src.text, "src"));
      });
    }
    if (s.violations.length) $("status").textContent += " Rules: " + s.violations.join(", ") + ".";
    $("pick").disabled = !idle.includes(s.phase);
    $("run").disabled = s.phase !== "uploaded";
    $("operator").disabled = s.phase !== "uploaded";
    const review = s.phase === "pending_review";
    ["reviewer", "approve", "reason", "reject"].forEach((id) => { $(id).disabled = !review; });
    $("reset").disabled = s.phase === "running";
    const dl = $("download");
    dl.hidden = !s.pdf_ready;
    dl.href = s.pdf_ready ? "/api/report.pdf?t=" + encodeURIComponent(token) : "#";
    $("approved").textContent = s.approver && s.phase === "approved" ? "Approved by " + s.approver + "." : (s.phase === "rejected" ? "Rejected by " + s.approver + "." : "");
  }
  async function refresh() { const r = await api("/api/status"); if (r.status === 200) render(r.data); }

  $("pick").addEventListener("change", () => {
    const f = $("pick").files[0]; if (!f) return;
    const ext = (f.name.split(".").pop() || "").toLowerCase();
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload?ext=" + encodeURIComponent(ext));
    xhr.setRequestHeader("X-CSRF-Token", token);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) { $("bar").value = e.loaded; $("bar").max = e.total; } };
    xhr.onload = () => { let d = {}; try { d = JSON.parse(xhr.responseText); } catch (e) { d = {}; }
      say(xhr.status === 200 ? "Uploaded " + d.size + " bytes." : (d.error || "Upload failed."), xhr.status !== 200); $("pick").value = ""; refresh(); };
    xhr.onerror = () => say("Upload failed.", true);
    $("bar").value = 0; xhr.send(f);
  });
  $("run").addEventListener("click", async () => { const r = await post("/api/run", {operator: $("operator").value}); say(r.data.error || "Started.", r.status >= 400); refresh(); });
  $("approve").addEventListener("click", async () => { const r = await post("/api/approve", {name: $("reviewer").value}); say(r.data.error || "Approved.", r.status >= 400); refresh(); });
  $("reject").addEventListener("click", async () => { const r = await post("/api/reject", {name: $("reviewer").value, reason: $("reason").value}); say(r.data.error || "Rejected.", r.status >= 400); refresh(); });
  $("reset").addEventListener("click", async () => { const r = await post("/api/reset"); say(r.data.error || "Ready for a new video.", r.status >= 400); refresh(); });
  refresh(); setInterval(refresh, 800);
})();
"""


def render_page(token, nonce):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SurgiDoc</title><style nonce="{nonce}">{CSS}</style></head>
<body data-token="{html.escape(token)}"><main>
<h1>SurgiDoc - endoscopy documentation assistant</h1>
<div class="banner"><strong>DRAFT only.</strong> AI-assisted documentation. It is not a diagnosis; a clinician must review, edit and approve every report.</div>
<h2>1. Upload a video</h2>
<div class="row"><input id="pick" type="file" accept=".mp4,.mov,.avi,.mkv,video/*"><progress id="bar" value="0" max="1"></progress></div>
<h2>2. Run</h2>
<div class="row"><input id="operator" type="text" placeholder="Your name (operator)" autocomplete="off"><button id="run" disabled>Run pipeline</button>
<button id="reset" disabled>New video</button></div>
<p>State: <strong id="phase">idle</strong> <span id="status"></span></p>
<ul class="steps" id="steps"></ul>
<h2>3. Review the draft</h2>
<ol class="st" id="statements"></ol><div id="sources"></div>
<div class="row"><input id="reviewer" type="text" placeholder="Reviewer name" autocomplete="off" disabled><button id="approve" disabled>Approve</button>
<input id="reason" type="text" placeholder="Reason (to reject)" autocomplete="off" disabled><button id="reject" disabled>Reject</button></div>
<p id="approved"></p>
<h2>4. Result</h2>
<div class="row"><a id="download" class="btn" href="#" hidden>Download approved PDF</a> <a class="btn" href="/dashboard" target="_blank" rel="noopener">Metrics dashboard</a></div>
<p id="msg" class="ok"></p>
</main><script nonce="{nonce}">{SCRIPT}</script></body></html>"""


# =============================== http ======================================================================================
def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version, sys_version = "SurgiDoc", ""

        def log_message(self, *args):  # no request logging: paths, tokens and clients are not needed anywhere
            pass

        def handle(self):
            try:
                super().handle()
            except (ConnectionError, TimeoutError):  # the client hung up or stalled mid-request: nothing to answer
                pass

        # --- guards ---
        def _allowed_hosts(self):
            port = self.server.server_address[1]
            return {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

        def _query(self):
            return urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query, keep_blank_values=True)

        def _guard(self, need_token):
            hosts = self._allowed_hosts()
            origin = self.headers.get("Origin")
            if self.headers.get("Host", "") not in hosts or (origin is not None and origin not in {f"http://{h}" for h in hosts}):
                self._json(403, {"error": "forbidden"})
                return False
            if need_token:
                given = self.headers.get("X-CSRF-Token") or (self._query().get("t") or [""])[0]
                if not hmac.compare_digest(given.encode(), app.token.encode()):
                    self._json(403, {"error": "forbidden"})
                    return False
            return True

        # --- responses ---
        def _send(self, status, body, ctype, extra=None):
            self.send_response(status)
            headers = {"Content-Type": ctype, "Content-Length": str(len(body)), "Cache-Control": "no-store",
                       "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
                       "Cross-Origin-Resource-Policy": "same-origin", "Content-Security-Policy": "default-src 'none'"}
            headers.update(extra or {})
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, payload):
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _fail(self, status, message):
            self._json(status, {"error": message})
            return None  # callers return this to mean "a response has already been sent"

        def _body(self, limit):
            if "Transfer-Encoding" in self.headers or self.headers.get("Content-Length") is None:
                return self._fail(411, "Content-Length is required")
            try:
                n = int(self.headers["Content-Length"])
            except ValueError:
                return self._fail(400, "bad request")
            if n < 0 or n > limit:
                return self._fail(413, "too large")
            return self.rfile.read(n)

        def _json_body(self):
            raw = self._body(MAX_JSON)
            if raw is None:
                return None
            try:
                data = json.loads(raw or b"{}")
            except ValueError:
                return self._fail(400, "invalid JSON")
            return data if isinstance(data, dict) else self._fail(400, "expected a JSON object")

        # --- routes ---
        GET_ROUTES = {"/", "/dashboard", "/metrics.json", "/api/status", "/api/report.pdf"}
        POST_ROUTES = {"/api/upload", "/api/run", "/api/approve", "/api/reject", "/api/reset"}

        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path
            if path not in self.GET_ROUTES:
                return self._json(405, {"error": "method not allowed"}) if path in self.POST_ROUTES else self._json(404, {"error": "not found"})
            if not self._guard(path.startswith("/api/")):
                return
            if path == "/":
                nonce = secrets.token_urlsafe(16)
                csp = (f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; connect-src 'self'; "
                       "img-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                return self._send(200, render_page(app.token, nonce).encode(), "text/html; charset=utf-8", {"Content-Security-Policy": csp})
            if path == "/dashboard":
                return self._send(200, render_dashboard_html(app.pipeline.telemetry.snapshot()).encode(), "text/html; charset=utf-8",
                                  {"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"})
            if path == "/metrics.json":
                return self._json(200, app.pipeline.telemetry.snapshot())
            if path == "/api/status":
                return self._json(200, app.status())
            pdf = app.get_pdf()
            if pdf is None:
                return self._json(409, {"error": "no approved report"})
            safe = "".join(c for c in str(app.report_id) if c.isalnum() or c in "-_") or "report"
            return self._send(200, pdf, "application/pdf", {"Content-Disposition": f'attachment; filename="surgidoc-{safe}.pdf"',
                                                            "Content-Security-Policy": "default-src 'none'; sandbox"})

        def do_POST(self):
            path = urllib.parse.urlsplit(self.path).path
            if path not in self.POST_ROUTES:
                return self._json(405, {"error": "method not allowed"}) if path in self.GET_ROUTES else self._json(404, {"error": "not found"})
            if not self._guard(True):
                return
            if path == "/api/upload":
                return self._upload()
            if path == "/api/reset":
                return self._json(*app.reset())
            data = self._json_body()
            if data is None:
                return
            if path == "/api/run":
                return self._json(*app.run(data.get("operator")))
            if path == "/api/approve":
                return self._json(*app.approve(data.get("name")))
            return self._json(*app.reject(data.get("name"), data.get("reason")))

        def _no(self):
            self._send(405, b'{"error": "method not allowed"}', "application/json", {"Allow": "GET, POST"})

        do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _no

        def _upload(self):
            ext = (self._query().get("ext") or [""])[0]
            ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if ext not in ALLOWED_EXT or not (ctype == "application/octet-stream" or ctype.startswith("video/")):
                return self._json(415, {"error": "unsupported file type"})
            if "Transfer-Encoding" in self.headers or self.headers.get("Content-Length") is None:
                return self._json(411, {"error": "Content-Length is required"})
            try:
                length = int(self.headers["Content-Length"])
            except ValueError:
                return self._json(400, {"error": "bad request"})
            if length > app.max_upload:
                return self._json(413, {"error": "file too large"})  # answered before any of the body is read
            if length <= 0:
                return self._json(400, {"error": "empty upload"})
            slot = app.begin_upload(ext)
            if slot is None:
                return self._json(409, {"error": "busy: finish or reset the current review first"})
            part, final = slot
            try:
                head, left = b"", length
                with open(part, "wb") as out:
                    while left:
                        chunk = self.rfile.read(min(65536, left))
                        if not chunk:
                            raise ConnectionError("upload ended early")
                        head = (head + chunk)[:16] if len(head) < 16 else head
                        out.write(chunk)
                        left -= len(chunk)
            except Exception:
                app.abort_upload(part)
                return self._json(400, {"error": "upload failed"})
            if not _looks_like_video(head):
                app.abort_upload(part)
                return self._json(415, {"error": "the file does not look like a video"})
            app.commit_upload(part, final, length)
            return self._json(200, {"ok": True, "size": length})

    return Handler


def serve_ui(app, host="127.0.0.1", port=8765):
    """Start the UI in a daemon thread and return the server (call .shutdown(), .server_close(), app.close())."""
    if host not in LOOPBACK:
        raise ValueError("the UI has no login and may only bind to loopback")
    srv = ThreadingHTTPServer((host, port), make_handler(app))
    srv.app = app
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    return srv


def main(argv=None, build=build_pipeline, block=True):
    ap = argparse.ArgumentParser(prog="python -m surgidoc.webui", description="SurgiDoc local web UI (loopback only).")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--keys", default="keys.txt", help="path to keys.txt")
    args = ap.parse_args(argv)
    if not 0 <= args.port <= 65535:
        ap.error("--port must be between 0 and 65535")
    try:
        keys = load_keys(args.keys)
    except FileNotFoundError:
        ap.error(f"keys file not found: {args.keys}")
    try:
        pipeline = build(keys)
    except MissingKey as e:
        ap.error(f"keys.txt is missing {e}")
    pipeline.setup()  # idempotent: creates the Tiger Cloud tables on first use
    srv = serve_ui(WebApp(pipeline), port=args.port)
    print(f"SurgiDoc UI: http://127.0.0.1:{srv.server_address[1]}   (loopback only; Ctrl-C to stop)")
    if not block:
        return srv
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        srv.server_close()
        srv.app.close()


if __name__ == "__main__":
    main()
