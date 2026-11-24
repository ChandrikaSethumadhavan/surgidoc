"""T16/T23: a small local web UI (FastAPI + uvicorn): upload a video, watch the pipeline steps, review the cited draft,
approve or reject as a registered reviewer, download the approved PDF, open the metrics dashboard.

Start:  python -m surgidoc.webui   then open http://127.0.0.1:8765

Single user, one run at a time, no login (the reviewer must be named in SURGIDOC_USERS and the existing role rules apply).
Hardening: loopback bind only; Host and Origin checks (DNS rebinding, cross-site posts); per-session CSRF token on every
API call (handed out by GET /api/session); script/style 'self' CSP; upload size/type/magic checks with a random server-side
file name; no exception text, path or report text in errors or audit entries; the React page renders report text as text
(no innerHTML or dangerouslySetInnerHTML: a T24 gate greps for them). The page is the build in frontend/dist (npm run build).
"""
import argparse
import base64
import hmac
import json
import mimetypes
import secrets
import shutil
import socket
import tempfile
import threading
import time
import types
import uuid
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool

from surgidoc.agents import Statement
from surgidoc.approval import NotApproved, ReviewError
from surgidoc.config import MissingKey, load_keys
from surgidoc.observability import render_dashboard_html
from surgidoc.gateway import GatewayBlocked
from surgidoc.pdf import citation_numbers, clock, is_guideline
from surgidoc.pipeline import build_pipeline
from surgidoc.qa import BadQuestion, GuidelineQA
from surgidoc.security import PermissionDenied

LOOPBACK = {"127.0.0.1", "::1", "localhost"}
ALLOWED_EXT = {"mp4", "mov", "avi", "mkv"}
MAX_JSON = 4096
MAX_EDIT = 65536
MAX_REGENERATIONS = 3  # per upload: each one spends free-tier model quota
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


DESCRIBERS = {"gemini-vision": "Gemini vision model", "endovit": "EndoViT model"}


def _public_source(n, sid, src):
    """What the page may show about a source: a number, readable text and (for frames) a link to the picture."""
    if src.get("kind") == "knowledge":
        return {"n": n, "kind": "knowledge", "title": str(src.get("title") or src.get("source")), "section": src.get("index"),
                "file": Path(str(src.get("source", ""))).name, "page": src.get("page"), "text": str(src.get("text", ""))[:600]}
    return {"n": n, "kind": "frame", "time": clock(src.get("t")), "text": str(src.get("text", ""))[:300],
            "by": DESCRIBERS.get(src.get("described_by"), "an AI vision model"),
            "image": f"/api/frame?id={sid}" if src.get("image") else None}


class WebApp:
    def __init__(self, pipeline, max_upload=200 * 1024 * 1024, qa=None):
        self.pipeline, self.max_upload = pipeline, max_upload
        self.qa, self.ask_lock = qa, threading.Lock()
        self.token = secrets.token_urlsafe(32)
        self.tmpdir = Path(tempfile.mkdtemp(prefix="surgidoc-ui-"))  # mode 0700
        self.lock, self.review_lock, self.upload_lock = threading.Lock(), threading.Lock(), threading.Lock()
        self.notice_n = 0
        self._reset()

    def _reset(self):
        self.phase, self.message, self.upload = "idle", "", None  # upload = (path, size)
        self.review = self.pdf = self.approver = self.report_id = None
        self.violations, self.baseline = [], {}
        self.inputs, self.regens, self.notice = None, 0, None  # inputs: what the agents were given, for regeneration

    def close(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # --- upload ---------------------------------------------------------------------------------------------
    def begin_upload(self, ext):
        """-> (part path, final path) or None when busy. The client's file name is never used."""
        with self.lock:
            if self.phase in ("running", "regenerating", "pending_review") or not self.upload_lock.acquire(blocking=False):
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
                self.inputs = res.inputs
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

    def edit(self, name, items, revision):
        def act(review, who):
            cur = review.item.statements
            if revision != review.item.revision:
                raise ReviewError("the draft changed since you opened it; reload the page")
            ok = (isinstance(items, list) and 0 < len(items) <= len(cur) and all(isinstance(x, dict) for x in items)
                  and all(type(x.get("i")) is int and 0 <= x["i"] < len(cur) and isinstance(x.get("text"), str) and x["text"].strip() for x in items)
                  and len({x["i"] for x in items}) == len(items))
            if not ok:
                raise ReviewError("each kept statement needs text, and at least one must remain")
            try:  # a statement keeps the sources it cited; the reviewer changes the words
                review.edit(who, [Statement(x["text"].strip(), list(cur[x["i"]].sources)) for x in items])
            except ReviewError as e:
                rules = ", ".join(sorted({v.rule for v in e.violations}))
                raise ReviewError(f"edit rejected by the safety rules ({rules})" if rules else str(e))

        return self._review_action(name, act, "pending_review")

    def reject(self, name, reason):
        def act(review, who):
            self.pipeline.reject(review, who, reason if isinstance(reason, str) else "")

        with self.lock:  # decided before the action so the phase flips in the same step as the rejection
            again = self.regens < MAX_REGENERATIONS and self.inputs is not None
        code, body = self._review_action(name, act, "regenerating" if again else "rejected")
        if code == 200 and again:
            with self.lock:
                self.regens += 1
                self.baseline = {k: v["count"] for k, v in self.pipeline.telemetry.snapshot()["operations"].items()}
                self.message, review = "Writing a new draft from your feedback.", self.review
            threading.Thread(target=self._regenerate, args=(review, name.strip()), daemon=True).start()
            body["regenerating"] = True
        return code, body

    def _notify(self, text):  # caller holds self.lock
        self.notice_n += 1
        self.notice = {"id": self.notice_n, "text": text}

    def _regenerate(self, review, who):
        try:
            res = self.pipeline.regenerate(review, self.inputs, who)
            with self.lock:
                self.phase, self.message = OUTCOMES[res.status]
                self.violations = list(res.violations)
                if res.status == "pending_review":
                    self.review, self.report_id, self.approver, self.pdf = res.review, res.report_id, None, None
                    self._notify(f"A new draft is ready, written from your feedback (version {self.regens + 1}).")
                else:
                    self._notify("The new draft could not be produced: " + self.message)
        except Exception as e:
            with self.lock:
                self.phase, self.message = "failed", f"Regeneration failed ({type(e).__name__}). Details are in the audit log."
                self._notify(self.message)

    def reset(self):
        with self.lock:
            if self.phase in ("running", "regenerating"):
                return 409, {"error": "a run is in progress"}
            if self.upload:
                self.upload[0].unlink(missing_ok=True)
            self._reset()
        return 200, {"ok": True}

    # --- guideline questions (independent of any video) ---------------------------------------------------------------
    def ask(self, name, question):
        if not isinstance(name, str) or not name.strip():
            return 400, {"error": "your name is required"}
        try:
            self.pipeline.authz.require(name.strip(), "report:view", "guidelines")  # a denial is audited
        except PermissionDenied:
            return 403, {"error": "this user may not ask questions"}
        if not self.ask_lock.acquire(blocking=False):
            return 429, {"error": "another question is being answered; try again in a moment"}
        try:
            self.qa = self.qa or GuidelineQA(self.pipeline.gateway, self.pipeline.retrieve)  # built on first use
            result = self.qa.ask(question)
            self.pipeline.audit.log(name.strip(), "guidelines:ask", "guidelines", "ok",
                                    {"question_length": len(question), "covered": result["covered"]})  # never the text itself
            return 200, result
        except BadQuestion as e:
            return 400, {"error": str(e)}
        except GatewayBlocked:
            return 422, {"error": "the model's safety filter blocked this question"}
        except Exception:
            return 502, {"error": "could not answer right now; try again"}
        finally:
            self.ask_lock.release()

    def get_frame(self, sid):
        with self.lock:
            if self.review is None or self.phase not in ("pending_review", "approved", "rejected"):
                return None
            src = self.review.item.sources.get(sid)
        try:
            raw = base64.b64decode(src["image"], validate=True)
        except (TypeError, KeyError, ValueError):
            return None
        return raw if raw[:2] == b"\xff\xd8" else None  # only ever serve what really is a JPEG

    def get_pdf(self):
        with self.lock:
            return self.pdf if self.phase == "approved" else None

    # --- status ----------------------------------------------------------------------------------------------------
    def status(self):
        ops = self.pipeline.telemetry.snapshot()["operations"]
        with self.lock:
            ran = self.phase not in ("idle", "uploaded")
            # a regeneration reuses the earlier video analysis, so those two steps stay ticked
            steps = [{"id": i, "label": label, "done": bool(self.regens and i in ("frames", "describe"))
                      or (ran and ops.get(op, {}).get("count", 0) > self.baseline.get(op, 0))}
                     for i, label, op in STEPS]
            steps.append({"id": "guardrails", "label": "Safety rules",
                          "done": self.phase in ("pending_review", "approved", "rejected", "guardrails_blocked")})
            steps.append({"id": "review", "label": "Human review", "done": self.phase in ("approved", "rejected")})
            report = None
            if self.review is not None and self.phase in ("pending_review", "approved", "rejected"):
                item = self.review.item
                nums = citation_numbers(item.statements)
                statements = [{"text": s.text, "cites": [nums[c] for c in s.sources if c in nums], "guideline": is_guideline(s)}
                              for s in item.statements]
                report = {"id": self.report_id, "revision": item.revision, "statements": statements,
                          "guidelines_retrieved": sum(v.get("kind") == "knowledge" for v in item.sources.values()),
                          "sources": [_public_source(n, sid, item.sources[sid]) for sid, n in nums.items() if sid in item.sources]}
            return {"phase": self.phase, "message": self.message, "steps": steps, "report": report,
                    "violations": list(self.violations), "approver": self.approver, "pdf_ready": self.pdf is not None and self.phase == "approved",
                    "upload": {"size": self.upload[1]} if self.upload else None, "notice": self.notice,
                    "regenerations_left": MAX_REGENERATIONS - self.regens}


# =============================== http (FastAPI) ============================================================================
GET_ROUTES = {"/", "/dashboard", "/metrics.json", "/api/session", "/api/status", "/api/report.pdf", "/api/frame"}
POST_ROUTES = {"/api/upload", "/api/run", "/api/approve", "/api/reject", "/api/reset", "/api/edit", "/api/ask"}
KNOWN = {("GET", p) for p in GET_ROUTES} | {("POST", p) for p in POST_ROUTES}
HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
           "X-Frame-Options": "DENY", "Cross-Origin-Resource-Policy": "same-origin", "Content-Security-Policy": "default-src 'none'"}
SANDBOX = "default-src 'none'; sandbox"
PAGE_CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; "
            "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"  # the built React app: npm run build in frontend/


def _reply(status, body, ctype, extra=None):
    return Response(content=body, status_code=status, headers=extra, media_type=ctype)


def _json(status, payload):
    return _reply(status, json.dumps(payload).encode(), "application/json")


def _fail(status, message):
    return _json(status, {"error": message})


def _origin_ok(request):
    port = request.scope["server"][1]
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
    origin = request.headers.get("origin")
    return request.headers.get("host", "") in hosts and (origin is None or origin in {f"http://{h}" for h in hosts})


def _length(request):
    """(length, error response): the body must announce its size, and no chunked encoding."""
    h = request.headers
    if "transfer-encoding" in h or h.get("content-length") is None:
        return None, _fail(411, "Content-Length is required")
    try:
        return int(h["content-length"]), None
    except ValueError:
        return None, _fail(400, "bad request")


async def _json_body(request, limit=MAX_JSON):
    n, err = _length(request)
    if err:
        return None, err
    if n < 0 or n > limit:
        return None, _fail(413, "too large")
    try:
        raw = await request.body()
    except Exception:  # the client hung up or stalled mid-request: nothing to answer
        return None, _fail(400, "bad request")
    try:
        data = json.loads(raw or b"{}")
    except ValueError:
        return None, _fail(400, "invalid JSON")
    return (data, None) if isinstance(data, dict) else (None, _fail(400, "expected a JSON object"))


def create_app(app, static_dir=STATIC_DIR):
    """The FastAPI application around a WebApp, serving the built React app from static_dir. No docs/OpenAPI routes and no CORS:
    the UI is same-origin and loopback only."""
    static_dir = Path(static_dir)
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)
    api.state.errors = []  # exceptions that escaped a route: any entry is a bug, even though the client saw a clean 500

    @api.middleware("http")
    async def guard(request, call_next):
        if request.method not in ("GET", "POST"):
            resp = _reply(405, b'{"error": "method not allowed"}', "application/json", {"Allow": "GET, POST"})
        elif (request.method, request.scope["path"]) in KNOWN and not _origin_ok(request):
            resp = _fail(403, "forbidden")
        else:
            try:
                resp = await call_next(request)
            except Exception as e:
                api.state.errors.append(e)
                resp = _fail(500, "internal error")
        for k, v in HEADERS.items():
            resp.headers.setdefault(k, v)
        return resp

    @api.exception_handler(HTTPException)
    def http_error(request, exc):
        return _fail(exc.status_code, str(exc.detail))

    api.add_exception_handler(404, lambda request, exc: _fail(404, "not found"))
    api.add_exception_handler(405, lambda request, exc: _fail(405, "method not allowed"))

    def require_token(request: Request):
        given = request.headers.get("x-csrf-token") or request.query_params.get("t", "")
        if not hmac.compare_digest(given.encode(), app.token.encode()):
            raise HTTPException(403, "forbidden")

    tok = [Depends(require_token)]

    @api.get("/")
    def page():
        index = static_dir / "index.html"
        if not index.is_file():
            return _fail(503, "the page is not built: run npm ci && npm run build in frontend/")
        return _reply(200, index.read_bytes(), "text/html; charset=utf-8", {"Content-Security-Policy": PAGE_CSP})

    @api.get("/assets/{name}")
    def asset(name: str):
        # One path segment only (the Vite bundle is flat), so there is nothing to traverse; no Host check: it is the public bundle.
        f = static_dir / "assets" / name
        if name != Path(name).name or name.startswith(".") or not f.is_file():
            return _fail(404, "not found")
        return _reply(200, f.read_bytes(), mimetypes.guess_type(name)[0] or "application/octet-stream")

    @api.get("/api/session")
    def session():
        # Same exposure as the old data-token attribute: the Host/Origin guard applies and the browser's same-origin policy
        # (plus CORP same-origin) keeps other sites from reading it.
        return _json(200, {"token": app.token})

    @api.get("/dashboard")
    def dashboard():
        return _reply(200, render_dashboard_html(app.pipeline.telemetry.snapshot()).encode(), "text/html; charset=utf-8",
                      {"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"})

    @api.get("/metrics.json")
    def metrics():
        return _json(200, app.pipeline.telemetry.snapshot())

    @api.get("/api/status", dependencies=tok)
    def status():
        return _json(200, app.status())

    @api.get("/api/frame", dependencies=tok)
    def frame(request: Request):
        jpeg = app.get_frame(request.query_params.get("id", ""))
        return _fail(404, "not found") if jpeg is None else _reply(200, jpeg, "image/jpeg", {"Content-Security-Policy": SANDBOX})

    @api.get("/api/report.pdf", dependencies=tok)
    def report_pdf():
        pdf = app.get_pdf()
        if pdf is None:
            return _fail(409, "no approved report")
        safe = "".join(c for c in str(app.report_id) if c.isalnum() or c in "-_") or "report"
        return _reply(200, pdf, "application/pdf", {"Content-Disposition": f'attachment; filename="surgidoc-{safe}.pdf"',
                                                    "Content-Security-Policy": SANDBOX})

    def action(path, call, limit=MAX_JSON):
        async def route(request: Request):
            data, err = await _json_body(request, limit)
            return err or _json(*await run_in_threadpool(call, data))  # WebApp methods take locks and may call a model
        api.post(path, dependencies=tok)(route)

    action("/api/ask", lambda d: app.ask(d.get("name"), d.get("question")))
    action("/api/edit", lambda d: app.edit(d.get("name"), d.get("statements"), d.get("revision")), MAX_EDIT)
    action("/api/run", lambda d: app.run(d.get("operator")))
    action("/api/approve", lambda d: app.approve(d.get("name")))
    action("/api/reject", lambda d: app.reject(d.get("name"), d.get("reason")))

    @api.post("/api/reset", dependencies=tok)
    def reset():
        return _json(*app.reset())

    @api.post("/api/upload", dependencies=tok)
    async def upload(request: Request):
        ext = request.query_params.get("ext", "")
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ext not in ALLOWED_EXT or not (ctype == "application/octet-stream" or ctype.startswith("video/")):
            return _fail(415, "unsupported file type")
        length, err = _length(request)
        if err:
            return err
        if length > app.max_upload:
            return _fail(413, "file too large")  # answered before any of the body is read
        if length <= 0:
            return _fail(400, "empty upload")
        slot = app.begin_upload(ext)
        if slot is None:
            return _fail(409, "busy: finish or reset the current review first")
        part, final = slot
        try:
            head, left = b"", length
            with open(part, "wb") as out:  # ponytail: blocking disk writes on the event loop, fine for one local user
                async for chunk in request.stream():
                    head = (head + chunk)[:16] if len(head) < 16 else head
                    out.write(chunk)
                    left -= len(chunk)
            if left:
                raise ConnectionError("upload ended early")
        except Exception:
            app.abort_upload(part)
            return _fail(400, "upload failed")
        if not _looks_like_video(head):
            app.abort_upload(part)
            return _fail(415, "the file does not look like a video")
        app.commit_upload(part, final, length)
        return _json(200, {"ok": True, "size": length})

    return api


def serve_ui(app, host="127.0.0.1", port=8765, static_dir=STATIC_DIR):
    """Start the UI (uvicorn in a daemon thread) and return a handle: .server_address, .app, .handler_errors,
    .shutdown(), .server_close(); call app.close() afterwards."""
    if host not in LOOPBACK:
        raise ValueError("the UI has no login and may only bind to loopback")
    api = create_app(app, static_dir)
    sock = socket.create_server((host, port))
    server = uvicorn.Server(uvicorn.Config(api, log_level="critical", access_log=False, server_header=False, date_header=False,
                                           http="h11", ws="none", lifespan="off", timeout_graceful_shutdown=2))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and thread.is_alive() and time.time() < deadline:
        time.sleep(0.01)

    def shutdown():
        server.should_exit = True
        thread.join(timeout=10)

    return types.SimpleNamespace(app=app, server_address=sock.getsockname()[:2], handler_errors=api.state.errors,
                                 shutdown=shutdown, server_close=sock.close)


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
