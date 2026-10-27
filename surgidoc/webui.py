"""T16: a small local web UI (standard library only): upload a video, watch the pipeline steps, review the cited draft,
approve or reject as a registered reviewer, download the approved PDF, open the metrics dashboard.

Start:  python -m surgidoc.webui   then open http://127.0.0.1:8765

Single user, one run at a time, no login (the reviewer must be named in SURGIDOC_USERS and the existing role rules apply).
Hardening: loopback bind only; Host and Origin checks (DNS rebinding, cross-site posts); per-session CSRF token on every
API call; nonce-based CSP; upload size/type/magic checks with a random server-side file name; no exception text, path or
report text in errors or audit entries; the page writes data with textContent only.
"""
import argparse
import base64
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


# =============================== the page ================================================================================
CSS = (":root{--bg:#f4f6fb;--card:#fff;--ink:#16202e;--mute:#5d6b7e;--line:#dfe5ee;--accent:#0b6bcb;--accent-ink:#fff;--good:#0a7a2f;--bad:#b00020;"
       "--warn-bg:#fff4d6;--warn-line:#e0b84c;--warn-ink:#5a4300;--chip:#e8f1fc}"
       "@media(prefers-color-scheme:dark){:root{--bg:#0f141b;--card:#18202b;--ink:#e8edf4;--mute:#9aa8ba;--line:#2a3544;--accent:#5aa9ff;--accent-ink:#06121f;"
       "--good:#5fd68a;--bad:#ff8a9b;--warn-bg:#3a3000;--warn-line:#8a7000;--warn-ink:#ffe9a0;--chip:#1f3248}}"
       "*{box-sizing:border-box}body{font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:var(--bg);color:var(--ink)}"
       "header{background:linear-gradient(120deg,#0b3d78,#0b6bcb 60%,#17a2b8);color:#fff;padding:1.4rem 1rem}"
       "header div,main{max-width:900px;margin:0 auto}header h1{margin:0;font-size:1.45rem}header p{margin:.2rem 0 0;opacity:.9}"
       "main{padding:1rem 1rem 3rem}h2{font-size:1.05rem;margin:0 0 .6rem}"
       ".card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:1rem 1.1rem;margin:1rem 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}"
       ".banner{background:var(--warn-bg);border:1px solid var(--warn-line);color:var(--warn-ink);padding:.6rem .9rem;border-radius:10px;margin:1rem 0 0}"
       ".row{display:flex;gap:.6rem;flex-wrap:wrap;align-items:center;margin:.5rem 0}"
       "button,a.btn{padding:.5rem 1rem;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer;font:inherit;text-decoration:none}"
       "button.primary,a.btn.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent);font-weight:600}"
       "button.danger{color:var(--bad);border-color:var(--bad)}button:disabled{opacity:.4;cursor:default}"
       "input[type=text]{padding:.5rem .7rem;border:1px solid var(--line);border-radius:8px;min-width:13rem;background:var(--card);color:var(--ink);font:inherit}"
       "input[type=file]{font:inherit;color:var(--mute)}progress{width:100%;height:.5rem;accent-color:var(--accent)}"
       ".phase{display:inline-block;background:var(--chip);color:var(--accent);padding:.1rem .7rem;border-radius:99px;font-weight:600;text-transform:capitalize}"
       "ul.steps{list-style:none;padding:0;margin:.7rem 0 0;display:flex;gap:.5rem;flex-wrap:wrap}"
       "ul.steps li{padding:.25rem .75rem;border-radius:99px;border:1px solid var(--line);font-size:.85rem}"
       "li.done{background:var(--good);border-color:var(--good);color:#fff}li.todo{color:var(--mute)}.bad{color:var(--bad)}.ok{color:var(--good)}.mute{color:var(--mute)}"
       ".stmt{border:1px solid var(--line);border-radius:10px;padding:.7rem .9rem;margin:.6rem 0}.stmt p{margin:0 0 .5rem}.stmt.removed{opacity:.45}"
       ".stmt.removed textarea{text-decoration:line-through}.stmt .tools{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}"
       "textarea.edit{width:100%;padding:.5rem .7rem;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);font:inherit;resize:vertical}"
       ".badge{display:inline-flex;align-items:center;justify-content:center;min-width:1.5rem;height:1.5rem;padding:0 .35rem;margin-right:.25rem;border-radius:99px;"
       "background:var(--accent);color:var(--accent-ink);font-size:.8rem;font-weight:700;cursor:pointer;border:0}"
       ".ev{display:flex;gap:.6rem;flex-wrap:wrap;margin-top:.4rem}.chip{display:flex;gap:.5rem;align-items:center;background:var(--chip);border-radius:8px;padding:.3rem .6rem;"
       "font-size:.82rem;cursor:pointer;border:1px solid transparent}.chip img{width:72px;height:48px;object-fit:cover;border-radius:5px}"
       ".src{border-top:1px solid var(--line);padding:.8rem .4rem}.src .head{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap}"
       ".src .body{margin:.5rem 0 0 2.1rem}.src img{display:block;max-width:360px;width:100%;border-radius:8px;border:1px solid var(--line);margin-bottom:.4rem}"
       ".src.hit{background:var(--chip);outline:2px solid var(--accent);border-radius:8px}"
       "blockquote{margin:.2rem 0;padding:.4rem .8rem;border-left:3px solid var(--accent);color:var(--mute);font-style:italic}"
       ".toast{position:fixed;right:1rem;bottom:1rem;max-width:22rem;background:var(--accent);color:var(--accent-ink);padding:.8rem 1rem;border-radius:10px;"
       "box-shadow:0 6px 20px rgba(0,0,0,.3);font-weight:600}.toast[hidden]{display:none}"
       ".chip-btn{font-size:.85rem;padding:.3rem .7rem}.answer{margin-top:.8rem;border-top:1px solid var(--line);padding-top:.8rem}.answer p{margin:0 0 .6rem;white-space:pre-wrap}"
       ".kind{font-size:.75rem;letter-spacing:.04em;text-transform:uppercase;color:var(--mute)}")

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
  const img = (src, cls) => { const i = el("img"); i.src = src.image + "&t=" + encodeURIComponent(token); i.alt = "Video frame at " + src.time; i.loading = "lazy"; if (cls) i.className = cls; return i; };
  const label = (src) => src.kind === "frame" ? "Video frame at " + src.time : src.title + (src.page ? ", page " + src.page : ", section " + src.section);
  const jump = (n) => { document.querySelectorAll(".src").forEach((x) => x.classList.toggle("hit", x.dataset.n === String(n)));
    const t = document.querySelector('.src[data-n="' + n + '"]'); if (t) t.scrollIntoView({behavior: "smooth", block: "center"}); };
  const badge = (n) => { const b = el("button", String(n), "badge"); b.type = "button"; b.title = "Show source " + n; b.addEventListener("click", () => jump(n)); return b; };

  function render(s) {
    if (s.notice && s.notice.id !== seenNotice) { if (seenNotice !== 0 || s.notice.id > 0) toast(s.notice.text); seenNotice = s.notice.id; }
    $("phase").textContent = s.phase.replace(/_/g, " ");
    $("status").textContent = s.message;
    const steps = $("steps"); steps.replaceChildren();
    s.steps.forEach((x) => steps.append(el("li", (x.done ? "✓ " : "") + x.label, x.done ? "done" : "todo")));
    const list = $("statements"), sources = $("sources"); list.replaceChildren(); sources.replaceChildren();
    if (s.report) {
      const byN = Object.fromEntries(s.report.sources.map((x) => [x.n, x]));
      const editing = s.phase === "pending_review";
      revision = s.report.revision;
      s.report.statements.forEach((st, idx) => {
        const card = el("div", undefined, "stmt"); card.dataset.i = idx;
        if (editing) {
          const box = el("textarea", undefined, "edit"); box.value = st.text; box.dataset.orig = st.text;
          box.rows = Math.max(2, Math.ceil(st.text.length / 90)); box.setAttribute("aria-label", "Statement " + (idx + 1));
          card.append(box);
        } else card.append(el("p", st.text));
        const tools = el("div", undefined, "tools");
        if (st.guideline) tools.append(el("span", "Guideline reminder", "kind"));
        st.cites.forEach((n) => tools.append(badge(n)));
        if (editing && s.report.statements.length > 1) {
          const rm = el("button", "Remove", "danger"); rm.type = "button";
          rm.addEventListener("click", () => { card.classList.toggle("removed"); rm.textContent = card.classList.contains("removed") ? "Restore" : "Remove"; });
          tools.append(rm);
        }
        card.append(tools);
        const ev = el("div", undefined, "ev");
        st.cites.forEach((n) => {
          const src = byN[n]; if (!src) return;
          const chip = el("div", undefined, "chip"); chip.addEventListener("click", () => jump(n));
          if (src.image) chip.append(img(src));
          chip.append(el("span", n + " · " + label(src)));
          ev.append(chip);
        });
        card.append(ev); list.append(card);
      });
      const cited = s.report.sources.filter((x) => x.kind === "knowledge").length;
      sources.append(el("p", cited ? "" : (s.report.guidelines_retrieved
        ? "No guideline passage supports these statements (" + s.report.guidelines_retrieved + " retrieved, none cited). They rest only on the AI's reading of the video frames."
        : "No guideline passage was found for this video. These statements rest only on the AI's reading of the video frames."), "mute"));
      s.report.sources.forEach((src) => {
        const row = el("div", undefined, "src"); row.dataset.n = src.n;
        const head = el("div", undefined, "head");
        head.append(el("span", String(src.n), "badge"), el("strong", label(src)),
          el("span", src.kind === "frame" ? "Video evidence · AI description by " + src.by : "Guideline reference · retrieved from the knowledge base", "kind"));
        const body = el("div", undefined, "body");
        if (src.kind === "frame") {
          if (src.image) body.append(img(src));
          body.append(el("div", "What the model said about this frame: " + src.text));
        } else {
          body.append(el("blockquote", "“" + src.text + "”"), el("div", "Document: " + src.file, "mute"));
        }
        row.append(head, body); sources.append(row);
      });
    }
    if (s.violations.length) $("status").textContent += " Rules: " + s.violations.join(", ") + ".";
    $("pick").disabled = !idle.includes(s.phase);
    $("run").disabled = s.phase !== "uploaded";
    $("operator").disabled = s.phase !== "uploaded";
    const review = s.phase === "pending_review";
    ["reviewer", "save", "approve", "reason", "reject"].forEach((id) => { $(id).disabled = !review; });
    $("reset").disabled = s.phase === "running";
    const dl = $("download");
    dl.hidden = !s.pdf_ready;
    dl.href = s.pdf_ready ? "/api/report.pdf?t=" + encodeURIComponent(token) : "#";
    $("approved").textContent = s.approver && s.phase === "approved" ? "Approved by " + s.approver + "." : (s.phase === "rejected" ? "Rejected by " + s.approver + "." : "");
  }
  let last = "", revision = 0, seenNotice = 0, toastTimer = null;
  function toast(text) { const t = $("toast"); t.textContent = text; t.hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 8000); }  // re-render only when the status changed, so a surgeon's unsaved typing is never wiped
  async function refresh() { const r = await api("/api/status"); if (r.status === 200) { const k = JSON.stringify(r.data); if (k !== last) { last = k; render(r.data); } } }
  const unsaved = () => [...document.querySelectorAll(".stmt")].some((c) => c.classList.contains("removed") || (c.querySelector("textarea") && c.querySelector("textarea").value !== c.querySelector("textarea").dataset.orig));

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
  document.querySelectorAll(".chip-btn").forEach((b) => b.addEventListener("click", () => { $("question").value = b.dataset.q; $("question").focus(); }));
  $("ask").addEventListener("click", async () => {
    $("ask").disabled = true; $("asking").hidden = false;
    const r = await post("/api/ask", {name: $("askname").value, question: $("question").value});
    $("ask").disabled = false; $("asking").hidden = true;
    if (r.status !== 200) { say(r.data.error || "Could not answer.", true); return; }
    say("");
    $("answer").hidden = false; $("answertext").textContent = r.data.answer;
    const refs = $("answerrefs"); refs.replaceChildren();
    r.data.citations.forEach((c) => {
      const row = el("div", undefined, "src");
      const head = el("div", undefined, "head");
      head.append(el("span", String(c.n), "badge"), el("strong", c.title + (c.page ? ", page " + c.page : "")), el("span", "Guideline reference · " + c.file, "kind"));
      const body = el("div", undefined, "body"); body.append(el("blockquote", "“" + c.text + "”"));
      row.append(head, body); refs.append(row);
    });
  });
  $("run").addEventListener("click", async () => { const r = await post("/api/run", {operator: $("operator").value}); say(r.data.error || "Started.", r.status >= 400); refresh(); });
  $("save").addEventListener("click", async () => {
    const items = [...document.querySelectorAll(".stmt")].filter((c) => !c.classList.contains("removed")).map((c) => ({i: Number(c.dataset.i), text: c.querySelector("textarea").value}));
    const r = await post("/api/edit", {name: $("reviewer").value, revision, statements: items});
    say(r.data.error || "Edits saved. Re-read the report before approving.", r.status >= 400); refresh();
  });
  $("approve").addEventListener("click", async () => {
    if (unsaved()) { say("You have unsaved edits. Save them first, or reload to discard them.", true); return; }
    const r = await post("/api/approve", {name: $("reviewer").value}); say(r.data.error || "Approved.", r.status >= 400); refresh(); });
  $("reject").addEventListener("click", async () => { const r = await post("/api/reject", {name: $("reviewer").value, reason: $("reason").value}); say(r.data.error || (r.data.regenerating ? "Rejected. Writing a new draft from your feedback…" : "Rejected. No regenerations left for this video."), r.status >= 400); refresh(); });
  $("reset").addEventListener("click", async () => { const r = await post("/api/reset"); say(r.data.error || "Ready for a new video.", r.status >= 400); refresh(); });
  refresh(); setInterval(refresh, 800);
})();
"""


def render_page(token, nonce):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SurgiDoc</title><style nonce="{nonce}">{CSS}</style></head>
<body data-token="{html.escape(token)}"><header><div><h1>SurgiDoc</h1><p>Endoscopy documentation assistant: every statement points to the frame or guideline it came from.</p></div></header>
<main>
<section class="card"><h2>Ask the guidelines</h2>
<p class="mute">Answers come only from the EU and German guideline documents in the knowledge base, with the page they came from.</p>
<div class="row"><input id="askname" type="text" placeholder="Your name" autocomplete="off"></div>
<textarea id="question" class="edit" rows="2" maxlength="500" placeholder="For example: What must patients be told before consent for an endoscopy?"></textarea>
<div class="row"><button id="ask" class="primary">Ask</button><span id="asking" class="mute" hidden>Looking it up…</span>
<button class="chip-btn" data-q="What should an upper GI endoscopy report contain?">Report contents</button>
<button class="chip-btn" data-q="What must patients be told before consent for an endoscopic procedure?">Consent</button>
<button class="chip-btn" data-q="Welche Qualitätsanforderungen gelten für die Dokumentation in der Endoskopie?">Auf Deutsch</button></div>
<div id="answer" class="answer" hidden><p id="answertext"></p><div id="answerrefs"></div></div></section>
<section class="card"><h2>1. Upload and run</h2>
<div class="row"><input id="pick" type="file" accept=".mp4,.mov,.avi,.mkv,video/*"></div><progress id="bar" value="0" max="1"></progress>
<div class="row"><input id="operator" type="text" placeholder="Your name (operator)" autocomplete="off"><button id="run" class="primary" disabled>Run pipeline</button>
<button id="reset" disabled>New video</button></div>
<p>State: <span class="phase" id="phase">idle</span> <span id="status" class="mute"></span></p>
<ul class="steps" id="steps"></ul></section>
<section class="card"><h2>2. Review the draft</h2>
<div id="statements"></div>
<div class="row"><input id="reviewer" type="text" placeholder="Reviewer name" autocomplete="off" disabled><button id="save" disabled>Save edits</button><button id="approve" class="primary" disabled>Approve</button>
<input id="reason" type="text" placeholder="Reason (to reject)" autocomplete="off" disabled><button id="reject" class="danger" disabled>Reject</button></div>
<p id="approved"></p></section>
<section class="card"><h2>Sources</h2><div id="sources"><p class="mute">Evidence for each statement appears here once a draft is ready.</p></div></section>
<section class="card"><h2>3. Result</h2>
<div class="row"><a id="download" class="btn primary" href="#" hidden>Download approved PDF</a> <a class="btn" href="/dashboard" target="_blank" rel="noopener">Metrics dashboard</a></div>
<p id="msg" class="ok"></p></section>
<div id="toast" class="toast" role="status" aria-live="polite" hidden></div>
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

        def _json_body(self, limit=MAX_JSON):
            raw = self._body(limit)
            if raw is None:
                return None
            try:
                data = json.loads(raw or b"{}")
            except ValueError:
                return self._fail(400, "invalid JSON")
            return data if isinstance(data, dict) else self._fail(400, "expected a JSON object")

        # --- routes ---
        GET_ROUTES = {"/", "/dashboard", "/metrics.json", "/api/status", "/api/report.pdf", "/api/frame"}
        POST_ROUTES = {"/api/upload", "/api/run", "/api/approve", "/api/reject", "/api/reset", "/api/edit", "/api/ask"}

        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path
            if path not in self.GET_ROUTES:
                return self._json(405, {"error": "method not allowed"}) if path in self.POST_ROUTES else self._json(404, {"error": "not found"})
            if not self._guard(path.startswith("/api/")):
                return
            if path == "/":
                nonce = secrets.token_urlsafe(16)
                csp = (f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; connect-src 'self'; "
                       "img-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                return self._send(200, render_page(app.token, nonce).encode(), "text/html; charset=utf-8", {"Content-Security-Policy": csp})
            if path == "/dashboard":
                return self._send(200, render_dashboard_html(app.pipeline.telemetry.snapshot()).encode(), "text/html; charset=utf-8",
                                  {"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"})
            if path == "/metrics.json":
                return self._json(200, app.pipeline.telemetry.snapshot())
            if path == "/api/status":
                return self._json(200, app.status())
            if path == "/api/frame":
                jpeg = app.get_frame((self._query().get("id") or [""])[0])
                if jpeg is None:
                    return self._json(404, {"error": "not found"})
                return self._send(200, jpeg, "image/jpeg", {"Content-Security-Policy": "default-src 'none'; sandbox"})
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
            data = self._json_body(MAX_EDIT if path == "/api/edit" else MAX_JSON)
            if data is None:
                return
            if path == "/api/ask":
                return self._json(*app.ask(data.get("name"), data.get("question")))
            if path == "/api/edit":
                return self._json(*app.edit(data.get("name"), data.get("statements"), data.get("revision")))
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
