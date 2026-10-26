"""FR-12: real-time dashboard (usage, latency, quality). Loopback only, read-only, no auth: never expose it.

Every value is HTML-escaped; responses carry a strict CSP. Refreshes every 5 s.
"""
import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOOPBACK = {"127.0.0.1", "::1", "localhost"}
CSS = ("body{font:14px system-ui;margin:2rem;background:#fff;color:#111}h1{font-size:1.3rem}h2{font-size:1rem;margin-top:1.5rem}"
       "table{border-collapse:collapse}td,th{padding:.25rem .8rem;border-bottom:1px solid #ccc;text-align:left}"
       "@media(prefers-color-scheme:dark){body{background:#111;color:#eee}td,th{border-color:#444}}")


def _e(v):
    return html.escape(str(v))


def _pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _num(v, digits=1):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _table(head, rows):
    return ("<table><tr>" + "".join(f"<th>{_e(h)}</th>" for h in head) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in r) + "</tr>" for r in rows) + "</table>")


def render_dashboard_html(s):
    ops, usage, q = s["operations"], s["usage"], s["quality"]
    latency = (_table(["operation", "count", "non-ok", "mean ms", "p50 ms", "p95 ms"],
                      [[n, o["count"], o["errors"], _num(o["mean_ms"]), _num(o["p50_ms"]), _num(o["p95_ms"])]
                       for n, o in sorted(ops.items())]) if ops else "<p>No operations recorded yet.</p>")
    llm = _table(["role/model", "requests"], sorted(usage["llm_requests"].items())) if usage["llm_requests"] else "<p>No LLM calls yet.</p>"
    quality = _table(["measure", "value"], [
        ["reports by status", ", ".join(f"{k}: {v}" for k, v in sorted(q["reports"].items())) or "none"],
        ["judge approval rate", _pct(q["approval_rate"])], ["mean attempts per report", _num(q["mean_attempts"], 2)],
        ["guardrail block rate", _pct(q["guardrail_block_rate"])],
        ["guardrail violations", ", ".join(f"{k}: {v}" for k, v in sorted(q["guardrail_violations"].items())) or "none"],
        ["cache hit rate", _pct(q["cache_hit_rate"])], ["human approval rate", _pct(q["human_approval_rate"])]])
    tokens = usage["tokens"]
    return ("<!doctype html><html lang=en><head><meta charset=utf-8><meta http-equiv=\"refresh\" content=\"5\">"
            f"<meta name=viewport content=\"width=device-width\"><title>SurgiDoc dashboard</title><style>{CSS}</style></head><body>"
            "<h1>SurgiDoc - live dashboard</h1><h2>Usage</h2>" + llm
            + f"<p>Tokens: prompt {_e(tokens.get('prompt', 0))}, output {_e(tokens.get('output', 0))}</p>"
            "<h2>Latency</h2>" + latency + "<h2>Quality</h2>" + quality + "</body></html>")


def serve(telemetry, host="127.0.0.1", port=8765):
    """Start the dashboard in a daemon thread and return the server (call .shutdown() then .server_close())."""
    if host not in LOOPBACK:
        raise ValueError("the dashboard has no authentication and may only bind to loopback")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body, ctype):
            data = body.encode()
            self.send_response(status)
            for k, v in (("Content-Type", ctype), ("Content-Length", str(len(data))), ("Cache-Control", "no-store"),
                         ("X-Content-Type-Options", "nosniff"), ("Referrer-Policy", "no-referrer"),
                         ("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")):
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, render_dashboard_html(telemetry.snapshot()), "text/html; charset=utf-8")
            elif path == "/metrics.json":
                self._send(200, json.dumps(telemetry.snapshot()), "application/json")
            else:
                self._send(404, "not found", "text/plain; charset=utf-8")

        def _no(self):
            self.send_response(405)
            self.send_header("Allow", "GET")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_POST = do_PUT = do_DELETE = do_PATCH = _no

        def log_message(self, *args):  # no request logging: paths and clients are not needed anywhere
            pass

    srv = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
