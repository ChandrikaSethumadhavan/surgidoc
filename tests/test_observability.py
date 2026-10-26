import http.client
import json
import threading

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from surgidoc.agents import Report, Statement
from surgidoc.gateway import GatewayBlocked, GatewayError, RateLimited, Result
from surgidoc.guardrails import Violation
from surgidoc.ingest import Chunk
from surgidoc.observability import (InstrumentedGateway, ReportCase, RetrievalCase, Telemetry, check_thresholds,
                                    evaluate_reports, evaluate_retrieval, render_dashboard_html, serve)


class Clock:
    def __init__(self, step=0.1):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


def telemetry(step=0.1):
    spans, reader = InMemorySpanExporter(), InMemoryMetricReader()
    return Telemetry(span_exporter=spans, metric_reader=reader, clock=Clock(step)), spans, reader


def metric_points(reader, name):
    out = []
    data = reader.get_metrics_data()
    for rm in (data.resource_metrics if data else []):
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name == name:
                    out += list(m.data.data_points)
    return out


# =============================== spans =============================================
def test_span_is_exported_with_name_and_attributes():
    t, spans, _ = telemetry()
    with t.span("retrieval", k=5, note="x"):
        pass
    (s,) = spans.get_finished_spans()
    assert s.name == "retrieval" and s.attributes["k"] == 5 and s.attributes["surgidoc.outcome"] == "ok"


def test_nested_spans_share_a_trace_and_link_parent():
    t, spans, _ = telemetry()
    with t.span("outer"):
        with t.span("inner"):
            pass
    inner, outer = spans.get_finished_spans()
    assert inner.parent.span_id == outer.context.span_id and inner.context.trace_id == outer.context.trace_id


def test_exception_marks_span_error_reraises_and_hides_the_message():
    t, spans, _ = telemetry()
    with pytest.raises(RuntimeError):
        with t.span("boom"):
            raise RuntimeError("connection postgres://u:SECRETPW@h failed")
    (s,) = spans.get_finished_spans()
    assert s.status.status_code.name == "ERROR" and s.attributes["surgidoc.outcome"] == "error"
    blob = json.dumps([dict(e.attributes) for e in s.events]) + str(s.status.description)
    assert "SECRETPW" not in blob and "RuntimeError" in blob


def test_attributes_are_sanitised():
    t, spans, _ = telemetry()
    with t.span("x", api_key="AIzaSECRET", Password="p", long="y" * 1000, obj=object(), ok=True):
        pass
    a = spans.get_finished_spans()[0].attributes
    assert a["api_key"] == a["Password"] == "***" and len(a["long"]) <= 201 and a["ok"] is True
    assert "object object" in a["obj"] and "AIzaSECRET" not in json.dumps(dict(a))


def test_op_can_set_attributes_and_outcome():
    t, spans, _ = telemetry()
    with t.span("x") as op:
        op.set("model", "m1")
        op.outcome = "blocked"
    s = spans.get_finished_spans()[0]
    assert s.attributes["model"] == "m1" and s.attributes["surgidoc.outcome"] == "blocked"


# =============================== metrics ===============================================
def test_operation_counter_and_duration_histogram_are_exported():
    t, _, reader = telemetry(step=0.25)
    with t.span("op"):
        pass
    with pytest.raises(ValueError):
        with t.span("op"):
            raise ValueError("x")
    pts = {tuple(sorted(p.attributes.items())): p.value for p in metric_points(reader, "surgidoc.operations")}
    assert pts[(("operation", "op"), ("outcome", "ok"))] == 1 and pts[(("operation", "op"), ("outcome", "error"))] == 1
    h = metric_points(reader, "surgidoc.operation.duration")
    assert sum(p.count for p in h) == 2 and all(p.sum > 0 for p in h)


def test_snapshot_operation_latency_and_errors():
    t, _, _ = telemetry(step=0.5)  # every span measures 0.5 s
    for _ in range(19):
        with t.span("gen"):
            pass
    with pytest.raises(ValueError):
        with t.span("gen"):
            raise ValueError
    g = t.snapshot()["operations"]["gen"]
    assert g["count"] == 20 and g["errors"] == 1 and g["mean_ms"] == pytest.approx(500) and g["p95_ms"] == pytest.approx(500)


def test_percentiles_use_nearest_rank():
    t = Telemetry(clock=Clock())
    for ms in range(1, 101):
        t.stats.observe("op", ms / 1000, "ok")
    g = t.snapshot()["operations"]["op"]
    assert g["p50_ms"] == pytest.approx(50) and g["p95_ms"] == pytest.approx(95)


def test_telemetry_works_without_exporters():
    t = Telemetry()
    with t.span("x"):
        pass
    assert t.snapshot()["operations"]["x"]["count"] == 1


def test_stats_are_thread_safe():
    t = Telemetry(clock=Clock(0.001))
    ts = [threading.Thread(target=lambda: [t.stats.observe("op", 0.01, "ok") for _ in range(100)]) for _ in range(8)]
    [x.start() for x in ts]
    [x.join() for x in ts]
    assert t.snapshot()["operations"]["op"]["count"] == 800


def test_duration_history_is_bounded():
    t = Telemetry()
    for _ in range(5000):
        t.stats.observe("op", 0.01, "ok")
    assert t.snapshot()["operations"]["op"]["count"] == 5000 and len(t.stats.durations["op"]) <= 1000


# =============================== quality metrics ============================================
def report(status="approved", attempts=1):
    return Report(status, [Statement("A polyp.", ["F0"])], {"F0": {"kind": "frame", "text": "polyp"}}, attempts, [], True)


def test_record_report_updates_quality_snapshot():
    t, _, reader = telemetry()
    t.record_report(report("approved", 1))
    t.record_report(report("approved", 3))
    t.record_report(report("rejected", 3), guardrails=[Violation("diagnosis", 0, "d"), Violation("injection", 0, "i")])
    q = t.snapshot()["quality"]
    assert q["reports"] == {"approved": 2, "rejected": 1}
    assert q["approval_rate"] == pytest.approx(2 / 3) and q["mean_attempts"] == pytest.approx(7 / 3)
    assert q["guardrail_violations"] == {"diagnosis": 1, "injection": 1}
    assert sum(p.value for p in metric_points(reader, "surgidoc.reports")) == 3


def test_guardrail_block_rate_and_cache_and_human_review_rates():
    t = Telemetry()
    t.record_report(report(), guardrails=[])
    t.record_report(report(), guardrails=[Violation("diagnosis", 0, "d")])
    for hit in (True, True, False, True):
        t.record_cache(hit)
    t.record_review("approved")
    t.record_review("rejected")
    t.record_review("approved")
    q = t.snapshot()["quality"]
    assert q["guardrail_block_rate"] == pytest.approx(0.5) and q["cache_hit_rate"] == pytest.approx(0.75)
    assert q["human_approval_rate"] == pytest.approx(2 / 3)


def test_rates_are_none_when_there_is_no_data():
    q = Telemetry().snapshot()["quality"]
    assert q["approval_rate"] is None and q["cache_hit_rate"] is None and q["human_approval_rate"] is None


def test_record_review_rejects_unknown_outcomes():
    with pytest.raises(ValueError):
        Telemetry().record_review("maybe")


# =============================== instrumented gateway ==========================================
class FakeGateway:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.stats = result, exc, {"m": 1}

    def generate(self, role, prompt, **kw):
        if self.exc:
            raise self.exc
        return self.result


R = Result("PATIENT-FREE-TEXT answer", "gemini-3.6-flash", 120, 45, 2)


def test_gateway_wrapper_returns_the_result_and_records_usage():
    t, spans, reader = telemetry()
    gw = InstrumentedGateway(FakeGateway(R), t)
    assert gw.generate("generator", "SECRET PROMPT TEXT") is R
    (s,) = spans.get_finished_spans()
    assert s.name == "gateway.generate" and s.attributes["role"] == "generator"
    assert s.attributes["model"] == "gemini-3.6-flash" and s.attributes["prompt_tokens"] == 120 and s.attributes["attempts"] == 2
    snap = t.snapshot()
    assert snap["usage"]["llm_requests"] == {"generator/gemini-3.6-flash": 1} and snap["usage"]["tokens"] == {"prompt": 120, "output": 45}
    assert sum(p.value for p in metric_points(reader, "surgidoc.llm.tokens")) == 165


def test_gateway_wrapper_never_records_prompt_or_response_text():
    t, spans, _ = telemetry()
    InstrumentedGateway(FakeGateway(R), t).generate("judge", "SECRET PROMPT TEXT")
    blob = json.dumps(dict(spans.get_finished_spans()[0].attributes)) + json.dumps(t.snapshot())
    assert "SECRET PROMPT TEXT" not in blob and "PATIENT-FREE-TEXT" not in blob


@pytest.mark.parametrize("exc,outcome", [(GatewayBlocked("b"), "blocked"), (RateLimited("r"), "rate_limited"),
                                         (GatewayError("e"), "error")])
def test_gateway_failures_are_classified_and_reraised(exc, outcome):
    t, spans, _ = telemetry()
    with pytest.raises(type(exc)):
        InstrumentedGateway(FakeGateway(exc=exc), t).generate("generator", "p")
    assert spans.get_finished_spans()[0].attributes["surgidoc.outcome"] == outcome
    assert t.snapshot()["operations"]["gateway.generate"]["errors"] == 1


def test_gateway_wrapper_delegates_other_attributes():
    assert InstrumentedGateway(FakeGateway(R), Telemetry()).stats == {"m": 1}


# =============================== dashboard =====================================================
def filled_telemetry():
    t = Telemetry(clock=Clock(0.2))
    InstrumentedGateway(FakeGateway(R), t).generate("generator", "p")
    t.record_report(report("approved", 2))
    t.record_cache(True)
    with t.span("<script>alert(1)</script>"):
        pass
    return t


def test_dashboard_html_shows_numbers_and_escapes_everything():
    html = render_dashboard_html(filled_telemetry().snapshot())
    assert "gateway.generate" in html and "gemini-3.6-flash" in html and "100.0%" in html
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert 'http-equiv="refresh"' in html


def test_dashboard_handles_an_empty_snapshot():
    html = render_dashboard_html(Telemetry().snapshot())
    assert "No operations recorded yet" in html and "n/a" in html


@pytest.fixture
def server():
    t = filled_telemetry()
    srv = serve(t, port=0)
    yield srv, t
    srv.shutdown()
    srv.server_close()


def get(srv, path, method="GET"):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
    c.request(method, path)
    r = c.getresponse()
    return r.status, dict(r.getheaders()), r.read().decode()


def test_dashboard_server_serves_html_and_json_with_safe_headers(server):
    srv, t = server
    status, headers, body = get(srv, "/")
    assert status == 200 and "text/html" in headers["Content-Type"] and "gateway.generate" in body
    assert "default-src 'none'" in headers["Content-Security-Policy"] and headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Cache-Control"] == "no-store"
    status, headers, body = get(srv, "/metrics.json")
    assert status == 200 and "application/json" in headers["Content-Type"]
    assert json.loads(body)["operations"]["gateway.generate"]["count"] == 1


def test_dashboard_server_rejects_other_paths_and_methods(server):
    srv, _ = server
    assert get(srv, "/etc/passwd")[0] == 404
    assert get(srv, "/", "POST")[0] == 405 and get(srv, "/", "DELETE")[0] == 405


def test_dashboard_server_binds_loopback_only():
    t = Telemetry()
    srv = serve(t, port=0)
    try:
        assert srv.server_address[0] == "127.0.0.1"
    finally:
        srv.shutdown()
        srv.server_close()
    with pytest.raises(ValueError):
        serve(t, host="0.0.0.0", port=0)


# =============================== evaluation: retrieval ===========================================
def chunk(i, source="s"):
    return Chunk(f"text {i}", source, i)


def test_retrieval_recall_and_mrr():
    cases = [RetrievalCase("q1", {"s#1"}), RetrievalCase("q2", {"s#9", "s#2"}), RetrievalCase("q3", {"s#7"})]
    results = {"q1": [chunk(0), chunk(1)], "q2": [chunk(2), chunk(5)], "q3": [chunk(0), chunk(3)]}
    m = evaluate_retrieval(lambda q: results[q], cases, k=5)
    assert m["recall@5"] == pytest.approx((1 + 0.5 + 0) / 3) and m["mrr"] == pytest.approx((1 / 2 + 1 / 1 + 0) / 3)
    assert m["n"] == 3


def test_retrieval_only_counts_the_top_k():
    m = evaluate_retrieval(lambda q: [chunk(i) for i in range(10)], [RetrievalCase("q", {"s#7"})], k=5)
    assert m["recall@5"] == 0 and m["mrr"] == 0
    m = evaluate_retrieval(lambda q: [chunk(i) for i in range(10)], [RetrievalCase("q", {"s#7"})], k=8)
    assert m["recall@8"] == 1 and m["mrr"] == pytest.approx(1 / 8)


def test_retrieval_eval_needs_cases_with_relevant_ids():
    with pytest.raises(ValueError):
        evaluate_retrieval(lambda q: [], [], k=5)
    with pytest.raises(ValueError):
        evaluate_retrieval(lambda q: [], [RetrievalCase("q", set())], k=5)


# =============================== evaluation: reports =============================================
SRC = {"F0": {"kind": "frame", "text": "Polyp about 6 mm in the sigmoid colon."},
       "K1": {"kind": "knowledge", "source": "e", "index": 1, "text": "Snare for polyps under 10 mm."}}


def rep(*statements, status="approved", attempts=1):
    return Report(status, [Statement(t, c) for t, c in statements], SRC, attempts, [], True)


class FakeAgents:
    def __init__(self, reports):
        self.reports = list(reports)

    def run(self, findings):
        r = self.reports.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_report_metrics_measure_coverage_citations_and_numeric_support():
    good = rep(("A 6 mm polyp in the sigmoid colon.", ["F0"]), ("Snare is suitable under 10 mm.", ["K1"]))
    bad = rep(("An 8 mm polyp was seen.", ["F0"]), ("Unsupported claim.", ["K9"]))
    out = evaluate_reports(FakeAgents([good, bad]), [ReportCase(["f"], ["polyp", "sigmoid"]), ReportCase(["f"], ["polyp", "sigmoid"])])
    g, b = out["cases"]
    assert (g["coverage"], g["citation_validity"], g["numeric_support"]) == (1.0, 1.0, 1.0)
    assert b["coverage"] == 0.5 and b["citation_validity"] == 0.5 and b["numeric_support"] == 0.5  # 8 mm not in source
    m = out["metrics"]
    assert m["n"] == 2 and m["coverage"] == pytest.approx(0.75) and m["approval_rate"] == 1.0 and m["error_rate"] == 0


def test_statements_without_numbers_are_vacuously_numerically_supported():
    r = rep(("Mucosa appeared normal.", ["F0"]))
    assert evaluate_reports(FakeAgents([r]), [ReportCase(["f"], [])])["cases"][0]["numeric_support"] == 1.0


def test_rejected_reports_lower_approval_rate_and_attempts_are_averaged():
    reps = [rep(("Polyp.", ["F0"]), attempts=1), rep(("Polyp.", ["F0"]), status="rejected", attempts=3)]
    m = evaluate_reports(FakeAgents(reps), [ReportCase(["f"], ["polyp"])] * 2)["metrics"]
    assert m["approval_rate"] == 0.5 and m["mean_attempts"] == 2.0


def test_agent_errors_are_counted_not_raised():
    out = evaluate_reports(FakeAgents([RuntimeError("down"), rep(("Polyp.", ["F0"]))]), [ReportCase(["f"], ["polyp"])] * 2)
    assert out["cases"][0]["error"] == "RuntimeError" and out["metrics"]["error_rate"] == 0.5
    assert out["cases"][0]["coverage"] == 0 and out["metrics"]["coverage"] == 0.5


def test_empty_statements_score_zero():
    out = evaluate_reports(FakeAgents([rep()]), [ReportCase(["f"], ["polyp"])])["cases"][0]
    assert out["coverage"] == 0 and out["citation_validity"] == 0


def test_report_eval_requires_cases():
    with pytest.raises(ValueError):
        evaluate_reports(FakeAgents([]), [])


# =============================== thresholds (CI gate) =============================================
def test_check_thresholds_flags_regressions_in_both_directions():
    metrics = {"coverage": 0.7, "error_rate": 0.1, "recall@5": 0.9}
    fails = check_thresholds(metrics, {"coverage": (">=", 0.8), "error_rate": ("<=", 0.0), "recall@5": (">=", 0.8)})
    assert len(fails) == 2 and any("coverage" in f for f in fails) and any("error_rate" in f for f in fails)
    assert check_thresholds(metrics, {"coverage": (">=", 0.7)}) == []


def test_check_thresholds_rejects_unknown_metrics_and_operators():
    with pytest.raises(ValueError):
        check_thresholds({"a": 1}, {"zzz": (">=", 0.5)})
    with pytest.raises(ValueError):
        check_thresholds({"a": 1}, {"a": ("!=", 0.5)})
