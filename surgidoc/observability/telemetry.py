"""FR-12: OpenTelemetry spans and metrics plus in-process aggregates for the dashboard.

No global state and no OTLP dependency: pass exporters/readers in (ponytail: add an OTLP exporter and a
BatchSpanProcessor when a collector exists). Privacy: prompts, responses and exception messages are never recorded
(exceptions are reduced to their type), and secret-like attribute keys are redacted.
"""
import math
import re
import threading
import time
from collections import Counter, defaultdict, deque
from contextlib import contextmanager

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode

from surgidoc.gateway import GatewayBlocked, RateLimited

_SECRET = re.compile(r"pass|secret|token(?!s)|key|auth|credential", re.I)  # "tokens" is a count, "api_token" a secret
HISTORY = 1000  # latency samples kept per operation


def _attr(key, value):
    if _SECRET.search(key):
        return "***"
    if isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    return text if len(text) <= 200 else text[:200] + "…"


def _pct(sorted_values, q):
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]  # nearest rank


class Stats:
    """Thread-safe aggregates behind the dashboard."""

    def __init__(self):
        self.lock = threading.Lock()
        self.durations = defaultdict(lambda: deque(maxlen=HISTORY))
        self.counts, self.errors = Counter(), Counter()
        self.llm, self.tokens = Counter(), Counter()
        self.reports, self.violations, self.reviews = Counter(), Counter(), Counter()
        self.attempts_sum = self.attempts_n = self.guard_checked = self.guard_blocked = 0
        self.cache_hits = self.cache_misses = 0

    def observe(self, name, seconds, outcome):
        with self.lock:
            self.counts[name] += 1
            self.errors[name] += outcome != "ok"
            self.durations[name].append(seconds)

    def snapshot(self):
        with self.lock:
            ops = {}
            for name, n in self.counts.items():
                d = sorted(self.durations[name])
                ops[name] = {"count": n, "errors": self.errors[name], "mean_ms": 1000 * sum(d) / len(d),
                             "p50_ms": 1000 * _pct(d, 0.5), "p95_ms": 1000 * _pct(d, 0.95)}
            total = sum(self.reports.values())
            cache = self.cache_hits + self.cache_misses
            decided = self.reviews["approved"] + self.reviews["rejected"]
            return {"operations": ops,
                    "usage": {"llm_requests": dict(self.llm), "tokens": dict(self.tokens)},
                    "quality": {"reports": dict(self.reports),
                                "approval_rate": self.reports["approved"] / total if total else None,
                                "mean_attempts": self.attempts_sum / self.attempts_n if self.attempts_n else None,
                                "guardrail_violations": dict(self.violations),
                                "guardrail_block_rate": self.guard_blocked / self.guard_checked if self.guard_checked else None,
                                "cache_hit_rate": self.cache_hits / cache if cache else None,
                                "human_approval_rate": self.reviews["approved"] / decided if decided else None}}


class Op:
    def __init__(self, span):
        self._span, self.outcome = span, "ok"

    def set(self, key, value):
        self._span.set_attribute(key, _attr(key, value))


class Telemetry:
    def __init__(self, span_exporter=None, metric_reader=None, clock=time.perf_counter, service="surgidoc"):
        res = Resource.create({"service.name": service})
        self.tracer_provider = TracerProvider(resource=res)
        if span_exporter:
            self.tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        self.meter_provider = MeterProvider(resource=res, metric_readers=[metric_reader] if metric_reader else [])
        self._tracer, meter = self.tracer_provider.get_tracer("surgidoc"), self.meter_provider.get_meter("surgidoc")
        self._ops = meter.create_counter("surgidoc.operations")
        self._dur = meter.create_histogram("surgidoc.operation.duration", unit="s")
        self._tokens = meter.create_counter("surgidoc.llm.tokens")
        self._reports = meter.create_counter("surgidoc.reports")
        self._attempts = meter.create_histogram("surgidoc.report.attempts")
        self._violations = meter.create_counter("surgidoc.guardrail.violations")
        self._cache = meter.create_counter("surgidoc.cache")
        self._reviews = meter.create_counter("surgidoc.reviews")
        self.clock, self.stats = clock, Stats()

    @contextmanager
    def span(self, name, **attrs):
        t0 = self.clock()
        with self._tracer.start_as_current_span(name, attributes={k: _attr(k, v) for k, v in attrs.items()},
                                                record_exception=False, set_status_on_exception=False) as sp:
            op = Op(sp)
            try:
                yield op
            except BaseException as e:  # message dropped on purpose: it may carry secrets
                op.outcome = "error" if op.outcome == "ok" else op.outcome
                sp.add_event("exception", {"exception.type": type(e).__name__})
                sp.set_status(Status(StatusCode.ERROR, type(e).__name__))
                raise
            finally:
                seconds = self.clock() - t0
                sp.set_attribute("surgidoc.outcome", op.outcome)
                self.stats.observe(name, seconds, op.outcome)
                self._ops.add(1, {"operation": name, "outcome": op.outcome})
                self._dur.record(seconds, {"operation": name})

    def record_llm(self, role, result):
        with self.stats.lock:
            self.stats.llm[f"{role}/{result.model}"] += 1
            self.stats.tokens["prompt"] += result.prompt_tokens
            self.stats.tokens["output"] += result.output_tokens
        self._tokens.add(result.prompt_tokens, {"direction": "prompt", "model": result.model})
        self._tokens.add(result.output_tokens, {"direction": "output", "model": result.model})

    def record_report(self, report, guardrails=None):
        violations = list(getattr(guardrails, "violations", guardrails) or []) if guardrails is not None else None
        with self.stats.lock:
            self.stats.reports[report.status] += 1
            self.stats.attempts_sum += report.attempts
            self.stats.attempts_n += 1
            if violations is not None:
                self.stats.guard_checked += 1
                self.stats.guard_blocked += bool(violations)
                self.stats.violations.update(v.rule for v in violations)
        self._reports.add(1, {"status": report.status})
        self._attempts.record(report.attempts)
        for v in violations or []:
            self._violations.add(1, {"rule": v.rule})

    def record_cache(self, hit):
        with self.stats.lock:
            self.stats.cache_hits += bool(hit)
            self.stats.cache_misses += not hit
        self._cache.add(1, {"result": "hit" if hit else "miss"})

    def record_review(self, outcome):
        if outcome not in ("approved", "rejected"):
            raise ValueError("review outcome must be approved or rejected")
        with self.stats.lock:
            self.stats.reviews[outcome] += 1
        self._reviews.add(1, {"outcome": outcome})

    def snapshot(self):
        return self.stats.snapshot()


class InstrumentedGateway:
    """Wraps a Gateway from the outside; records role, model, tokens, attempts, latency, outcome - never the text."""

    def __init__(self, gateway, telemetry):
        self._gateway, self._t = gateway, telemetry

    def __getattr__(self, name):
        return getattr(self._gateway, name)

    def generate(self, role, prompt, **kw):
        with self._t.span("gateway.generate", role=role) as op:
            try:
                result = self._gateway.generate(role, prompt, **kw)
            except GatewayBlocked:
                op.outcome = "blocked"
                raise
            except RateLimited:
                op.outcome = "rate_limited"
                raise
            for k, v in (("model", result.model), ("prompt_tokens", result.prompt_tokens),
                         ("output_tokens", result.output_tokens), ("attempts", result.attempts)):
                op.set(k, v)
            self._t.record_llm(role, result)
            return result
