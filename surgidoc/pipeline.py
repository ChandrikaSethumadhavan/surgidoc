"""AC-1: the whole flow, wired from the parts: video -> frames -> findings -> retrieval -> generator/judge ->
guardrails -> named human review -> PDF -> Tiger Cloud, with RBAC, audit log and telemetry around every step.

The pipeline never approves anything: a judge-approved, guardrail-clean draft stops at `pending_review` and returns a
GuardedReview for a named reviewer. Drafts that fail a gate never reach a human. Audit entries hold file names, ids,
rule names and counts only, never report text, paths or exception messages.
"""
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from surgidoc.agents import ReportAgents
from surgidoc.approval import ReviewItem
from surgidoc.cache import Cache, CachedEmbedder, cached_retrieve
from surgidoc.gateway import Gateway
from surgidoc.guardrails import check_report
from surgidoc.observability import InstrumentedGateway, Telemetry
from surgidoc.pdf import render_pdf
from surgidoc.retrieval import GeminiEmbedder, PgStore, connect_tiger
from surgidoc.retrieval import retrieve as hybrid_retrieve
from surgidoc.security import AuditLog, Authorizer, GuardedReview, PgSink
from surgidoc.storage import ReportStore
from surgidoc.vision import describer_from_keys, extract_frames


@dataclass
class RunResult:
    status: str  # pending_review | rejected | blocked | blocked_by_guardrails | no_usable_findings
    report: object = None
    findings: list = field(default_factory=list)
    frames: int = 0
    report_id: str = None
    review: GuardedReview = None
    violations: list = field(default_factory=list)


@dataclass(frozen=True)
class Published:
    report_id: str
    pdf: bytes
    report: dict


def _usable(text):
    return text.strip().lower().rstrip(".") != "unclear"


class Pipeline:
    def __init__(self, *, extract_frames, describer, retrieve, gateway, store, audit, authz, telemetry, cache=None,
                 max_attempts=3, new_id=lambda: uuid.uuid4().hex, components=None):
        self.extract_frames, self.describer, self.retrieve = extract_frames, describer, retrieve
        self.gateway = gateway if isinstance(gateway, InstrumentedGateway) else InstrumentedGateway(gateway, telemetry)
        self.store, self.audit, self.authz, self.telemetry = store, audit, authz, telemetry
        self.cache, self.max_attempts, self.new_id, self.components = cache, max_attempts, new_id, components or [store]

    def setup(self):
        for c in self.components:
            c.setup()

    def _retrieve(self, query):
        with self.telemetry.span("pipeline.retrieve"):
            if self.cache is None:
                return self.retrieve(query)
            called = []
            out = cached_retrieve(self.cache, query, lambda: called.append(1) or self.retrieve(query))
            self.telemetry.record_cache(not called)
            return out

    def run(self, video, actor):
        resource = Path(video).name  # never log the full path
        self.authz.require(actor, "pipeline:run", resource)  # a denial is audited and nothing executes
        t = self.telemetry
        try:
            with t.span("pipeline.run"):
                with t.span("pipeline.frames"):
                    frames = self.extract_frames(video)
                with t.span("pipeline.describe"):
                    findings = self.describer.describe(frames)
                texts = [f.text for f in findings if _usable(f.text)]
                if not texts:
                    return self._done(actor, resource, RunResult("no_usable_findings", findings=findings, frames=len(frames)))
                with t.span("pipeline.agents"):
                    report = ReportAgents(self.gateway, self._retrieve, self.max_attempts).run(texts)
                guard = check_report(report) if report.status == "approved" else None
                t.record_report(report, guard)
                res = RunResult("rejected", report, findings, len(frames))
                if report.status == "blocked":
                    res.status = "blocked"
                elif report.status == "approved" and not guard.passed:
                    res.status, res.violations = "blocked_by_guardrails", sorted({v.rule for v in guard.violations})
                elif report.status == "approved":
                    res.report_id = self.new_id()
                    res.status = "pending_review"
                    res.review = GuardedReview(ReviewItem(report, guard, id=res.report_id), self.authz, self.audit)
                return self._done(actor, resource, res)
        except Exception as e:
            self.audit.log(actor, "pipeline:run", resource, "error", {"error": type(e).__name__})
            raise

    def _done(self, actor, resource, res):
        self.audit.log(actor, "pipeline:run", resource, "ok", {
            "status": res.status, "frames": res.frames, "report_id": res.report_id, "violations": res.violations,
            "attempts": res.report.attempts if res.report else 0})
        return res

    def approve(self, review, actor):
        review.approve(actor)
        self.telemetry.record_review("approved")
        return self.publish(review, actor)

    def publish(self, review, actor):
        """Render and store an approved report. Safe to retry if storage failed after the approval."""
        self.authz.require(actor, "report:approve", review.item.id)
        with self.telemetry.span("pipeline.publish"):
            final = review.final_report(actor)  # NotApproved unless a human approved and nothing changed since
            pdf = render_pdf(final)
            self.store.save_report(final, pdf)
            self.store.record_events(final["report_id"], review.item.events)
            self.audit.log(actor, "report:publish", final["report_id"], "ok", {"content_hash": final["content_hash"]})
            return Published(final["report_id"], pdf, final)

    def reject(self, review, actor, reason):
        review.reject(actor, reason)
        self.telemetry.record_review("rejected")
        self.store.record_events(review.item.id, review.item.events)


def build_pipeline(keys, pg_connect=None, redis_connect=None, max_attempts=3):
    """Real wiring from keys.txt. ponytail: one shared Postgres connection (single-user prototype); use a pool for more."""
    telemetry = Telemetry()
    gateway = InstrumentedGateway(Gateway.from_keys(keys), telemetry)
    conn = connect_tiger(keys, connect=pg_connect)
    audit_sink, chunks, reports = PgSink(conn), PgStore(conn), ReportStore(conn)
    audit = AuditLog(audit_sink)
    authz = Authorizer.from_keys(keys, audit=audit)
    cache = Cache.from_keys(keys, connect=redis_connect)
    embedder = CachedEmbedder(GeminiEmbedder.from_keys(keys), cache)
    return Pipeline(extract_frames=extract_frames, describer=describer_from_keys(keys, gateway),
                    retrieve=lambda q: hybrid_retrieve(q, chunks, embedder), gateway=gateway, store=reports, audit=audit,
                    authz=authz, telemetry=telemetry, cache=cache, max_attempts=max_attempts,
                    components=[chunks, reports, audit_sink])
