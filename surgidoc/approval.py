"""FR-9, AC-3: the human approval gate.

A ReviewItem wraps a judge-approved, guardrail-checked draft. A named human may edit (re-checked by the guardrails),
approve or reject. Approval binds to a sha256 of the content, so anything changed afterwards is detected and no final
report can be produced. The approver types their own name; there is no authentication until RBAC (T10).
"""
import hashlib
import json
import unicodedata
from datetime import datetime, timezone

from surgidoc.agents import Report, Statement
from surgidoc.guardrails import check_report, label_draft


class ReviewError(Exception):
    def __init__(self, message, violations=()):
        super().__init__(message)
        self.violations = list(violations)


class NotApproved(Exception):
    pass


def _utcnow():
    return datetime.now(timezone.utc)


def _as_dicts(statements):
    return [{"text": s["text"], "sources": list(s["sources"])} if isinstance(s, dict)
            else {"text": s.text, "sources": list(s.sources)} for s in statements]


def content_hash(statements, sources):
    body = json.dumps({"statements": _as_dicts(statements), "sources": sources}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _clean(value, what, limit):
    if not isinstance(value, str) or not value.strip():
        raise ReviewError(f"{what} is required")
    value = value.strip()
    if len(value) > limit or any(unicodedata.category(c).startswith("C") for c in value):
        raise ReviewError(f"{what} must be at most {limit} characters with no control characters")
    return value


class ReviewItem:
    def __init__(self, report, result, id, clock=_utcnow):
        label_draft(report, result)  # raises ValueError unless the guardrails passed
        self.id, self.clock, self.sources = id, clock, {k: dict(v) for k, v in report.sources.items()}
        self._attempts = report.attempts
        self._statements = tuple(Statement(s.text, tuple(s.sources)) for s in report.statements)
        self.state, self.revision, self.approval, self.events = "pending_review", 1, None, []
        self._event("created", "system", {})

    @property
    def statements(self):
        return list(self._statements)

    def _event(self, type_, actor, detail):
        self.events.append({"type": type_, "actor": actor, "revision": self.revision, "at": self.clock().isoformat(),
                            "detail": detail})

    def _pending(self):
        if self.state != "pending_review":
            raise ReviewError(f"report is {self.state}, not pending review")

    def edit(self, name, statements):
        name = _clean(name, "editor name", 100)
        self._pending()
        candidate = Report("approved", [Statement(s.text, list(s.sources)) for s in statements], self.sources,
                           self._attempts, [], True)
        result = check_report(candidate)
        if not result.passed:
            raise ReviewError("edit rejected by guardrails", result.violations)
        self._statements = tuple(Statement(s.text, tuple(s.sources)) for s in statements)
        self.revision += 1
        self._event("edited", name, {})

    def approve(self, name):
        name = _clean(name, "approver name", 100)
        self._pending()
        self.approval = {"approver_name": name, "approved_at": self.clock().isoformat(), "revision": self.revision,
                         "content_hash": content_hash(self._statements, self.sources)}
        self.state = "approved"
        self._event("approved", name, {"content_hash": self.approval["content_hash"]})
        return dict(self.approval)

    def reject(self, name, reason):
        name, reason = _clean(name, "reviewer name", 100), _clean(reason, "reason", 500)
        self._pending()
        self.state = "rejected"
        self._event("rejected", name, {"reason": reason})

    def final_report(self):
        if self.state != "approved":
            raise NotApproved(f"report is {self.state}: only a human-approved report can be finalised")
        if content_hash(self._statements, self.sources) != self.approval["content_hash"]:
            raise NotApproved("report content changed after approval")
        return {"status": "APPROVED", "report_id": self.id, "revision": self.approval["revision"],
                "approver_name": self.approval["approver_name"], "approved_at": self.approval["approved_at"],
                "content_hash": self.approval["content_hash"], "statements": _as_dicts(self._statements),
                "sources": {k: dict(v) for k, v in self.sources.items()}}
