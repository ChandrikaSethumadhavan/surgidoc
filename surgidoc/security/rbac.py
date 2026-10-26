"""FR-13, AC-7: role-based access control. Users come from keys.txt: SURGIDOC_USERS=Name:role+role;Name2:role.

There is no login: the registry maps a name to roles (single-tester prototype). Approval and review are separated from
operating and administering on purpose: only `reviewer` may approve, edit or reject.
"""
from surgidoc.approval import ReviewError  # noqa: F401  (re-exported for callers of GuardedReview)

ROLES = {
    "reviewer": {"report:view", "report:edit", "report:approve", "report:reject"},
    "operator": {"pipeline:run", "report:view"},
    "admin": {"audit:view", "users:manage", "report:view"},
}


class PermissionDenied(Exception):
    pass


def _who(name):
    return "anonymous" if not name else str(name)


class Authorizer:
    def __init__(self, users, audit=None):
        self.users, self.audit = {n: set(r) for n, r in users.items()}, audit

    @classmethod
    def from_keys(cls, keys, audit=None):
        users = {}
        for entry in keys.require("SURGIDOC_USERS").split(";"):
            name, sep, roles = entry.partition(":")
            roles = roles.split("+")
            if not sep or not name.strip() or any(r not in ROLES for r in roles):
                raise ValueError("SURGIDOC_USERS entries must look like Name:role+role with known roles")
            users.setdefault(name.strip(), set()).update(roles)
        return cls(users, audit)

    def can(self, name, permission):
        return isinstance(name, str) and any(permission in ROLES[r] for r in self.users.get(name, ()))

    def require(self, name, permission, resource=""):
        if self.can(name, permission):
            return
        if self.audit:  # AC-7: a denial is itself an auditable event
            self.audit.log(_who(name), permission, resource, "denied")
        raise PermissionDenied(f"{_who(name)[:50]!r} may not {permission}")


class GuardedReview:
    """The only way callers should touch a ReviewItem: every action is authorised, bound to the named user, audited."""

    def __init__(self, item, authz, audit):
        self.item, self.authz, self.audit = item, authz, audit

    def _do(self, actor, permission, fn, detail=lambda result: {}):
        self.authz.require(actor, permission, self.item.id)
        try:
            result = fn()
        except Exception as e:  # business-rule failure (e.g. guardrails): audited, never silent
            self.audit.log(actor, permission, self.item.id, "error", {"error": type(e).__name__})
            raise
        self.audit.log(actor, permission, self.item.id, "ok", detail(result))
        return result

    def view(self, actor):
        self.authz.require(actor, "report:view", self.item.id)
        i = self.item
        return {"report_id": i.id, "state": i.state, "revision": i.revision,
                "statements": [{"text": s.text, "sources": list(s.sources)} for s in i.statements],
                "sources": {k: dict(v) for k, v in i.sources.items()}}

    def edit(self, actor, statements):
        return self._do(actor, "report:edit", lambda: self.item.edit(actor, statements),
                        lambda _: {"revision": self.item.revision})

    def approve(self, actor):
        return self._do(actor, "report:approve", lambda: self.item.approve(actor),
                        lambda rec: {"revision": rec["revision"], "content_hash": rec["content_hash"]})

    def reject(self, actor, reason):  # the reason is clinician free text: record only that one was given
        return self._do(actor, "report:reject", lambda: self.item.reject(actor, reason),
                        lambda _: {"reason_length": len(reason)})

    def final_report(self, actor):
        self.authz.require(actor, "report:view", self.item.id)
        return self.item.final_report()
