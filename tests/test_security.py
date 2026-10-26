import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from surgidoc.agents import Report, Statement
from surgidoc.approval import ReviewItem
from surgidoc.config import MissingKey, load_keys
from surgidoc.guardrails import check_report
from surgidoc.security import (AuditLog, Authorizer, Cipher, ComplianceError, DecryptionError, GuardedReview,
                               MemorySink, PermissionDenied, PgSink, assert_compliant, check_compliance,
                               generate_key, verify_chain)


class Clock:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return datetime(2026, 10, 26, 12, 0, self.n % 60, tzinfo=timezone.utc)


def keys(tmp_path, text):
    p = tmp_path / "keys.txt"
    p.write_text(text)
    return load_keys(p)


USERS = {"Chandrika": {"reviewer", "admin", "operator"}, "Olive": {"operator"}, "Ada": {"admin"}, "Rex": {"reviewer"}}


def audit():
    return AuditLog(MemorySink(), clock=Clock())


# =============================== RBAC =============================================
def test_reviewer_can_review_but_operator_and_admin_cannot_approve():
    a = Authorizer(USERS)
    assert all(a.can("Rex", p) for p in ["report:view", "report:edit", "report:approve", "report:reject"])
    assert not a.can("Olive", "report:approve") and not a.can("Ada", "report:approve")  # separation of duties
    assert a.can("Olive", "pipeline:run") and a.can("Ada", "audit:view") and not a.can("Rex", "audit:view")


def test_user_with_several_roles_gets_the_union():
    a = Authorizer(USERS)
    assert a.can("Chandrika", "report:approve") and a.can("Chandrika", "audit:view") and a.can("Chandrika", "pipeline:run")


@pytest.mark.parametrize("who", ["", None, "nobody", "chandrika", "Chandrika ", "Chаndrika"])  # case, space, cyrillic a
def test_unknown_or_lookalike_users_are_denied(who):
    assert not Authorizer(USERS).can(who, "report:view")


def test_unknown_permission_is_denied():
    assert not Authorizer(USERS).can("Rex", "root:everything")


def test_require_denial_raises_and_is_audited():  # AC-7
    log = audit()
    a = Authorizer(USERS, audit=log)
    with pytest.raises(PermissionDenied):
        a.require("Olive", "report:approve", resource="rep1")
    e = log.entries()[-1]
    assert (e["actor"], e["action"], e["outcome"], e["resource"]) == ("Olive", "report:approve", "denied", "rep1")
    assert log.verify().ok


def test_require_allowed_does_not_raise():
    Authorizer(USERS, audit=audit()).require("Rex", "report:approve", resource="rep1")


def test_denial_for_unknown_user_is_audited_without_crashing_on_odd_names():
    log = audit()
    with pytest.raises(PermissionDenied):
        Authorizer(USERS, audit=log).require("x" * 5000 + "\n\x00", "report:view")
    assert len(log.entries()[-1]["actor"]) <= 200


def test_from_keys_parses_users_and_roles(tmp_path):
    a = Authorizer.from_keys(keys(tmp_path, "SURGIDOC_USERS=Chandrika:reviewer+admin;Olive:operator\n"))
    assert a.can("Chandrika", "audit:view") and a.can("Olive", "pipeline:run") and not a.can("Olive", "report:approve")


@pytest.mark.parametrize("bad", ["Bob:superuser", "Bob", ":reviewer", "Bob:", "Bob:reviewer;;", "A:reviewer;A:bogus"])
def test_from_keys_rejects_malformed_registry(tmp_path, bad):
    with pytest.raises(ValueError):
        Authorizer.from_keys(keys(tmp_path, f"SURGIDOC_USERS={bad}\n"))


def test_from_keys_requires_users(tmp_path):
    with pytest.raises(MissingKey, match="SURGIDOC_USERS"):
        Authorizer.from_keys(keys(tmp_path, "X=1\n"))


# =============================== guarded review ====================================
SOURCES = {"F0": {"kind": "frame", "text": "Small sessile polyp."}}


def guarded(authz=None, log=None):
    r = Report("approved", [Statement("A small sessile polyp was observed.", ["F0"])], SOURCES, 1, [], True)
    item = ReviewItem(r, check_report(r), id="rep1", clock=Clock())
    log = log or audit()
    return GuardedReview(item, authz or Authorizer(USERS, audit=log), log), item, log


def test_reviewer_approval_binds_the_authenticated_name_and_is_audited():
    g, item, log = guarded()
    g.approve("Rex")
    assert item.approval["approver_name"] == "Rex" and item.state == "approved"
    assert log.entries()[-1]["action"] == "report:approve" and log.entries()[-1]["outcome"] == "ok"
    assert g.final_report("Rex")["approver_name"] == "Rex"


def test_non_reviewers_cannot_approve_edit_or_reject_and_item_is_untouched():
    g, item, log = guarded()
    for call in (lambda: g.approve("Olive"), lambda: g.reject("Ada", "no"),
                 lambda: g.edit("Olive", [Statement("A polyp.", ["F0"])]), lambda: g.approve("stranger")):
        with pytest.raises(PermissionDenied):
            call()
    assert item.state == "pending_review" and item.revision == 1
    assert [e["outcome"] for e in log.entries()].count("denied") == 4


def test_view_is_allowed_for_operators_but_not_strangers():
    g, _, _ = guarded()
    assert g.view("Olive")["statements"][0]["text"].startswith("A small")
    with pytest.raises(PermissionDenied):
        g.view("stranger")


def test_edit_and_reject_by_reviewer_are_audited_with_revision():
    g, item, log = guarded()
    g.edit("Rex", [Statement("A 6 mm sessile polyp was observed.", ["F0"])])
    assert item.revision == 2
    g2, item2, log2 = guarded()
    g2.reject("Rex", "Inaccurate.")
    assert item2.state == "rejected"
    assert [e["action"] for e in log.entries()] == ["report:edit"]
    assert log.entries()[0]["detail"]["revision"] == 2


def test_guard_does_not_audit_secrets_or_report_text():
    g, _, log = guarded()
    g.edit("Rex", [Statement("A 6 mm sessile polyp was observed.", ["F0"])])
    assert "sessile" not in json.dumps(log.entries())


def test_final_report_requires_view_permission_and_approval():
    g, _, _ = guarded()
    with pytest.raises(PermissionDenied):
        g.final_report("stranger")


# =============================== audit log ==========================================
def filled(n=5):
    log = audit()
    for i in range(n):
        log.log("Rex", "report:view", resource=f"r{i}", detail={"i": i})
    return log


def test_chain_links_every_entry_to_the_previous():
    log = filled()
    es = log.entries()
    assert [e["seq"] for e in es] == [1, 2, 3, 4, 5]
    assert es[0]["prev_hash"] == "0" * 64 and all(es[i]["prev_hash"] == es[i - 1]["hash"] for i in range(1, 5))
    assert log.verify().ok and log.head() == (5, es[-1]["hash"])


def test_empty_log_verifies_and_has_genesis_head():
    log = audit()
    assert log.verify().ok and log.head() == (0, "0" * 64)


def test_modified_entry_is_detected():
    log = filled()
    log.sink.entries[2]["actor"] = "someone else"
    r = log.verify()
    assert not r.ok and r.index == 2


def test_deleted_entry_is_detected():
    log = filled()
    del log.sink.entries[1]
    assert not log.verify().ok


def test_reordered_entries_are_detected():
    log = filled()
    log.sink.entries[1], log.sink.entries[2] = log.sink.entries[2], log.sink.entries[1]
    assert not log.verify().ok


def test_tail_truncation_is_detected_only_with_an_anchored_head():
    log = filled()
    anchor = log.head()
    del log.sink.entries[-2:]
    assert log.verify().ok  # a chain alone cannot see a cut tail
    assert not log.verify(expected_head=anchor).ok


def test_verify_chain_works_on_plain_lists():
    es = filled(3).entries()
    assert verify_chain(es).ok and not verify_chain(es[1:]).ok


def test_secret_like_detail_keys_are_redacted_and_long_values_truncated():
    log = audit()
    log.log("Rex", "x", detail={"api_key": "AIzaSECRET", "password": "p", "Token": "t", "note": "y" * 2000, "ok": 1})
    d = log.entries()[0]["detail"]
    assert d["api_key"] == d["password"] == d["Token"] == "***" and len(d["note"]) <= 520 and d["ok"] == 1
    assert "AIzaSECRET" not in json.dumps(log.entries())


def test_concurrent_appends_keep_a_valid_chain():
    log = audit()
    ts = [threading.Thread(target=lambda i=i: log.log("Rex", "a", resource=str(i))) for i in range(50)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(log.entries()) == 50 and log.verify().ok and len({e["seq"] for e in log.entries()}) == 50


def test_actor_and_action_are_required():
    with pytest.raises(ValueError):
        audit().log("", "x")
    with pytest.raises(ValueError):
        audit().log("Rex", "")


# --- Postgres sink (SQL contract) ----------------------------------------------------------
class FakeCur:
    def __init__(self, rows=()):
        self.log, self.rows = [], list(rows)

    def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class FakeConn:
    def __init__(self, rows=()):
        self.cur, self.commits, self.rollbacks = FakeCur(rows), 0, 0

    def cursor(self):
        return self.cur

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def test_pg_sink_setup_forbids_update_delete_truncate():
    conn = FakeConn()
    PgSink(conn, prefix="t1").setup()
    sql = " | ".join(s for s, _ in conn.cur.log)
    for needle in ["CREATE TABLE IF NOT EXISTS t1_audit", "detail text", "RAISE EXCEPTION", "BEFORE UPDATE OR DELETE ON t1_audit",
                   "BEFORE TRUNCATE ON t1_audit"]:
        assert needle in sql
    assert conn.commits == 1


@pytest.mark.parametrize("bad", ["x; DROP TABLE y", "Audit", "", "a-b", "a" * 40])
def test_pg_sink_prefix_is_whitelisted(bad):
    with pytest.raises(ValueError):
        PgSink(FakeConn(), prefix=bad)


def test_pg_sink_append_locks_reads_head_then_inserts_parameterized():
    conn = FakeConn(rows=[(3, "h" * 64)])
    log = AuditLog(PgSink(conn, prefix="t1"), clock=Clock())
    e = log.log("Rex", "report:view", resource="r'; DROP TABLE x;--")
    sqls = [s for s, _ in conn.cur.log]
    assert sqls[0].startswith("SELECT pg_advisory_xact_lock") and "ORDER BY seq DESC LIMIT 1" in sqls[1]
    insert, params = conn.cur.log[-1]
    assert insert.startswith("INSERT INTO t1_audit") and "DROP" not in insert and "r'; DROP TABLE x;--" in params
    assert e["seq"] == 4 and e["prev_hash"] == "h" * 64 and conn.commits == 1


def test_pg_sink_rolls_back_on_failure():
    class Boom(FakeConn):
        def cursor(self):
            c = super().cursor()
            orig = c.execute

            def execute(sql, params=None):
                if sql.startswith("INSERT"):
                    raise RuntimeError("db down")
                orig(sql, params)

            c.execute = execute
            return c

    conn = Boom()
    with pytest.raises(RuntimeError):
        AuditLog(PgSink(conn, prefix="t1"), clock=Clock()).log("Rex", "x")
    assert conn.rollbacks == 1 and conn.commits == 0


# =============================== encryption ===========================================
def cipher(prev=None):
    return Cipher(generate_key(), previous=prev)


def test_encrypt_decrypt_roundtrip_bytes_and_json():
    c = cipher()
    assert c.decrypt(c.encrypt(b"report text", "rep1"), "rep1") == b"report text"
    obj = {"a": ["ä", 1], "b": None}
    assert c.decrypt_json(c.encrypt_json(obj, "rep1"), "rep1") == obj


def test_ciphertext_is_randomised_versioned_and_hides_plaintext():
    c = cipher()
    a, b = c.encrypt(b"same", "r"), c.encrypt(b"same", "r")
    assert a != b and a.startswith(b"SD1") and b"same" not in a


def test_wrong_aad_wrong_key_or_tampering_fails():
    c = cipher()
    blob = c.encrypt(b"secret", "rep1")
    with pytest.raises(DecryptionError):
        c.decrypt(blob, "rep2")  # ciphertext cannot be moved to another record
    with pytest.raises(DecryptionError):
        cipher().decrypt(blob, "rep1")
    flipped = blob[:-1] + bytes([blob[-1] ^ 1])
    with pytest.raises(DecryptionError):
        c.decrypt(flipped, "rep1")
    with pytest.raises(DecryptionError):
        c.decrypt(b"garbage", "rep1")
    with pytest.raises(DecryptionError):
        c.decrypt(b"", "rep1")


def test_key_rotation_previous_key_still_decrypts_old_data():
    old_key = generate_key()
    old = Cipher(old_key).encrypt(b"old data", "r")
    new = Cipher(generate_key(), previous=old_key)
    assert new.decrypt(old, "r") == b"old data"
    assert Cipher(generate_key()).encrypt(b"x", "r")[3:7] != old[3:7]  # key fingerprint differs


@pytest.mark.parametrize("bad", ["", "short", "!!!notbase64!!!", "QUJD"])
def test_invalid_keys_are_rejected(bad):
    with pytest.raises(ValueError):
        Cipher(bad)


def test_generated_keys_are_unique_and_32_bytes():
    import base64

    a, b = generate_key(), generate_key()
    assert a != b and len(base64.urlsafe_b64decode(a)) == 32


def test_cipher_repr_never_shows_the_key():
    k = generate_key()
    assert k not in repr(Cipher(k)) and k not in str(Cipher(k))


def test_cipher_from_keys(tmp_path):
    k, p = generate_key(), generate_key()
    c = Cipher.from_keys(keys(tmp_path, f"SURGIDOC_ENCRYPTION_KEY={k}\nSURGIDOC_ENCRYPTION_KEY_PREVIOUS={p}\n"))
    assert Cipher(p).decrypt(Cipher(p).encrypt(b"x", "a"), "a") == b"x"
    assert c.decrypt(Cipher(p).encrypt(b"x", "a"), "a") == b"x"
    with pytest.raises(MissingKey, match="SURGIDOC_ENCRYPTION_KEY"):
        Cipher.from_keys(keys(tmp_path, "X=1\n"))


# =============================== compliance =============================================
def git_repo(tmp_path, ignored=True, tracked=False):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "keys.txt").write_text("A=1\n")
    if ignored:
        (tmp_path / ".gitignore").write_text("keys.txt\n")
    if tracked:
        subprocess.run(["git", "add", "-f", "keys.txt"], cwd=tmp_path, check=True)
    return tmp_path


GOOD = ("SURGIDOC_USERS=Rex:reviewer\nSURGIDOC_ENCRYPTION_KEY={k}\n"
        "TIGERCLOUD_SERVICE_URL=postgres://u@h.cloud:5432/tsdb?sslmode=require\nREDIS_URL=rediss://default:pw@h:6379\n")


def git_repo_at(tmp_path):
    d = tmp_path / "repo"
    d.mkdir(exist_ok=True)
    return git_repo(d)


def compliance(tmp_path, text=None, repo=None, log=None):
    k = keys(tmp_path, (GOOD if text is None else text).format(k=generate_key()))
    authz = Authorizer.from_keys(k) if "SURGIDOC_USERS" in k else Authorizer({})
    return k, check_compliance(k, authz, log or audit(), repo=repo or git_repo_at(tmp_path))


def failed(checks):
    return {c.name for c in checks if not c.passed}


def test_a_correct_setup_is_fully_compliant(tmp_path):
    _, checks = compliance(tmp_path)
    assert failed(checks) == set() and len(checks) >= 6
    assert_compliant(checks)


def test_secrets_file_must_be_ignored_and_untracked(tmp_path):
    d = tmp_path / "r1"
    d.mkdir()
    _, checks = compliance(tmp_path, repo=git_repo(d, ignored=False))
    assert "secrets-file-ignored" in failed(checks)
    d2 = tmp_path / "r2"
    d2.mkdir()
    _, checks = compliance(tmp_path, repo=git_repo(d2, ignored=True, tracked=True))
    assert "secrets-file-ignored" in failed(checks)


def test_missing_reviewer_key_or_users_fail(tmp_path):
    _, c1 = compliance(tmp_path, GOOD.replace("Rex:reviewer", "Rex:operator"))
    assert "reviewer-configured" in failed(c1)
    _, c2 = compliance(tmp_path, GOOD.replace("SURGIDOC_ENCRYPTION_KEY={k}\n", ""))
    assert "encryption-key-valid" in failed(c2)
    _, c3 = compliance(tmp_path, GOOD.replace("SURGIDOC_ENCRYPTION_KEY={k}", "SURGIDOC_ENCRYPTION_KEY=short"))
    assert "encryption-key-valid" in failed(c3)


def test_tls_is_required_for_database_and_redis(tmp_path):
    _, c = compliance(tmp_path, GOOD.replace("?sslmode=require", ""))
    assert "database-tls" in failed(c)
    _, c = compliance(tmp_path, GOOD.replace("rediss://", "redis://"))
    assert "redis-tls" in failed(c)
    _, c = compliance(tmp_path, GOOD.replace("TIGERCLOUD_SERVICE_URL=postgres://u@h.cloud:5432/tsdb?sslmode=require\n", ""))
    assert "database-tls" in failed(c)


def test_broken_audit_chain_fails_compliance(tmp_path):
    log = filled()
    log.sink.entries[1]["actor"] = "evil"
    _, c = compliance(tmp_path, log=log)
    assert "audit-chain-intact" in failed(c)


def test_assert_compliant_lists_failures_without_leaking_values(tmp_path):
    k, c = compliance(tmp_path, GOOD.replace("rediss://", "redis://"))
    with pytest.raises(ComplianceError) as e:
        assert_compliant(c)
    assert "redis-tls" in str(e.value) and "pw@h" not in str(e.value) and k["SURGIDOC_ENCRYPTION_KEY"] not in str(e.value)
    assert all("pw@h" not in c_.detail for c_ in c)


# =============================== live (opt-in) ============================================
@pytest.mark.skipif(not os.environ.get("SURGIDOC_LIVE"), reason="set SURGIDOC_LIVE=1 to run live checks")
def test_live_pg_audit_log_is_append_only_and_chained():
    import uuid

    from surgidoc.retrieval import connect_tiger

    conn = connect_tiger(load_keys(Path(__file__).resolve().parent.parent / "keys.txt"))
    sink = PgSink(conn, prefix="smoke_" + uuid.uuid4().hex[:8])
    try:
        sink.setup()
        log = AuditLog(sink)
        log.log("Rex", "report:view", resource="r1", detail={"n": 1, "u": "ü"})
        log.log("Olive", "report:approve", resource="r1", outcome="denied")
        log.log("Rex", "report:approve", resource="r1")
        assert log.verify().ok and log.head()[0] == 3
        for sql in (f"UPDATE {sink.table} SET actor = 'evil' WHERE seq = 1", f"DELETE FROM {sink.table} WHERE seq = 2",
                    f"TRUNCATE {sink.table}"):
            with pytest.raises(Exception, match="append-only"):
                with conn.cursor() as cur:
                    cur.execute(sql)
            conn.rollback()
        assert log.verify().ok and len(log.entries()) == 3
    finally:
        conn.rollback()
        sink.drop()
        conn.close()
