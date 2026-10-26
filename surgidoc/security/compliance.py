"""FR-13: compliance checks. Each check reports pass/fail with a safe detail; values and secrets are never echoed."""
import subprocess
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

from surgidoc.config import MissingKey
from surgidoc.security.crypto import Cipher


class ComplianceError(Exception):
    pass


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True).returncode


def check_compliance(keys, authz, audit, repo=".", keys_file="keys.txt"):
    out = []

    def add(name, passed, ok, bad):
        out.append(Check(name, bool(passed), ok if passed else bad))

    ignored = _git(repo, "check-ignore", "-q", keys_file) == 0
    untracked = _git(repo, "ls-files", "--error-unmatch", keys_file) != 0
    add("secrets-file-ignored", ignored and untracked, "secrets file is git-ignored and untracked",
        "secrets file is tracked, not ignored, or the repository is missing")
    add("reviewer-configured", any("reviewer" in r for r in authz.users.values()), "a reviewer is configured",
        "no user has the reviewer role")
    try:
        Cipher.from_keys(keys)
        key_ok, why = True, ""
    except (MissingKey, ValueError):
        key_ok, why = False, "encryption key is missing or invalid"
    add("encryption-key-valid", key_ok, "encryption key is valid", why)
    add("audit-chain-intact", audit.verify().ok, "audit chain verifies", "audit chain is broken")
    db = keys.get("TIGERCLOUD_SERVICE_URL")
    mode = parse_qs(urlsplit(db).query).get("sslmode", [""])[0] if db else ""
    add("database-tls", mode in {"require", "verify-ca", "verify-full"}, "database connection requires TLS",
        "database URL is missing or does not require TLS (sslmode)")
    add("redis-tls", bool(keys.get("REDIS_URL")) and urlsplit(keys["REDIS_URL"]).scheme == "rediss",
        "redis connection uses TLS", "redis URL is missing or does not use rediss://")
    return out


def assert_compliant(checks):
    bad = [c for c in checks if not c.passed]
    if bad:
        raise ComplianceError("non-compliant: " + "; ".join(f"{c.name} ({c.detail})" for c in bad))
