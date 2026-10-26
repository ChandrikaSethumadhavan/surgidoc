"""AC-8: no key material in tracked files; keys.txt stays out of git."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),  # Google API key
    re.compile(r"pa-[0-9A-Za-z_\-]{30,}"),  # Voyage
    re.compile(r"gh[pousr]_[0-9A-Za-z]{36,}"),  # GitHub tokens
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def tracked():
    return [ROOT / f for f in git("ls-files").stdout.splitlines()]


def test_keys_txt_is_ignored_and_untracked():
    assert git("check-ignore", "keys.txt").returncode == 0
    assert "keys.txt" not in git("ls-files").stdout.split()


def test_no_secret_patterns_in_tracked_files():
    hits = []
    for f in tracked():
        if f.suffix in {".png", ".pdf"} or not f.is_file():
            continue
        text = f.read_text(errors="ignore")
        hits += [f.name for p in PATTERNS if p.search(text)]
    assert not hits, hits


def test_scanner_detects_a_planted_key():
    assert PATTERNS[0].search("k=AIza" + "a" * 35)
