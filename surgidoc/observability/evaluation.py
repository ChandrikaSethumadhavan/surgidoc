"""FR-12: evaluation (accuracy, relevance, quality). Deterministic measures, no LLM-as-judge in the measuring loop.

Retrieval: recall@k and MRR against labelled relevant chunk ids. Reports: coverage of expected terms, citation validity,
and numeric support (every number in a statement must appear in a cited source), a cheap faithfulness proxy.
These are proxies. A real evaluation set (real frames plus reference findings) does not exist yet.
"""
import re
from dataclasses import dataclass, field

from surgidoc.retrieval import chunk_id

_NUM = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant: set = field(default_factory=set)


@dataclass(frozen=True)
class ReportCase:
    findings: list
    expected_terms: list = field(default_factory=list)


def evaluate_retrieval(retrieve, cases, k=5):
    if not cases or any(not c.relevant for c in cases):
        raise ValueError("every retrieval case needs a query and at least one relevant chunk id")
    recalls, rrs = [], []
    for c in cases:
        ids = [chunk_id(x) for x in retrieve(c.query)[:k]]
        recalls.append(len(c.relevant & set(ids)) / len(c.relevant))
        rrs.append(next((1 / r for r, i in enumerate(ids, 1) if i in c.relevant), 0.0))
    n = len(cases)
    return {f"recall@{k}": sum(recalls) / n, "mrr": sum(rrs) / n, "n": n}


def _scores(report, expected_terms):
    st = report.statements
    if not st:
        return {"coverage": 0.0, "citation_validity": 0.0, "numeric_support": 0.0}
    text = " ".join(s.text for s in st).casefold()
    coverage = sum(t.casefold() in text for t in expected_terms) / len(expected_terms) if expected_terms else 1.0
    valid = sum(bool(s.sources) and all(c in report.sources for c in s.sources) for s in st) / len(st)

    def supported(s):
        pool = " ".join(report.sources[c]["text"] for c in s.sources if c in report.sources)
        return set(_NUM.findall(s.text)) <= set(_NUM.findall(pool))

    return {"coverage": coverage, "citation_validity": valid, "numeric_support": sum(map(supported, st)) / len(st)}


def evaluate_reports(agents, cases):
    if not cases:
        raise ValueError("no report cases to evaluate")
    rows = []
    for c in cases:
        try:
            r = agents.run(c.findings)
        except Exception as e:  # a crashing case is a failed case, not a crashed evaluation
            rows.append({"status": "error", "error": type(e).__name__, "attempts": None,
                         "coverage": 0.0, "citation_validity": 0.0, "numeric_support": 0.0})
            continue
        rows.append({"status": r.status, "error": None, "attempts": r.attempts, **_scores(r, c.expected_terms)})
    n = len(rows)
    attempts = [r["attempts"] for r in rows if r["attempts"] is not None]
    mean = lambda key: sum(r[key] for r in rows) / n  # noqa: E731
    return {"cases": rows, "metrics": {
        "n": n, "approval_rate": sum(r["status"] == "approved" for r in rows) / n,
        "error_rate": sum(r["status"] == "error" for r in rows) / n,
        "mean_attempts": sum(attempts) / len(attempts) if attempts else None,
        "coverage": mean("coverage"), "citation_validity": mean("citation_validity"),
        "numeric_support": mean("numeric_support")}}


def check_thresholds(metrics, thresholds):
    """thresholds: {metric: (">=" | "<=", limit)}. Returns the list of failures (empty means the gate passes)."""
    failures = []
    for name, (op, limit) in thresholds.items():
        if name not in metrics:
            raise ValueError(f"unknown metric: {name}")
        if op not in (">=", "<="):
            raise ValueError(f"unsupported operator: {op}")
        v = metrics[name]
        if v is None or (v < limit if op == ">=" else v > limit):
            failures.append(f"{name}: {v} is not {op} {limit}")
    return failures
