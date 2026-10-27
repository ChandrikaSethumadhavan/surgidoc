"""FR-3, FR-4: load the guideline PDFs in dataset/ into the knowledge base (Tiger Cloud) and report what is stored.

    python -m surgidoc.kb load [DIR] [--dry-run] [--max-chunks N] [--replace] [--keys keys.txt]
    python -m surgidoc.kb status [--keys keys.txt]
    python -m surgidoc.kb eval [--keyword-only] [--out data/kb_eval.json] [--keys keys.txt]

Resumable and idempotent: chunks already stored are skipped, every batch is committed, so a run that hits the free-tier
embedding quota (HTTP 429) stops cleanly and the next run continues where it stopped. Smallest documents go first.
"""
import argparse
import json
import math
import time
from datetime import datetime, timezone
import urllib.error
from pathlib import Path

from surgidoc.config import MissingKey, load_keys
from surgidoc.ingest import is_pdf, pdf_chunks
from surgidoc.retrieval import GeminiEmbedder, PgStore, chunk_id, connect_tiger, retrieve, rrf, term_overlap_rerank

BATCH = 64
TPM = 24000  # tokens per minute: stays under the 30k free-tier figure (community-reported, verify in AI Studio)
CHARS_PER_TOKEN = 3.5


class QuotaReached(Exception):
    pass


def discover(folder):
    """PDF files directly inside folder, recognised by content when the name has no extension; sidecar files are skipped."""
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and is_pdf(p))


def _embed(embedder, chunks, sleep, retries=2, wait=65):
    for attempt in range(retries + 1):
        try:
            return embedder.embed_documents([c.text for c in chunks], titles=[c.title for c in chunks])
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            if attempt == retries:
                raise QuotaReached("the embedding quota is used up for now") from e
            sleep(wait)  # a per-minute limit clears within a minute; a daily one does not, and we give up above


def load(folder, store, embedder, *, max_chunks=None, replace=False, batch=BATCH, tpm=TPM, sleep=time.sleep,
         clock=time.monotonic, log=print):
    """-> {"documents", "chunks", "stored_before", "stored_now", "remaining", "stopped"}."""
    docs = []
    for p in discover(folder):
        try:
            docs.append((p.name, pdf_chunks(p, p.name)))
        except Exception as e:  # one unreadable file must not stop the rest
            log(f"skipped {p.name}: {type(e).__name__}")
    docs.sort(key=lambda d: len(d[1]))
    total = sum(len(c) for _, c in docs)
    sum_ = {"documents": len(docs), "chunks": total, "stored_before": 0, "stored_now": 0, "remaining": 0, "stopped": None}
    for name, chunks in docs:
        if replace:
            store.delete_source(name)
        have = store.ids_for(name)
        sum_["stored_before"] += len(have)
        todo = [c for c in chunks if chunk_id(c) not in have]
        log(f"{name}: {len(chunks)} chunks, {len(have)} stored, {len(todo)} to embed")
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            if max_chunks is not None and sum_["stored_now"] + len(part) > max_chunks:
                part = part[:max(0, max_chunks - sum_["stored_now"])]
                if not part:
                    sum_["stopped"] = "chunk limit for this run reached"
                    break
            started = clock()
            try:
                vectors = _embed(embedder, part, sleep)
            except QuotaReached as e:
                sum_["stopped"] = str(e)
                break
            store.add(part, vectors)
            sum_["stored_now"] += len(part)
            tokens = sum(len(c.text) for c in part) / CHARS_PER_TOKEN
            sleep(max(0.0, 60 * tokens / tpm - (clock() - started)))
        if sum_["stopped"]:
            break
    sum_["remaining"] = total - sum_["stored_before"] - sum_["stored_now"]
    return sum_


# Question -> the document(s) that should answer it. A question whose documents are not stored yet is skipped, not failed.
EVAL_SET = [
    ("What elements should an upper GI endoscopy report contain?", {"2025_a-2652-8941.pdf"}),
    ("What information must patients receive before giving consent for an endoscopic procedure?", {"2023_a-2133-3365.pdf"}),
    ("Who may administer propofol for endoscopy and what monitoring is required?", {"2015_s_0034_1393414.pdf", "2026_a-2898-6540.pdf"}),
    ("How should flexible endoscopes be reprocessed after use?", {"2018_a_0759_1629.pdf"}),
    ("How can duodenoscope-related multidrug-resistant infections be prevented?", {"2017_s_0043_120523.pdf"}),
    ("When should anticoagulants or antiplatelet drugs be stopped before endoscopy?", {"2021_a_1547_2282.pdf"}),
    ("How can radiation exposure of staff be reduced during endoscopy?", {"2012_radiation_protection.pdf"}),
    ("What techniques exist for endoscopic submucosal dissection?", {"2023_a-2031-0874.pdf"}),
    ("How is EUS-guided tissue sampling performed?", {"2017_s_0043_119219.pdf"}),
    ("Which advanced imaging technologies help detect lesions, such as narrow-band imaging?", {"2016_s_0042_118087.pdf"}),
    ("How is endoscopic ultrasound used in liver disease?", {"2026_a-2964-7886.pdf"}),
    ("Welche Qualitätsanforderungen gelten für die gastrointestinale Endoskopie?",
     {"s-0041-109520", "Deutsch_German_Version_S2k-LL-Qualitaetsanforderungen-Endoskopie-v2.1_Langfassung_12.01.26.pdf"}),
]


# Questions the guidelines cannot answer: the chat should say "not covered". The in-scope ones check it does not refuse too much.
OUT_OF_SCOPE = ["What is the best pizza topping?", "Who won the football world cup in 2018?", "How do I fix a flat bicycle tyre?",
                "What is the capital of Australia?", "Which Python web framework is the fastest?", "How many moons does Jupiter have?"]
IN_SCOPE = [q for q, _ in EVAL_SET[:4]]


def evaluate_abstention(ask, off_topic=OUT_OF_SCOPE, in_scope=IN_SCOPE):
    """ask(question) -> {"covered": bool}. abstain_correct = share of off-topic questions answered 'not covered';
    answer_rate = share of in-scope questions that did get a covered answer."""
    def covered(q):  # None when the model could not be reached: counted as an error, not as a right or wrong answer
        try:
            return bool(ask(q)["covered"])
        except Exception:
            return None

    off = [(q, covered(q)) for q in off_topic]
    on = [(q, covered(q)) for q in in_scope]
    off_ok, on_ok = [c is False for _, c in off if c is not None], [c for _, c in on if c is not None]
    return {"abstain_correct": sum(off_ok) / len(off_ok) if off_ok else None,
            "answer_rate": sum(on_ok) / len(on_ok) if on_ok else None,
            "errors": sum(c is None for _, c in off + on),
            "off_topic": [{"question": q, "abstained": None if c is None else not c} for q, c in off],
            "in_scope": [{"question": q, "answered": c} for q, c in on]}


def _gains(got, expected, k=5):
    return [1.0 if c.source in expected else 0.0 for c in got[:k]]


def evaluate(questions, search, available):
    """search(question) -> ranked chunks (top 5). -> {"rows", "asked", "skipped", "hit_at_5", "mrr", "precision_at_5", "ndcg_at_5"}.
    rank = 1-based position of the first chunk from an expected document; hit@5 = share of asked questions with a rank;
    precision@5 = share of the 5 results from an expected document; nDCG@5 = rank-discounted gain against 5 relevant results.
    Relevance is judged per document, not per passage."""
    rows = []
    for question, expected in questions:
        if not expected & set(available):
            rows.append({"question": question, "expected": sorted(expected), "skipped": "documents not loaded"})
            continue
        got = search(question)
        rank = next((i for i, c in enumerate(got, 1) if c.source in expected), None)
        top = got[0] if got else None
        gains = _gains(got, expected)
        ideal = sum(1 / math.log2(i + 1) for i in range(1, 6))
        rows.append({"question": question, "expected": sorted(expected), "rank": rank, "precision": sum(gains) / 5,
                     "ndcg": sum(g / math.log2(i + 1) for i, g in enumerate(gains, 1)) / ideal,
                     "top": {"source": top.source, "page": top.page} if top else None})
    asked = [r for r in rows if "skipped" not in r]
    return {"rows": rows, "asked": len(asked), "skipped": len(rows) - len(asked),
            "hit_at_5": sum(r["rank"] is not None for r in asked) / len(asked) if asked else 0.0,
            "mrr": sum(1 / r["rank"] for r in asked if r["rank"]) / len(asked) if asked else 0.0,
            "precision_at_5": sum(r["precision"] for r in asked) / len(asked) if asked else 0.0,
            "ndcg_at_5": sum(r["ndcg"] for r in asked) / len(asked) if asked else 0.0}


def dry_run(folder, log=print):
    total = 0
    for p in discover(folder):
        chunks = pdf_chunks(p, p.name)
        total += len(chunks)
        c = chunks[0] if chunks else None
        log(f"{p.name}: {len(chunks)} chunks up to page {max((x.page for x in chunks), default=0)}, "
            f"{c.lang if c else '-'}, {c.title[:70] if c else '-'}")
    log(f"total {total} chunks = {total} embedding requests; at 1,000 requests/day (free tier, unverified) about "
        f"{-(-total // 1000)} day(s)")
    return total


def stages(store, embedder):
    """The retrieval stages to compare, each -> top 5 chunks. Vector stages need an embedder (they spend embedding quota)."""
    out = {"bm25": lambda q: store.get(store.bm25(q, 5)),
           "bm25 + rerank": lambda q: term_overlap_rerank(q, store.get(store.bm25(q, 30)))[:5]}
    if embedder:
        memo = {}

        def vec(q):  # one embedding call per question, shared by the three vector stages
            if q not in memo:
                memo[q] = embedder.embed_query(q)
            return memo[q]

        class Memo:
            embed_query = staticmethod(vec)

        out["vector"] = lambda q: store.get(store.vector(vec(q), 5))
        out["hybrid (RRF)"] = lambda q: store.get([i for i, _ in rrf([store.bm25(q, 30), store.vector(vec(q), 30)], top=5)])
        out["hybrid + rerank"] = lambda q: retrieve(q, store, Memo)
    return out


def compare(store, embedder, log=print, questions=EVAL_SET):
    available = {src for src, _, _ in store.counts(any_model=embedder is None)}
    res = {"stages": {}}
    log(f"{'stage':18} {'hit@5':>6} {'MRR':>6} {'P@5':>6} {'nDCG@5':>7}")
    for name, search in stages(store, embedder).items():
        r = evaluate(questions, search, available)
        res["stages"][name] = {k: r[k] for k in ("asked", "hit_at_5", "mrr", "precision_at_5", "ndcg_at_5")}
        log(f"{name:18} {r['hit_at_5']:6.2f} {r['mrr']:6.2f} {r['precision_at_5']:6.2f} {r['ndcg_at_5']:7.2f}")
    return res


def _eval(store, embedder, out, log):
    search = (lambda q: retrieve(q, store, embedder)) if embedder else (lambda q: term_overlap_rerank(q, store.get(store.bm25(q, 30)))[:5])
    res = evaluate(EVAL_SET, search, {src for src, _, _ in store.counts(any_model=embedder is None)})
    res.update(mode="hybrid" if embedder else "keyword-only", at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    for r in res["rows"]:
        top = r.get("top")
        log(("skip  " if "skipped" in r else f"rank {r['rank'] or '-'}  ") + r["question"][:78]
            + (f"  -> {top['source']} p.{top['page']}" if top else ""))
    log(f"{res['mode']}: asked {res['asked']}, skipped {res['skipped']}, hit@5 {res['hit_at_5']:.2f}, MRR {res['mrr']:.2f}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


def _save(res, out, kind, keyword_only):
    res.update(kind=kind, mode="keyword-only" if keyword_only else "hybrid", at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


def main(argv=None, connect=connect_tiger, embedder_cls=GeminiEmbedder, log=print):
    ap = argparse.ArgumentParser(prog="python -m surgidoc.kb", description="Load and inspect the guideline knowledge base.")
    ap.add_argument("command", choices=["load", "status", "eval"])
    ap.add_argument("folder", nargs="?", default="dataset")
    ap.add_argument("--keys", default="keys.txt")
    ap.add_argument("--dry-run", action="store_true", help="read and chunk the PDFs only; no keys, no network")
    ap.add_argument("--max-chunks", type=int, help="embed at most this many chunks in this run")
    ap.add_argument("--keyword-only", action="store_true", help="eval without embeddings (BM25 + rerank only): spends no embedding quota")
    ap.add_argument("--compare", action="store_true", help="eval: compare bm25, vector, hybrid and rerank stages (vector stages spend embedding quota)")
    ap.add_argument("--abstain", action="store_true", help="eval: check the chat says 'not covered' to off-topic questions (about 10 model calls)")
    ap.add_argument("--out", help="where eval writes its result (default data/kb_eval.json, kb_compare.json or kb_abstain.json)")
    ap.add_argument("--replace", action="store_true", help="delete each document's stored chunks first (after changing chunking)")
    args = ap.parse_args(argv)
    if args.command == "load" and args.dry_run:
        return dry_run(args.folder, log)
    try:
        keys = load_keys(args.keys)
        store = PgStore(connect(keys))
        store.setup()
        if args.command == "status":
            rows = store.counts()
            for source, n, pages in rows:
                log(f"{source}: {n} chunks, up to page {pages}")
            log(f"total {sum(n for _, n, _ in rows)} chunks in {len(rows)} documents")
            return rows
        if args.command == "eval" and args.compare:
            res = compare(store, None if args.keyword_only else embedder_cls.from_keys(keys), log)
            return _save(res, args.out or "data/kb_compare.json", "compare", args.keyword_only)
        if args.command == "eval" and args.abstain:
            from surgidoc.gateway import Gateway
            from surgidoc.pipeline import search_with_fallback
            from surgidoc.qa import GuidelineQA

            embedder = embedder_cls.from_keys(keys)
            qa = GuidelineQA(Gateway.from_keys(keys), lambda q: search_with_fallback(q, store, embedder))
            res = evaluate_abstention(qa.ask)
            fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
            log(f"off-topic answered 'not covered': {fmt(res['abstain_correct'])}; in-scope questions answered: {fmt(res['answer_rate'])}; "
                f"model errors: {res['errors']}")
            return _save(res, args.out or "data/kb_abstain.json", "abstention", False)
        if args.command == "eval":
            return _eval(store, None if args.keyword_only else embedder_cls.from_keys(keys), args.out or "data/kb_eval.json", log)
        result = load(args.folder, store, embedder_cls.from_keys(keys), max_chunks=args.max_chunks, replace=args.replace, log=log)
    except FileNotFoundError:
        ap.error(f"file or folder not found (keys: {args.keys}, folder: {args.folder})")
    except MissingKey as e:
        ap.error(f"keys.txt is missing {e}")
    log(f"stored now {result['stored_now']}, remaining {result['remaining']}" + (f", stopped: {result['stopped']}" if result["stopped"] else ""))
    return result


if __name__ == "__main__":
    main()
