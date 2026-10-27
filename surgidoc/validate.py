"""FR-12: validation run. The whole pipeline on one held-out CholecSeg8k video, scored against its segmentation masks.

    python -m surgidoc.validate [--video video09] [--split dev|test] [--final] [--root data/cholecseg8k]
                                [--out data/validation_run.json] [--keys keys.txt]

The sampled frames of the video are written as an mp4 (one frame every 2 s, which is how the pipeline samples), so the real
upload path runs: frame extraction, vision description, retrieval, generator, judge, guardrails. The dev split is the default;
the test split needs --final and every look at it is logged to test_runs.json (same rule as dataset.evaluate_split).
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

from surgidoc.config import MissingKey, load_keys
from surgidoc.dataset import (DatasetError, FrameRef, HeldOutSplitError, _atomic_write, frame_coverage, load_manifest,
                              mentioned_classes, present_classes, score_frame)
from surgidoc.pdf import is_guideline

MAX_FRAMES = 16  # what the pipeline's frame extraction keeps


def clip_refs(root, video, split):
    refs = [FrameRef(f["video"], f["clip"], f["index"], f["split"]) for f in load_manifest(root)["frames"]
            if f["video"] == video and f["split"] == split]
    refs.sort(key=lambda r: (r.clip, r.index))
    missing = [p for r in refs for p in (r.endo, r.mask) if not (Path(root) / p).is_file()]
    if not refs:
        raise DatasetError(f"no {split} frames for {video} in the manifest")
    if missing:
        raise DatasetError(f"{len(missing)} file(s) are not downloaded yet (e.g. {missing[0]})")
    return refs[:MAX_FRAMES]


def build_video(root, refs, path):
    """Each frame is held for 2 s (1 fps, written twice), so sampling every 2 s returns exactly these frames in order."""
    import cv2

    first = cv2.imread(str(Path(root) / refs[0].endo))
    h, w = first.shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (w, h))
    for r in refs:
        img = cv2.imread(str(Path(root) / r.endo))
        out.write(cv2.resize(img, (w, h)))
        out.write(cv2.resize(img, (w, h)))
    out.release()
    return path


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def validate(pipeline, operator, root, video, split="dev", final=False, workdir=None, clock=time.monotonic):
    if split not in ("dev", "test"):
        raise ValueError("split must be 'dev' or 'test'")
    if split == "test" and not final:
        raise HeldOutSplitError("the test split is held out: pass final=True for a final report (every run is logged)")
    root = Path(root)
    refs = clip_refs(root, video, split)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        mp4 = build_video(root, refs, Path(tmp) / f"{video}.mp4")
        started = clock()
        res = pipeline.run(str(mp4), operator)
        seconds = clock() - started
    rows = []
    for f in res.findings:
        cov = frame_coverage(root / refs[f.frame].mask)
        rows.append({"clip": refs[f.frame].clip, "index": refs[f.frame].index, "text": f.text,
                     "reference": sorted(present_classes(cov)), "mentioned": sorted(mentioned_classes(f.text)), **score_frame(f.text, cov)})
    stmts = res.report.statements if res.report else []
    sources = res.report.sources if res.report else {}
    metrics = {
        "frames_sent": len(refs), "frames_described": len(rows),
        "structure_recall": _mean(r["recall"] for r in rows),
        "absent_mention_rate": sum(bool(r["absent_mentions"]) for r in rows) / len(rows) if rows else None,
        "unclear_rate": sum(r["unclear"] for r in rows) / len(rows) if rows else None,
        "status": res.status, "judge_approved": bool(res.report and res.report.status == "approved"),
        "attempts": res.report.attempts if res.report else None,
        "guardrails_passed": res.status == "pending_review",
        "statements": len(stmts), "guideline_reminders": sum(is_guideline(s) for s in stmts),
        "citation_validity": (sum(bool(s.sources) and all(c in sources for c in s.sources) for s in stmts) / len(stmts)) if stmts else None,
        "seconds": round(seconds, 1)}
    out = {"video": video, "split": split, "metrics": metrics, "frames": rows}
    if split == "test":  # a visible record of every look at the held-out videos
        log = root / "test_runs.json"
        runs = json.loads(log.read_text()) if log.is_file() else []
        runs.append({"video": video, "metrics": metrics})
        _atomic_write(log, json.dumps(runs, indent=1).encode())
    return out


def main(argv=None, build=None, log=print):
    ap = argparse.ArgumentParser(prog="python -m surgidoc.validate", description="Run the pipeline on a held-out video and score it.")
    ap.add_argument("--video", default="video09")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--final", action="store_true", help="required for the held-out test split; every run is logged")
    ap.add_argument("--root", default="data/cholecseg8k")
    ap.add_argument("--out", default="data/validation_run.json")
    ap.add_argument("--keys", default="keys.txt")
    args = ap.parse_args(argv)
    if args.split == "test" and not args.final:
        ap.error("the test split is held out: add --final (every run is logged)")
    try:
        keys = load_keys(args.keys)
        users = dict(u.split(":", 1) for u in keys.require("SURGIDOC_USERS").split(";") if ":" in u)
        operator = next((n for n, r in users.items() if "operator" in r.split("+")), None)
        if operator is None:
            ap.error("SURGIDOC_USERS has no user with the operator role")
        if build is None:
            from surgidoc.pipeline import build_pipeline as build
        pipeline = build(keys)
        pipeline.setup()
        result = validate(pipeline, operator, args.root, args.video, args.split, args.final)
    except FileNotFoundError:
        ap.error(f"file not found (keys: {args.keys}, dataset: {args.root})")
    except MissingKey as e:
        ap.error(f"keys.txt is missing {e}")
    except DatasetError as e:
        ap.error(str(e))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    for k, v in result["metrics"].items():
        log(f"{k}: {round(v, 3) if isinstance(v, float) else v}")
    return result


if __name__ == "__main__":
    main()
