"""FR-2, FR-12: real-frame evaluation on CholecSeg8k (newslab/cholecseg8k, CC BY-NC-SA 4.0: non-commercial, never committed).

- Split BY VIDEO (frames of one video are near-duplicates, a frame split would leak): deterministic seeded hash.
- Resumable subset download: one file at a time, atomic, validated PNG, checksummed manifest, safe to interrupt.
- Reference labels from the watershed masks (class table from the dataset description).
- Evaluation of frame descriptions against those labels: recall of labelled structures, mentions of labelled classes that
  are absent (a hallucination proxy), and the share of "unclear" answers. Dev runs are free; test runs need `final=True`
  and are logged, so tuning on the held-out videos would be visible.
Only the 13 labelled classes can be judged: clips, bags and other tools are not labelled. It is cholecystectomy video.
"""
import base64
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SLUG = "newslab/cholecseg8k"
VIDEOS = ["video01", "video09", "video12", "video17", "video18", "video20", "video24", "video25", "video26", "video27",
          "video28", "video35", "video37", "video43", "video48", "video52", "video55"]  # verified via the Kaggle API
BACKGROUND = "Black Background"
# Pixel values are DECIMAL class numbers (verified on a real mask: 21 liver, 50 background, 22 gallbladder, 12 fat, 13 GI tract).
# The dataset description's "RGB hexcode" column prints the same digits ("#505050"), which is easy to misread as hex.
CLASSES = {50: BACKGROUND, 11: "Abdominal Wall", 21: "Liver", 13: "Gastrointestinal Tract", 12: "Fat", 31: "Grasper",
           23: "Connective Tissue", 24: "Blood", 25: "Cystic Duct", 32: "L-hook Electrocautery", 22: "Gallbladder",
           33: "Hepatic Vein", 5: "Liver Ligament"}
UNLABELLED = 255  # boundary pixels between regions (~0.6% on the real frame inspected); ignored
PNG_MAGIC, PNG_END = b"\x89PNG\r\n\x1a\n", b"IEND\xaeB`\x82"
_SAFE = re.compile(r"[A-Za-z0-9_\-]+")
_FRAME = re.compile(r"^frame_(\d+)_endo\.png$")


class DatasetError(Exception):
    pass


class HeldOutSplitError(Exception):
    pass


# =============================== split and sampling ===============================================================
def assign_splits(videos=VIDEOS, n_dev=3, n_test=3, seed="surgidoc-split-v1"):
    if n_dev < 1 or n_test < 1 or n_dev + n_test > len(videos):
        raise ValueError("need at least one dev and one test video and not more than the dataset has")
    ranked = sorted(videos, key=lambda v: hashlib.sha256(f"{seed}:{v}".encode()).hexdigest())
    return {v: ("test" if i < n_test else "dev" if i < n_test + n_dev else "unused") for i, v in enumerate(ranked)}


@dataclass(frozen=True)
class FrameRef:
    video: str
    clip: str
    index: int
    split: str

    def __post_init__(self):
        if not (_SAFE.fullmatch(self.video) and _SAFE.fullmatch(self.clip)) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("invalid frame reference")

    @property
    def endo(self):
        return f"{self.video}/{self.clip}/frame_{self.index}_endo.png"

    @property
    def mask(self):
        return f"{self.video}/{self.clip}/frame_{self.index}_endo_watershed_mask.png"


def _walk(tree, path):
    dirs, files, token = [], [], None
    while True:
        d, f, token = tree(path, token)
        dirs += d
        files += f
        if not token:
            return dirs, files


def list_dataset(tree, videos):
    """tree(path, page_token) -> (dirs, files, next_token). Returns {video: {clip: sorted frame indices}}."""
    root, _ = _walk(tree, "")
    unknown = [v for v in videos if v not in root]
    if unknown:
        raise DatasetError(f"video(s) not in the dataset: {unknown}")
    out = {}
    for v in videos:
        clips, _ = _walk(tree, v)
        out[v] = {}
        for c in sorted(clips, key=lambda n: int(n.rsplit("_", 1)[1])):
            _, files = _walk(tree, f"{v}/{c}")
            out[v][c] = sorted(int(m.group(1)) for m in map(_FRAME.match, files) if m)
    return out


def sample_frames(listing, splits, per_video=12, clips_per_video=3, use=("dev", "test")):
    """Evenly spaced clips, evenly spaced frames inside each: deterministic, no near-duplicate neighbours."""
    out = []
    for video in sorted(listing):
        if splits.get(video) not in use or not listing[video]:
            continue
        clips = sorted(listing[video], key=lambda n: int(n.rsplit("_", 1)[1]))
        k = max(1, min(clips_per_video, len(clips), per_video))
        chosen = [clips[len(clips) // 2]] if k == 1 else [clips[round(i * (len(clips) - 1) / (k - 1))] for i in range(k)]
        for i, clip in enumerate(dict.fromkeys(chosen)):
            quota = per_video // k + (1 if i < per_video % k else 0)
            frames = listing[video][clip]
            m = min(quota, len(frames))
            out += [FrameRef(video, clip, frames[int((j + 0.5) * len(frames) / m)], splits[video]) for j in range(m)] if m else []
    return sorted(set(out), key=lambda r: (r.video, r.clip, r.index))


# =============================== resumable downloader =============================================================
def _valid_png(data):
    return data.startswith(PNG_MAGIC) and data.endswith(PNG_END)


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def load_manifest(root):
    p = Path(root) / "manifest.json"
    if not p.is_file():
        raise DatasetError(f"no manifest in {root}: run the dataset download/prepare step first")
    return json.loads(p.read_text())


class Downloader:
    def __init__(self, fetch, dest, sleep=time.sleep, retries=3):
        self.fetch, self.dest, self.sleep, self.retries = fetch, Path(dest), sleep, retries

    def _get(self, remote):
        for attempt in range(self.retries):
            try:
                data = self.fetch(remote)
                if not _valid_png(data):
                    raise ValueError("not a complete PNG")
                return data
            except Exception:  # transient network errors and bad content alike are retried; Ctrl-C is not swallowed
                if attempt == self.retries - 1:
                    return None
                self.sleep(2.0 ** attempt)

    def download(self, refs, max_files=None):
        root = self.dest.resolve()
        root.mkdir(parents=True, exist_ok=True)
        for stale in root.rglob("*.part"):
            stale.unlink()
        mpath = root / "manifest.json"
        manifest = json.loads(mpath.read_text()) if mpath.is_file() else {"slug": SLUG, "files": {}, "frames": []}
        known = {(f["video"], f["clip"], f["index"]) for f in manifest["frames"]}
        manifest["frames"] += [{"video": r.video, "clip": r.clip, "index": r.index, "split": r.split}
                               for r in refs if (r.video, r.clip, r.index) not in known]
        _atomic_write(mpath, json.dumps(manifest, indent=1).encode())
        done = skipped = remaining = 0
        failed = []
        for remote in [p for r in refs for p in (r.endo, r.mask)]:
            target = (root / remote).resolve()
            if root not in target.parents:
                raise ValueError(f"path escapes the destination: {remote}")
            if target.is_file() and _valid_png(target.read_bytes()):
                skipped += 1
                continue
            if max_files is not None and done >= max_files:
                remaining += 1
                continue
            data = self._get(remote)
            if data is None:
                failed.append(remote)
                continue
            _atomic_write(target, data)
            manifest["files"][remote] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            _atomic_write(mpath, json.dumps(manifest, indent=1).encode())  # after every file: a restart loses nothing
            done += 1
        return {"downloaded": done, "skipped": skipped, "failed": failed, "remaining": remaining}


def kagglehub_fetch(keys, hub=None):
    """fetch(remote_path) -> bytes via kagglehub's single-file download; logs in once; drops kagglehub's cache copy."""
    from surgidoc.ingest import _kagglehub, kaggle_login

    hub = hub or _kagglehub()
    kaggle_login(keys, hub)

    def fetch(remote):
        local = Path(hub.dataset_download(SLUG, path=remote))
        try:
            return local.read_bytes()
        finally:
            local.unlink(missing_ok=True)

    return fetch


def _auth_header(keys):
    value = lambda name: (keys.get(name) or "").strip()  # noqa: E731
    token, key, user = value("KAGGLE_API_TOKEN"), value("KAGGLE_KEY"), value("KAGGLE_USERNAME")
    if token or key.startswith("KGAT_"):
        return f"Bearer {token or key}"
    if user and key:
        return "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()
    raise DatasetError("no usable Kaggle credentials in keys.txt: set KAGGLE_API_TOKEN, or KAGGLE_KEY (an API token), "
                       "or KAGGLE_USERNAME together with a legacy KAGGLE_KEY")


def kaggle_tree(keys, urlopen=urllib.request.urlopen):
    """tree(path, page_token) -> (dirs, files, next_token) over Kaggle's public REST list-tree endpoint (stdlib only)."""
    header = _auth_header(keys)

    def tree(path, token=None):
        query = {"path": path, "pageSize": 200, **({"pageToken": token} if token else {})}
        req = urllib.request.Request(f"https://www.kaggle.com/api/v1/datasets/list-tree/{SLUG}?" + urllib.parse.urlencode(query),
                                     headers={"Authorization": header})
        try:
            with urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
        except Exception as e:  # the message can echo the token: keep only the type
            raise DatasetError(f"Kaggle listing failed ({type(e).__name__})") from None
        return ([d["name"] for d in data.get("directories", [])], [f["name"] for f in data.get("files", [])],
                data.get("nextPageToken") or None)

    return tree


def prepare(dest, tree, fetch, per_video=12, clips_per_video=3, max_files=None, seed="surgidoc-split-v1", sleep=time.sleep):
    """List, split by video, sample, and download (resumable). Videos in the 'unused' split are never even listed."""
    splits = assign_splits(seed=seed)
    refs = sample_frames(list_dataset(tree, [v for v, k in splits.items() if k != "unused"]), splits, per_video, clips_per_video)
    return Downloader(fetch, dest, sleep=sleep).download(refs, max_files=max_files)


# =============================== reference labels from masks ======================================================
def _imread(path, flags=1):
    import cv2  # by path: OpenCV needs no direct numpy import from us (numpy is only an OpenCV dependency)

    img = cv2.imread(str(path), flags)
    if img is None:
        raise DatasetError(f"not a readable image: {Path(path).name}")
    return img


def frame_coverage(mask_path):
    """{class name: share of pixels} for every class present in a watershed mask file."""
    import cv2

    img = _imread(mask_path, 0)  # grayscale: the watershed mask repeats the class id in all three channels
    counts = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel().tolist()
    total, cov, unknown = img.size, {}, 0
    for value, count in enumerate(counts):
        if not count:
            continue
        if value in CLASSES:
            cov[CLASSES[value]] = count / total
        elif value != UNLABELLED:
            unknown += count
    if unknown / total > 0.02:
        raise DatasetError(f"mask has unknown pixel values ({unknown / total:.0%} of pixels): not a CholecSeg8k watershed mask?")
    return cov


def present_classes(coverage, min_fraction=0.005):
    return {k: v for k, v in coverage.items() if k != BACKGROUND and v >= min_fraction}


# =============================== what a description mentions ======================================================
TERMS = {
    "Abdominal Wall": [r"\babdominal wall\b", r"\bbody wall\b"],
    "Liver": [r"\bliver\b(?!\s+ligament)"],
    "Gastrointestinal Tract": [r"\bgastrointestinal\b", r"\bbowel\b", r"\bintestin\w*", r"\bcolon\b", r"\bstomach\b", r"\bduoden\w*"],
    "Fat": [r"\bfat\b", r"\bfatty\b", r"\badipose\b", r"\bomentum\b"],
    "Grasper": [r"\bgraspers?\b", r"\bforceps\b", r"\bgrasping\b"],
    "Connective Tissue": [r"\bconnective tissue\b", r"\bfascia\b", r"\badhesions?\b"],
    "Blood": [r"\bblood\b", r"\bbleeding\b", r"\bhemorrhage\b", r"\bhaemorrhage\b"],
    "Cystic Duct": [r"\bcystic duct\b", r"\bcystic\b"],
    "L-hook Electrocautery": [r"\bhook\b", r"\belectrocaut\w*", r"\bcautery\b", r"\bdiathermy\b"],
    "Gallbladder": [r"\bgall ?bladder\b"],
    "Hepatic Vein": [r"\bhepatic vein\b"],
    "Liver Ligament": [r"\bliver ligament\b", r"\bfalciform\b", r"\bligament\b"],
}
_NEGATORS = {"no", "not", "without", "absence", "absent", "none", "neither", "nor"}
_COMPILED = {c: [re.compile(p, re.I) for p in ps] for c, ps in TERMS.items()}


def _negated(text, start):
    clause = re.split(r"[.;:!?,]", text[:start])[-1]
    return bool(set(re.findall(r"[a-z']+", clause.lower())[-4:]) & _NEGATORS)


def mentioned_classes(text):
    return {c for c, pats in _COMPILED.items()
            if any(not _negated(text, m.start()) for p in pats for m in p.finditer(text))}


def score_frame(text, coverage, min_fraction=0.005):
    present = set(present_classes(coverage, min_fraction))
    mention = mentioned_classes(text)
    return {"recall": len(mention & present) / len(present) if present else None,
            "absent_mentions": sorted(c for c in mention if coverage.get(c, 0) == 0),
            "unclear": text.strip().lower().rstrip(". ") == "unclear"}


# =============================== evaluation over a prepared split =================================================
def _load_jpeg(path, max_side=768):
    import cv2

    img = _imread(path)
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise DatasetError("could not encode frame")
    return buf.tobytes()


def _mean(values):
    return sum(values) / len(values) if values else None


def evaluate_split(describer, root, split, final=False, batch=6, min_fraction=0.005,
                   clock=lambda: datetime.now(timezone.utc)):
    from surgidoc.vision import Frame

    if split not in ("dev", "test"):
        raise ValueError("split must be 'dev' or 'test'")
    if split == "test" and not final:
        raise HeldOutSplitError("the test split is held out: pass final=True for a final report (every run is logged)")
    root = Path(root)
    refs = [FrameRef(f["video"], f["clip"], f["index"], f["split"]) for f in load_manifest(root)["frames"] if f["split"] == split]
    missing = [p for r in refs for p in (r.endo, r.mask) if not (root / p).is_file()]
    if missing:
        raise DatasetError(f"{len(missing)} file(s) are not downloaded yet (e.g. {missing[0]}): run the download/prepare step")
    frames = [Frame(i, 0.0, _load_jpeg(root / r.endo)) for i, r in enumerate(refs)]
    texts = {}
    for i in range(0, len(frames), batch):
        texts.update({f.frame: f.text for f in describer.describe(frames[i:i + batch])})
    rows = []
    for i, r in enumerate(refs):
        cov = frame_coverage(root / r.mask)
        rows.append({"video": r.video, "clip": r.clip, "index": r.index, "text": texts[i], "reference": present_classes(cov, min_fraction),
                     **score_frame(texts[i], cov, min_fraction)})
    per_class = {}
    for row in rows:
        for cls in row["reference"]:
            per_class.setdefault(cls, []).append(cls in mentioned_classes(row["text"]))
    n = len(rows)
    metrics = {"n": n, "mean_recall": _mean([r["recall"] for r in rows if r["recall"] is not None]),
               "absent_mention_rate": sum(bool(r["absent_mentions"]) for r in rows) / n if n else None,
               "unclear_rate": sum(r["unclear"] for r in rows) / n if n else None,
               "per_class_recall": {c: sum(v) / len(v) for c, v in sorted(per_class.items())}}
    if split == "test":  # a visible record of every look at the held-out videos
        log = root / "test_runs.json"
        runs = json.loads(log.read_text()) if log.is_file() else []
        runs.append({"at": clock().isoformat(), "n": n, "metrics": metrics})
        _atomic_write(log, json.dumps(runs, indent=1).encode())
    return {"split": split, "metrics": metrics, "frames": rows}
