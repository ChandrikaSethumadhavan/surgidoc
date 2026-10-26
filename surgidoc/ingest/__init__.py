"""FR-3: extract text from PDFs, images (OCR), transcripts and text; chunk; load public sources."""
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

TEXT = {".txt", ".md"}
TRANSCRIPT = {".vtt", ".srt"}
IMAGE = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_TS = re.compile(r"^\s*(\d+|[\d:.,]+\s*-->.*|WEBVTT.*)\s*$")


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    index: int


def _tesseract(path):  # lazy: needs the tesseract binary + pytesseract/Pillow
    import pytesseract
    from PIL import Image

    return pytesseract.image_to_string(Image.open(path))


def extract_text(path, ocr=_tesseract):
    path = Path(path)
    ext = path.suffix.lower()
    if ext in TEXT:
        return path.read_text(errors="ignore")
    if ext == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(p.extract_text() or "" for p in PdfReader(path).pages).strip()
    if ext in TRANSCRIPT:
        lines = [l.strip() for l in path.read_text(errors="ignore").splitlines()]
        return " ".join(l for l in lines if l and not _TS.match(l))
    if ext in IMAGE:
        return ocr(path)
    raise ValueError(f"unsupported file type: {ext}")


def chunk_text(text, max_chars=1000):
    # ponytail: paragraph/sentence boundaries, not embedding-based; upgrade if retrieval eval shows poor splits
    units = []
    for para in re.split(r"\n\s*\n", text.strip()):
        for sent in re.split(r"(?<=[.!?])\s+", para.strip()):
            units += [sent[i:i + max_chars] for i in range(0, len(sent), max_chars)] if sent else []
    chunks, cur = [], ""
    for u in units:
        if cur and len(cur) + 1 + len(u) > max_chars:
            chunks.append(cur)
            cur = u
        else:
            cur = f"{cur} {u}" if cur else u
    return chunks + [cur] if cur else chunks


def ingest_paths(paths, max_chars=1000, ocr=_tesseract, on_error=None):
    out = []
    for p in paths:
        try:
            pieces = chunk_text(extract_text(p, ocr=ocr), max_chars)
        except Exception as e:  # one bad file must not stop the corpus build
            if on_error:
                on_error(p, e)
                continue
            raise
        out += [Chunk(t, str(p), i) for i, t in enumerate(pieces)]
    return out


def _fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def download_public_texts(sources, dest, fetch=_fetch):
    """sources: {filename: url}. Saves into dest; filenames may not escape it."""
    dest = Path(dest).resolve()
    saved = []
    for name, url in sources.items():
        target = (dest / name).resolve()
        if dest not in target.parents:
            raise ValueError(f"unsafe filename: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fetch(url))
        saved.append(target)
    return saved


def fetch_cholecseg8k(hub=None):
    if hub is None:
        import kagglehub as hub
    return hub.dataset_download("newslab/cholecseg8k")
