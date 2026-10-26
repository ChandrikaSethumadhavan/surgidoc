"""FR-3: extract text from PDFs, images (OCR), transcripts and text; chunk; load public sources."""
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

TEXT = {".txt", ".md"}
TRANSCRIPT = {".vtt", ".srt"}
IMAGE = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_TS = re.compile(r"^\s*(\d+|[\d:.,]+\s*-->.*|WEBVTT.*)\s*$") #_TS is a regex used to ignore subtitle/timing lines like “00:00:01.000 --> 00:00:02.000” and the “WEBVTT” header


@dataclass(frozen=True)
class Chunk:
    text: str  #chunk's content
    source: str #the file path it came from
    index: int #the chunk's position in the original file.


def _tesseract(path):  # lazy: needs the tesseract binary + pytesseract/Pillow  this is a helper for OCR
    import pytesseract
    from PIL import Image

    return pytesseract.image_to_string(Image.open(path))  #In other words: “open this image file and convert it to text from the visual content”


def extract_text(path, ocr=_tesseract):
    """This is the file-type normalization step. Without this, downstream chunking would only work on plain text."""
#     If it is a PDF:

# It imports pypdf
# Uses PdfReader(path).pages
# Joins text from each page with blank lines
# strips whitespace around it
# If it is a transcript file (.vtt or .srt):

# Reads the raw text
# Splits by lines
# strips spaces
# ignores empty lines and subtitle timestamp/header lines
# joins the remaining transcript text into a single string
# If it is an image:

# calls the OCR function passed in (default: _tesseract)
# Otherwise:

# raises ValueError with “unsupported file type”
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
    """It does this in stages:
Split by paragraph breaks: re.split(r"\n\s*\n", text.strip())
Then split each paragraph by sentence boundary using punctuation like . ! ?
For each sentence, it slices it into smaller segments of length max_chars
Then it builds chunk strings while enforcing a max length, So it is structure-aware text chunking, but still rule-based and 
lexical, not meaning-based. It groups by readability boundaries, not by topic/semantic similarity."""

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
    """This is the top-level bulk ingestion function.
It accepts a list of file paths.
For each path:
calls extract_text to convert the file to text
calls chunk_text to split it into chunks
turns each chunk into a Chunk object with source path and index
appends all chunks to a list
Errors:
if any file fails:
if on_error is provided, it calls on_error(p, e) and continues
otherwise, it re-raises the exception and stops
Return:
a list of Chunk objects"""

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


from .sources import SOURCES, Source, sync_text_sources  # noqa: E402
