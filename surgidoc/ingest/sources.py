"""Registry of public knowledge-base sources (FR-3). Public data only; check licences before adding.

`verified` means the URL was fetched successfully and the licence confirmed. Unverified entries are
never auto-downloaded. Checked 2026-10-26 from this dev machine.
"""
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlparse


@dataclass(frozen=True)
class Source:
    id: str
    kind: str  # text | image_video | kaggle
    url: str
    license: str
    language: str
    region: str
    verified: bool = False


SOURCES = [
    Source("cholecseg8k", "kaggle", "https://www.kaggle.com/datasets/newslab/cholecseg8k",
           "see dataset page", "n/a", "EU", verified=True),
    # CC-BY 4.0 per the dataset paper; OSF page reachable. The simula.no mirror was unreachable here.
    Source("hyperkvasir", "image_video", "https://osf.io/mh9sj/", "CC-BY 4.0", "n/a", "EU", verified=True),
    # AWMF PDF URLs returned 404/500 on 2026-10-26; confirm the current link and licence, then verify.
    Source("awmf-s3-sedation-2023", "text",
           "https://register.awmf.org/assets/guidelines/021-014l_S3_KF-Gastrointestinale-Endoskopie-Sedierung_2023-01.pdf",
           "unconfirmed", "de", "DE"),
]


def sync_text_sources(dest, sources=SOURCES, fetch=None):
    """Download verified text sources into dest. Returns (saved paths, [(source, error)])."""
    from surgidoc.ingest import _fetch, download_public_texts

    fetch = fetch or _fetch
    saved, failed = [], []
    for s in sources:
        if s.kind != "text" or not s.verified:
            continue
        name = s.id + (PurePosixPath(urlparse(s.url).path).suffix or ".txt")
        try:
            saved += download_public_texts({name: s.url}, dest, fetch=fetch)
        except Exception as e:  # one dead link must not stop the sync
            failed.append((s, e))
    return saved, failed
