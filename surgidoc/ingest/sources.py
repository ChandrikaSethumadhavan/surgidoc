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
    ext: str = ""  # override when the URL has no file suffix


SOURCES = [
    Source("cholecseg8k", "kaggle", "https://www.kaggle.com/datasets/newslab/cholecseg8k",
           "see dataset page", "n/a", "EU", verified=True),
    # CC-BY 4.0 per the dataset paper; OSF page reachable. The simula.no mirror was unreachable here.
    Source("hyperkvasir", "image_video", "https://osf.io/mh9sj/", "CC-BY 4.0", "n/a", "EU", verified=True),
    # AWMF PDF URLs returned 404/500 on 2026-10-26; confirm the current link and licence, then verify.
    Source("awmf-s3-sedation-2023", "text",
           "https://register.awmf.org/assets/guidelines/021-014l_S3_KF-Gastrointestinale-Endoskopie-Sedierung_2023-01.pdf",
           "unconfirmed", "de", "DE"),
    # ESGE guidelines via Europe PMC full-text XML; fetched and licence read from the API on 2026-10-26.
    # NC/ND licences: fine for a non-commercial prototype with citations, not for republishing or commercial use.
    Source("esge-polypectomy-emr-2022", "text", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9666068/fullTextXML",
           "CC BY-NC-ND", "en", "EU", verified=True, ext=".xml"),
    Source("esge-variceal-bleeding-2020", "text", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7329372/fullTextXML",
           "CC BY-NC-ND", "en", "EU", verified=True, ext=".xml"),
    Source("esge-nonvariceal-ugib-2018", "text", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6175685/fullTextXML",
           "CC BY-NC-ND", "en", "EU", verified=True, ext=".xml"),
    Source("esge-oesophageal-stenting-2019", "text", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6561763/fullTextXML",
           "CC BY-NC-ND", "en", "EU", verified=True, ext=".xml"),
    Source("esge-ct-colonography-2015", "text", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4291518/fullTextXML",
           "CC BY-NC", "en", "EU", verified=True, ext=".xml"),
    Source("esge-covid-cascade-2021", "text", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7969134/fullTextXML",
           "CC BY-NC-ND", "en", "EU", verified=True, ext=".xml"),
    # CholecT45 repo reachable, CC BY-NC-SA 4.0 (non-commercial). Triplet labels, no report text.
    Source("cholect45", "image_video", "https://github.com/CAMMA-public/cholect45", "CC BY-NC-SA 4.0", "n/a", "EU", verified=True),
    # Landing pages with per-dataset licences: check each dataset before use.
    Source("camma-datasets", "image_video", "https://camma.unistra.fr/datasets", "varies per dataset", "n/a", "EU"),
    Source("endovis", "image_video", "https://opencas.dkfz.de/endovis/datasetspublications", "varies per dataset", "n/a", "EU"),
]

REJECTED = {
    "mimic-iv-note": "Credentialed access under a data use agreement; real (de-identified) patient clinical notes, "
                     "which breaks the public-data-only decision. US ICU notes, not endoscopy or EU.",
}


def sync_text_sources(dest, sources=SOURCES, fetch=None):
    """Download verified text sources into dest. Returns (saved paths, [(source, error)])."""
    from surgidoc.ingest import _fetch, download_public_texts

    fetch = fetch or _fetch
    saved, failed = [], []
    for s in sources:
        if s.kind != "text" or not s.verified:
            continue
        name = s.id + (s.ext or PurePosixPath(urlparse(s.url).path).suffix or ".txt")
        try:
            saved += download_public_texts({name: s.url}, dest, fetch=fetch)
        except Exception as e:  # one dead link must not stop the sync
            failed.append((s, e))
    return saved, failed
