import re

import pytest

from surgidoc.ingest import SOURCES, Source, sync_text_sources


def test_registry_entries_are_well_formed():
    assert SOURCES
    assert len({s.id for s in SOURCES}) == len(SOURCES)
    for s in SOURCES:
        assert re.fullmatch(r"[a-z0-9_\-]+", s.id)
        assert s.url.startswith("https://")
        assert s.license and s.kind in {"text", "image_video", "kaggle"}


def test_registry_covers_required_kinds_and_regions():
    kinds = {s.kind for s in SOURCES}
    assert {"text", "image_video", "kaggle"} <= kinds
    assert any(s.kind == "text" and s.region in {"DE", "EU"} for s in SOURCES)
    assert any(s.id == "cholecseg8k" for s in SOURCES)


def test_unverified_sources_are_never_auto_downloaded(tmp_path):
    src = [Source("a", "text", "https://x/a.pdf", "CC-BY", "DE", "DE", verified=False)]
    saved, failed = sync_text_sources(tmp_path, sources=src, fetch=lambda u: b"x")
    assert saved == [] and failed == []


def test_sync_downloads_verified_text_only(tmp_path):
    src = [
        Source("a", "text", "https://x/a.pdf", "CC-BY", "DE", "DE", verified=True),
        Source("v", "image_video", "https://x/v", "CC-BY", "en", "EU", verified=True),
    ]
    saved, failed = sync_text_sources(tmp_path, sources=src, fetch=lambda u: b"data")
    assert [p.name for p in saved] == ["a.pdf"] and failed == []
    assert saved[0].read_bytes() == b"data"


def test_sync_reports_failures_and_continues(tmp_path):
    src = [
        Source("bad", "text", "https://x/bad.pdf", "CC-BY", "DE", "DE", verified=True),
        Source("ok", "text", "https://x/ok.pdf", "CC-BY", "DE", "DE", verified=True),
    ]

    def fetch(url):
        if "bad" in url:
            raise OSError("404")
        return b"ok"

    saved, failed = sync_text_sources(tmp_path, sources=src, fetch=fetch)
    assert [p.name for p in saved] == ["ok.pdf"]
    assert [s.id for s, _ in failed] == ["bad"]


def test_sync_includes_extension_from_url(tmp_path):
    src = [Source("g", "text", "https://x/guide.html", "CC-BY", "en", "EU", verified=True)]
    saved, _ = sync_text_sources(tmp_path, sources=src, fetch=lambda u: b"x")
    assert saved[0].name == "g.html"


def test_esge_sources_are_verified_europepmc_with_cc_licence():
    from surgidoc.ingest import SOURCES

    esge = [s for s in SOURCES if s.id.startswith("esge-")]
    assert len(esge) >= 5
    for s in esge:
        assert s.kind == "text" and s.verified and s.region == "EU" and s.language == "en"
        assert s.url.startswith("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC")
        assert s.license.startswith("CC BY")


def test_esge_filename_is_xml(tmp_path):
    src = [Source("esge-x", "text", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML",
                  "CC BY-NC-ND", "en", "EU", verified=True, ext=".xml")]
    saved, _ = sync_text_sources(tmp_path, sources=src, fetch=lambda u: b"<a/>")
    assert saved[0].name == "esge-x.xml"


def test_image_video_entries_present_but_not_blindly_verified():
    from surgidoc.ingest import SOURCES

    by_id = {s.id: s for s in SOURCES}
    assert by_id["cholect45"].license.startswith("CC BY-NC-SA")
    assert not by_id["endovis"].verified and not by_id["camma-datasets"].verified


def test_mimic_is_rejected_and_never_in_registry():
    from surgidoc.ingest import REJECTED, SOURCES

    assert "mimic-iv-note" in REJECTED and "patient" in REJECTED["mimic-iv-note"].lower()
    assert not any("mimic" in s.id or "physionet" in s.url for s in SOURCES)
