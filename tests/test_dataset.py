import hashlib
import json
from datetime import datetime, timezone

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from surgidoc.dataset import (CLASSES, VIDEOS, DatasetError, Downloader, FrameRef, HeldOutSplitError, assign_splits,  # noqa: E402
                              evaluate_split, frame_coverage, kaggle_tree, prepare, kagglehub_fetch, list_dataset, load_manifest,
                              mentioned_classes, present_classes, sample_frames, score_frame)
from surgidoc.vision import Finding  # noqa: E402


def png(array):
    ok, buf = cv2.imencode(".png", array)
    assert ok
    return buf.tobytes()


def mask_file(tmp_path, fractions, size=100, name="mask.png"):
    p = tmp_path / name
    p.write_bytes(mask_png(fractions, size))
    return p


def mask_png(fractions, size=100):
    """Watershed-style mask: background 50 everywhere except horizontal bands covering the given pixel fractions."""
    m = np.full((size, size, 3), 50, dtype=np.uint8)
    row = 0
    for value, frac in fractions.items():
        rows = round(frac * size)
        m[row:row + rows, :, :] = value
        row += rows
    return png(m)


# =============================== split =========================================================
def test_split_is_deterministic_and_partitions_the_videos():
    a, b = assign_splits(), assign_splits()
    assert a == b and set(a) == set(VIDEOS)
    assert sorted(v for v in a.values()) == ["dev"] * 3 + ["test"] * 3 + ["unused"] * 11


def test_split_does_not_depend_on_input_order():
    assert assign_splits(list(reversed(VIDEOS))) == assign_splits(VIDEOS)


def test_split_changes_with_the_seed_and_keeps_dev_and_test_disjoint():
    a, b = assign_splits(seed="a"), assign_splits(seed="b")
    assert a != b
    for s in (a, b):
        dev = {v for v, k in s.items() if k == "dev"}
        test = {v for v, k in s.items() if k == "test"}
        assert dev and test and not dev & test


def test_split_sizes_are_validated():
    with pytest.raises(ValueError):
        assign_splits(VIDEOS, n_dev=10, n_test=10)
    with pytest.raises(ValueError):
        assign_splits(VIDEOS, n_dev=0, n_test=3)


def test_known_video_list_matches_the_dataset_description():
    assert len(VIDEOS) == 17 and len(set(VIDEOS)) == 17 and VIDEOS[0] == "video01"


# =============================== listing and sampling ============================================
class FakeTree:
    """tree(path, token) -> (dirs, files, next_token), paginated two entries at a time."""

    def __init__(self, structure, page=2):
        self.structure, self.page, self.calls = structure, page, []

    def __call__(self, path, token=None):
        self.calls.append((path, token))
        dirs, files = self.structure[path]
        entries = [("d", d) for d in dirs] + [("f", f) for f in files]
        start = int(token or 0)
        chunk = entries[start:start + self.page]
        nxt = str(start + self.page) if start + self.page < len(entries) else None
        return [n for k, n in chunk if k == "d"], [n for k, n in chunk if k == "f"], nxt


def frame_files(indices):
    return [f"frame_{i}_{suffix}.png" for i in indices for suffix in ("endo", "endo_color_mask", "endo_mask", "endo_watershed_mask")]


def structure(videos=("video01", "video09"), clips=(80, 160, 240), frames=range(1, 81)):
    s = {"": (list(videos), [])}
    for v in videos:
        s[v] = ([f"{v}_{c:05d}" for c in clips], [])
        for c in clips:
            s[f"{v}/{v}_{c:05d}"] = ([], frame_files([c + i for i in frames]))
    return s


def test_listing_follows_pagination_and_parses_frame_indices():
    tree = FakeTree(structure(videos=("video01",), clips=(80, 160), frames=range(1, 6)), page=3)
    out = list_dataset(tree, ["video01"])
    assert out == {"video01": {"video01_00080": [81, 82, 83, 84, 85], "video01_00160": [161, 162, 163, 164, 165]}}
    assert any(token for _, token in tree.calls)  # more than one page was requested


def test_listing_ignores_non_endo_files_and_unknown_videos_raise():
    tree = FakeTree({"": (["video01"], []), "video01": (["video01_00080"], []),
                     "video01/video01_00080": ([], ["frame_1_endo.png", "frame_1_endo_mask.png", "notes.txt"])})
    assert list_dataset(tree, ["video01"]) == {"video01": {"video01_00080": [1]}}
    with pytest.raises(DatasetError, match="video99"):
        list_dataset(tree, ["video99"])


def listing():
    return list_dataset(FakeTree(structure(), page=50), ["video01", "video09"])


def test_sampling_is_deterministic_spread_and_capped_per_video():
    splits = {"video01": "dev", "video09": "test"}
    a = sample_frames(listing(), splits, per_video=12, clips_per_video=3)
    assert a == sample_frames(listing(), splits, per_video=12, clips_per_video=3)
    for video in splits:
        ours = [r for r in a if r.video == video]
        assert len(ours) == 12 and {r.split for r in ours} == {splits[video]}
        assert len({r.clip for r in ours}) == 3  # clips are spread across the video
        clip = [r.index for r in ours if r.clip.endswith("00080")]
        assert all(b - a_ >= 5 for a_, b in zip(clip, clip[1:]))  # not neighbouring near-duplicates


def test_sampling_skips_unused_videos_and_handles_short_videos():
    splits = {"video01": "dev", "video09": "unused"}
    out = sample_frames(listing(), splits, per_video=6, clips_per_video=5)
    assert {r.video for r in out} == {"video01"} and len(out) == 6


def test_sampling_never_returns_duplicates_or_more_than_exist():
    tiny = {"video01": {"video01_00080": [81, 82]}}
    out = sample_frames(tiny, {"video01": "dev"}, per_video=12, clips_per_video=3)
    assert sorted(r.index for r in out) == [81, 82]


def test_frame_ref_names_and_remote_paths():
    r = FrameRef("video01", "video01_00080", 100, "dev")
    assert r.endo == "video01/video01_00080/frame_100_endo.png"
    assert r.mask == "video01/video01_00080/frame_100_endo_watershed_mask.png"


@pytest.mark.parametrize("bad", [("..", "c", 1), ("v", "../x", 1), ("v/w", "c", 1), ("v", "c", -1)])
def test_frame_ref_rejects_path_tricks(bad):
    with pytest.raises(ValueError):
        FrameRef(*bad, "dev")


# =============================== downloader ===============================================================
class Fetcher:
    def __init__(self, fail=None, content=None):
        self.calls, self.fail, self.content = [], dict(fail or {}), content

    def __call__(self, remote):
        self.calls.append(remote)
        err = self.fail.get(remote)
        if isinstance(err, int):  # transient: fail this many times
            if self.fail[remote] > 0:
                self.fail[remote] -= 1
                raise OSError("network blip")
        elif err is not None:
            raise err
        if self.content is not None:
            return self.content
        return mask_png({31: 0.2}) if remote.endswith("watershed_mask.png") else png(np.full((8, 8, 3), 7, np.uint8))


REFS = [FrameRef("video01", "video01_00080", i, "dev") for i in (81, 120)] + [FrameRef("video09", "video09_00160", 200, "test")]


def downloader(tmp_path, fetch=None, **kw):
    sleeps = []
    return Downloader(fetch or Fetcher(), tmp_path / "data", sleep=sleeps.append, **kw), sleeps


def test_downloads_images_and_masks_and_writes_a_checksummed_manifest(tmp_path):
    d, _ = downloader(tmp_path)
    summary = d.download(REFS)
    assert summary == {"downloaded": 6, "skipped": 0, "failed": [], "remaining": 0}
    m = load_manifest(tmp_path / "data")
    assert len(m["frames"]) == 3 and {f["split"] for f in m["frames"]} == {"dev", "test"}
    for rel, info in m["files"].items():
        data = (tmp_path / "data" / rel).read_bytes()
        assert info["bytes"] == len(data) and info["sha256"] == hashlib.sha256(data).hexdigest()


def test_rerun_skips_existing_valid_files(tmp_path):
    fetch = Fetcher()
    d, _ = downloader(tmp_path, fetch)
    d.download(REFS)
    n = len(fetch.calls)
    assert d.download(REFS) == {"downloaded": 0, "skipped": 6, "failed": [], "remaining": 0} and len(fetch.calls) == n


def test_a_crash_mid_download_loses_nothing_and_resumes(tmp_path):
    fetch = Fetcher(fail={REFS[1].endo: KeyboardInterrupt()})
    d, _ = downloader(tmp_path, fetch)
    with pytest.raises(KeyboardInterrupt):
        d.download(REFS)
    root = tmp_path / "data"
    assert not list(root.rglob("*.part"))  # no half-written files
    done = [p for p in root.rglob("*.png")]
    assert len(done) == 2  # the first frame's image and mask were saved before the interrupt on the second frame
    fetch2 = Fetcher()
    d2, _ = downloader(tmp_path, fetch2)
    summary = d2.download(REFS)
    assert summary["failed"] == [] and summary["downloaded"] + summary["skipped"] == 6 and summary["skipped"] >= 2
    assert len(fetch2.calls) == summary["downloaded"]


def test_transient_errors_are_retried_with_backoff(tmp_path):
    fetch = Fetcher(fail={REFS[0].endo: 2})
    d, sleeps = downloader(tmp_path, fetch, retries=3)
    assert d.download(REFS[:1])["failed"] == [] and fetch.calls.count(REFS[0].endo) == 3
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]


def test_permanent_failures_are_reported_and_do_not_stop_other_files(tmp_path):
    fetch = Fetcher(fail={REFS[0].endo: 99})
    d, _ = downloader(tmp_path, fetch, retries=2)
    s = d.download(REFS)
    assert s["failed"] == [REFS[0].endo] and s["downloaded"] == 5
    assert not (tmp_path / "data" / REFS[0].endo).exists()


def test_a_write_that_dies_halfway_never_leaves_a_final_file(tmp_path, monkeypatch):
    from pathlib import Path

    real = Path.write_bytes

    def dying(self, data):
        if self.name.endswith((".png", ".png.part")):  # only image writes die, not the manifest
            real(self, data[: len(data) // 2])  # half the bytes land on disk, then the process "dies"
            raise OSError("disk error")
        return real(self, data)

    monkeypatch.setattr(Path, "write_bytes", dying)
    d, _ = downloader(tmp_path, retries=1)
    with pytest.raises(OSError):
        d.download(REFS[:1])
    monkeypatch.undo()
    assert not list((tmp_path / "data").rglob("*.png"))  # only a .part leftover, never a truncated final file
    d2, _ = downloader(tmp_path)
    assert d2.download(REFS[:1])["downloaded"] == 2 and not list((tmp_path / "data").rglob("*.part"))


def test_non_png_content_is_rejected_and_never_written(tmp_path):
    d, _ = downloader(tmp_path, Fetcher(content=b"<html>login page</html>"), retries=1)
    s = d.download(REFS[:1])
    assert len(s["failed"]) == 2 and not list((tmp_path / "data").rglob("*.png"))


def test_corrupt_existing_files_are_downloaded_again(tmp_path):
    d, _ = downloader(tmp_path)
    d.download(REFS[:1])
    target = tmp_path / "data" / REFS[0].endo
    target.write_bytes(b"truncated")
    fetch = Fetcher()
    d2, _ = downloader(tmp_path, fetch)
    assert d2.download(REFS[:1]) == {"downloaded": 1, "skipped": 1, "failed": [], "remaining": 0}
    assert target.read_bytes().startswith(b"\x89PNG")


def test_max_files_bounds_one_call_and_reports_what_remains(tmp_path):
    d, _ = downloader(tmp_path)
    s = d.download(REFS, max_files=4)
    assert s["downloaded"] == 4 and s["remaining"] == 2
    s2 = d.download(REFS, max_files=4)
    assert s2["downloaded"] == 2 and s2["skipped"] == 4 and s2["remaining"] == 0


def test_downloads_stay_inside_the_destination(tmp_path):
    d, _ = downloader(tmp_path)
    d.download(REFS)
    root = (tmp_path / "data").resolve()
    assert all(root in p.resolve().parents for p in root.rglob("*.png"))


# =============================== reference labels from masks ==================================================
def test_coverage_maps_pixel_values_to_the_official_class_names(tmp_path):
    cov = frame_coverage(mask_file(tmp_path, {31: 0.2, 22: 0.3, 24: 0.001}))
    assert cov["Grasper"] == pytest.approx(0.2) and cov["Gallbladder"] == pytest.approx(0.3)
    assert cov["Black Background"] == pytest.approx(0.5, abs=0.01) and "Liver" not in cov or cov["Liver"] == 0


def test_present_classes_apply_a_minimum_share_and_ignore_background(tmp_path):
    cov = frame_coverage(mask_file(tmp_path, {31: 0.2, 24: 0.002}, size=1000))  # tall enough for a 0.2% band to exist
    assert set(present_classes(cov, min_fraction=0.005)) == {"Grasper"}
    assert set(present_classes(cov, min_fraction=0.001)) == {"Grasper", "Blood"}


def test_class_table_matches_the_dataset_description():
    assert len(CLASSES) == 13 and CLASSES[31] == "Grasper" and CLASSES[32] == "L-hook Electrocautery"
    assert CLASSES[50] == "Black Background" and CLASSES[5] == "Liver Ligament" and CLASSES[33] == "Hepatic Vein"


def test_unknown_pixel_values_or_bad_png_raise_clear_errors(tmp_path):
    with pytest.raises(DatasetError, match="unknown"):
        frame_coverage(mask_file(tmp_path, {77: 0.5}))
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    with pytest.raises(DatasetError):
        frame_coverage(bad)
    with pytest.raises(DatasetError):
        frame_coverage(tmp_path / "missing.png")


# =============================== mentions and per-frame scoring ===================================================
@pytest.mark.parametrize("text,expected", [
    ("A grasper holds the gallbladder.", {"Grasper", "Gallbladder"}),
    ("Electrocautery hook dissecting fat near the liver.", {"L-hook Electrocautery", "Fat", "Liver"}),
    ("No active bleeding is visible.", set()),
    ("Active bleeding is seen near the cystic duct.", {"Blood", "Cystic Duct"}),
    ("The scene is without blood and there is no grasper.", set()),
    ("Forceps retract the abdominal wall; liver ligament visible.", {"Grasper", "Abdominal Wall", "Liver Ligament"}),
    ("Delivery of fatigue measures.", set()),  # whole words only
    ("unclear", set()),
])
def test_mentions_use_whole_words_synonyms_and_simple_negation(text, expected):
    assert mentioned_classes(text) == expected


def test_score_frame_reports_recall_absent_mentions_and_unclear(tmp_path):
    cov = frame_coverage(mask_file(tmp_path, {31: 0.2, 22: 0.3}))
    s = score_frame("A grasper holds the gallbladder; the liver is visible.", cov)
    assert s["recall"] == 1.0 and s["absent_mentions"] == ["Liver"] and s["unclear"] is False
    s = score_frame("A grasper is visible.", cov)
    assert s["recall"] == 0.5 and s["absent_mentions"] == []
    s = score_frame("Unclear.", cov)
    assert s["recall"] == 0.0 and s["unclear"] is True


def test_score_frame_without_reference_structures_has_no_recall(tmp_path):
    assert score_frame("Nothing notable.", frame_coverage(mask_file(tmp_path, {})))["recall"] is None


# =============================== evaluation over a prepared split ==================================================
class Env:
    pass


def prepared(tmp_path):
    """4 dev frames + 2 test frames on disk with known masks, downloaded through the real Downloader."""
    masks = {81: {31: 0.2, 22: 0.3}, 120: {21: 0.4, 24: 0.1}, 160: {32: 0.2}, 200: {}}
    refs = [FrameRef("video01", "video01_00080", i, "dev") for i in (81, 120)] + \
           [FrameRef("video09", "video09_00160", i, "dev") for i in (160, 200)] + \
           [FrameRef("video12", "video12_00240", i, "test") for i in (300, 301)]
    masks.update({300: {23: 0.3}, 301: {31: 0.1, 25: 0.2}})

    def fetch(remote):
        if remote.endswith("watershed_mask.png"):
            idx = int(remote.rsplit("frame_", 1)[1].split("_")[0])
            return mask_png(masks[idx])
        return png(np.full((16, 16, 3), 90, np.uint8))

    Downloader(fetch, tmp_path / "data", sleep=lambda s: None).download(refs)
    return tmp_path / "data"


class FakeDescriber:
    def __init__(self, answers):
        self.answers, self.seen = answers, []

    def describe(self, frames):
        self.seen += [f.index for f in frames]
        return [Finding(f.index, f.t, self.answers[f.index], "gemini-vision") for f in frames]


def run_dev(tmp_path, answers, **kw):
    root = prepared(tmp_path)
    d = FakeDescriber(answers)
    return evaluate_split(d, root, "dev", clock=lambda: datetime(2026, 10, 27, tzinfo=timezone.utc), **kw), d, root


DEV_ANSWERS = ["A grasper holds the gallbladder.", "Liver with some bleeding.", "Unclear.", "Fat and the liver."]


def test_dev_evaluation_aggregates_recall_absent_mentions_unclear_and_per_class(tmp_path):
    out, d, _ = run_dev(tmp_path, {0: DEV_ANSWERS[0], 1: DEV_ANSWERS[1], 2: DEV_ANSWERS[2], 3: DEV_ANSWERS[3]})
    m = out["metrics"]
    assert m["n"] == 4 and len(out["frames"]) == 4
    # frames: 81 -> recall 1.0 ; 120 (liver+blood) -> 1.0 ; 160 (hook) unclear -> 0 ; 200 (no classes) -> excluded
    assert m["mean_recall"] == pytest.approx((1.0 + 1.0 + 0.0) / 3)
    assert m["unclear_rate"] == pytest.approx(0.25)
    assert m["absent_mention_rate"] == pytest.approx(0.25)  # only the "Fat and the liver" frame mentions absent classes
    assert m["per_class_recall"]["Grasper"] == 1.0 and m["per_class_recall"]["L-hook Electrocautery"] == 0.0


def test_evaluation_batches_frames_to_the_describer(tmp_path):
    root = prepared(tmp_path)

    class Batches(FakeDescriber):
        sizes = []

        def describe(self, frames):
            Batches.sizes.append(len(frames))
            return super().describe(frames)

    d = Batches({i: "x" for i in range(4)})
    evaluate_split(d, root, "dev", batch=3)
    assert Batches.sizes == [3, 1]


def test_the_test_split_needs_an_explicit_final_flag_and_every_run_is_logged(tmp_path):
    root = prepared(tmp_path)
    d = FakeDescriber({0: "A grasper.", 1: "Connective tissue and cystic duct."})
    with pytest.raises(HeldOutSplitError):
        evaluate_split(d, root, "test")
    assert d.seen == [] and not (root / "test_runs.json").exists()
    clock = lambda: datetime(2026, 10, 27, 12, tzinfo=timezone.utc)  # noqa: E731
    out = evaluate_split(d, root, "test", final=True, clock=clock)
    evaluate_split(FakeDescriber({0: "x", 1: "y"}), root, "test", final=True, clock=clock)
    runs = json.loads((root / "test_runs.json").read_text())
    assert len(runs) == 2 and runs[0]["metrics"] == out["metrics"] and runs[0]["at"].startswith("2026-10-27T12")


def test_dev_runs_are_not_logged_as_test_runs(tmp_path):
    _, _, root = run_dev(tmp_path, {i: "x" for i in range(4)})
    assert not (root / "test_runs.json").exists()


def test_missing_local_files_give_an_actionable_error(tmp_path):
    root = prepared(tmp_path)
    for p in list(root.rglob("frame_81_*")):
        p.unlink()
    with pytest.raises(DatasetError, match="prepare|download"):
        evaluate_split(FakeDescriber({}), root, "dev")


def test_unknown_split_and_missing_manifest_are_errors(tmp_path):
    with pytest.raises(DatasetError):
        evaluate_split(FakeDescriber({}), tmp_path / "nothing", "dev")
    with pytest.raises(ValueError):
        evaluate_split(FakeDescriber({}), prepared(tmp_path), "unused")


def test_evaluation_sends_downsized_jpeg_frames_to_the_describer(tmp_path):
    root = prepared(tmp_path)
    big = root / "video01" / "video01_00080" / "frame_81_endo.png"
    big.write_bytes(png(np.full((1000, 1400, 3), 120, np.uint8)))

    class Capture(FakeDescriber):
        frames = []

        def describe(self, frames):
            Capture.frames += list(frames)
            return super().describe(frames)

    evaluate_split(Capture({i: "x" for i in range(4)}), root, "dev")
    f = Capture.frames[0]
    img = cv2.imdecode(np.frombuffer(f.jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert f.jpeg[:2] == b"\xff\xd8" and max(img.shape[:2]) <= 768


# =============================== real fetch adapter (fake hub) ===================================================
class Hub:
    def __init__(self, tmp_path):
        self.tmp, self.calls = tmp_path, []
        self.auth = self

    def set_kaggle_api_token(self, token):
        self.calls.append("login")

    def whoami(self, verbose=True):
        return {"username": "alice"}

    def dataset_download(self, slug, path=None):
        self.calls.append((slug, path))
        f = self.tmp / "cache" / path.replace("/", "_")
        f.parent.mkdir(exist_ok=True)
        f.write_bytes(b"\x89PNG-bytes")
        return str(f)


def test_kagglehub_fetch_logs_in_once_downloads_one_file_and_removes_the_cache_copy(tmp_path):
    from surgidoc.config import load_keys

    k = tmp_path / "keys.txt"
    k.write_text("KAGGLE_KEY=KGAT_" + "a" * 32 + "\n")
    hub = Hub(tmp_path)
    fetch = kagglehub_fetch(load_keys(k), hub=hub)
    assert fetch("video01/video01_00080/frame_100_endo.png") == b"\x89PNG-bytes"
    assert fetch("video01/video01_00080/frame_101_endo.png") == b"\x89PNG-bytes"
    assert hub.calls.count("login") == 1
    assert hub.calls[1] == ("newslab/cholecseg8k", "video01/video01_00080/frame_100_endo.png")
    assert list((tmp_path / "cache").iterdir()) == []


# =============================== real listing over REST (fake urlopen) =============================================
class Resp:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def read(self, *a):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def listing_keys(tmp_path, text):
    from surgidoc.config import load_keys

    p = tmp_path / "keys.txt"
    p.write_text(text)
    return load_keys(p)


PAGE = {"directories": [{"name": "video01_00080", "relativeUrl": "video01/video01_00080"}],
        "files": [{"name": "frame_100_endo.png"}], "nextPageToken": "tok2"}


def test_kaggle_tree_calls_the_list_tree_endpoint_with_bearer_auth_and_parses_names(tmp_path):
    seen = []

    def urlopen(req, timeout=None):
        seen.append(req)
        return Resp(PAGE)

    tree = kaggle_tree(listing_keys(tmp_path, "KAGGLE_KEY=KGAT_" + "a" * 32 + "\n"), urlopen=urlopen)
    assert tree("video01", "tok1") == (["video01_00080"], ["frame_100_endo.png"], "tok2")
    req = seen[0]
    assert req.full_url.startswith("https://www.kaggle.com/api/v1/datasets/list-tree/newslab/cholecseg8k?")
    assert "path=video01" in req.full_url and "pageToken=tok1" in req.full_url and "pageSize=200" in req.full_url
    assert req.get_header("Authorization") == "Bearer KGAT_" + "a" * 32


def test_kaggle_tree_without_a_page_token_and_with_the_last_page(tmp_path):
    seen = []

    def urlopen(req, timeout=None):
        seen.append(req.full_url)
        return Resp({"directories": [], "files": []})

    tree = kaggle_tree(listing_keys(tmp_path, "KAGGLE_API_TOKEN=KGAT_" + "b" * 32 + "\n"), urlopen=urlopen)
    assert tree("", None) == ([], [], None) and "pageToken" not in seen[0]


def test_kaggle_tree_uses_basic_auth_for_a_legacy_pair(tmp_path):
    import base64

    seen = []
    kaggle_tree(listing_keys(tmp_path, "KAGGLE_USERNAME=alice\nKAGGLE_KEY=0123456789abcdef\n"),
                urlopen=lambda req, timeout=None: seen.append(req) or Resp(PAGE))("", None)
    assert seen[0].get_header("Authorization") == "Basic " + base64.b64encode(b"alice:0123456789abcdef").decode()


def test_kaggle_tree_without_credentials_or_on_http_errors_raises_without_leaking_the_token(tmp_path):
    with pytest.raises(DatasetError, match="KAGGLE_API_TOKEN"):
        kaggle_tree(listing_keys(tmp_path, "X=1\n"))
    token = "KGAT_" + "c" * 32

    def boom(req, timeout=None):
        raise OSError(f"401 for {token}")

    tree = kaggle_tree(listing_keys(tmp_path, f"KAGGLE_KEY={token}\n"), urlopen=boom)
    with pytest.raises(DatasetError) as e:
        tree("video01", None)
    assert token not in str(e.value) and e.value.__suppress_context__


# =============================== prepare: list -> split -> sample -> download ======================================
def prepare_structure():
    return structure(videos=tuple(VIDEOS), clips=(80, 160, 240), frames=range(1, 81))


def test_prepare_downloads_only_dev_and_test_videos_with_the_right_counts(tmp_path):
    tree, fetch = FakeTree(prepare_structure(), page=500), Fetcher()
    summary = prepare(tmp_path / "data", tree=tree, fetch=fetch, per_video=4, clips_per_video=2, sleep=lambda s: None)
    m = load_manifest(tmp_path / "data")
    splits = assign_splits()
    assert len(m["frames"]) == 6 * 4 and summary["downloaded"] == 6 * 4 * 2 and summary["failed"] == []
    assert {f["video"] for f in m["frames"]} == {v for v, k in splits.items() if k != "unused"}
    assert {f["split"] for f in m["frames"]} == {"dev", "test"} and all(splits[f["video"]] == f["split"] for f in m["frames"])
    unused_paths = [c for c in tree.calls if c[0].split("/")[0] in {v for v, k in splits.items() if k == "unused"}]
    assert unused_paths == []  # the unused videos are never even listed


def test_prepare_is_resumable_and_bounded(tmp_path):
    kw = dict(tree=FakeTree(prepare_structure(), page=500), per_video=4, clips_per_video=2, sleep=lambda s: None)
    first = prepare(tmp_path / "data", fetch=Fetcher(), max_files=10, **kw)
    assert first["downloaded"] == 10 and first["remaining"] == 6 * 4 * 2 - 10
    fetch2 = Fetcher()
    second = prepare(tmp_path / "data", fetch=fetch2, **kw)
    assert second["skipped"] == 10 and second["downloaded"] == 6 * 4 * 2 - 10 and len(fetch2.calls) == second["downloaded"]


def test_class_values_match_the_decimal_numbers_observed_in_a_real_mask(tmp_path):
    """Pinned from a real frame (video01_00080/frame_80): 21 liver 51%, 50 background 17%, 22 gallbladder 17%, 12 fat 7%,
    13 GI tract 6.5%, 255 boundary 0.6%. The description's '#505050' column is decimal 50, not hex 0x50."""
    import numpy as np

    img = np.full((1000, 1000, 3), 21, np.uint8)
    img[:172] = 50
    img[172:343] = 22
    img[343:414] = 12
    img[414:479] = 13
    img[479:485] = 255
    p = tmp_path / "real_like.png"
    p.write_bytes(png(img))
    cov = frame_coverage(p)
    assert set(cov) == {"Liver", "Black Background", "Gallbladder", "Fat", "Gastrointestinal Tract"}
    assert cov["Gallbladder"] == pytest.approx(0.171, abs=0.001) and 255 not in CLASSES  # boundary pixels are ignored, not an error
