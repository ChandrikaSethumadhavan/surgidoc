"""FR-12: validation run on a held-out video (surgidoc.validate)."""
import json

import cv2
import numpy as np
import pytest

from surgidoc import validate
from surgidoc.agents import Report, Statement
from surgidoc.dataset import DatasetError, HeldOutSplitError
from surgidoc.pipeline import RunResult
from surgidoc.vision import Finding


def png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), array)


def dataset(tmp_path, n=3, split="dev", classes=(21, 12, 22), video="video09"):
    """n frames in one clip; frame i's mask is filled with classes[i] (watershed values) plus background."""
    frames = []
    for i in range(n):
        clip = f"{video}_00080"
        png(tmp_path / video / clip / f"frame_{i + 1}_endo.png", np.full((48, 64, 3), (40 + 20 * i) % 200, np.uint8))
        mask = np.full((48, 64, 3), 50, np.uint8)
        mask[:24] = classes[i % len(classes)]
        png(tmp_path / video / clip / f"frame_{i + 1}_endo_watershed_mask.png", mask)
        frames.append({"video": video, "clip": clip, "index": i + 1, "split": split})
    (tmp_path / "manifest.json").write_text(json.dumps({"slug": "x", "files": [], "frames": frames}))
    return tmp_path


class FakePipe:
    """Stands in for the real pipeline: counts the frames of the uploaded mp4 and returns canned findings and a report."""

    def __init__(self, texts, statements=None, status="pending_review"):
        self.texts, self.status, self.seen = texts, status, {}
        self.statements = statements or [Statement("Liver visible.", ["F0"]), Statement("Guideline reminder.", ["K1"])]

    def setup(self):
        pass

    def run(self, video, actor):
        cap = cv2.VideoCapture(video)
        self.seen = {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": cap.get(cv2.CAP_PROP_FPS), "actor": actor}
        cap.release()
        findings = [Finding(i, 2.0 * i, t, "gemini-vision") for i, t in enumerate(self.texts)]
        report = Report("approved", self.statements, {"F0": {"kind": "frame", "text": "x"}, "K1": {"kind": "knowledge", "text": "y"}}, 2, [], True)
        return RunResult(self.status, report, findings, len(findings))


def test_the_video_holds_each_frame_for_two_seconds_in_order(tmp_path):
    root = dataset(tmp_path)
    pipe = FakePipe(["Liver.", "Fat.", "Gallbladder."])
    validate.validate(pipe, "Olive", root, "video09")
    assert pipe.seen["fps"] == 1.0 and pipe.seen["frames"] == 6 and pipe.seen["actor"] == "Olive"  # 3 frames x 2 s


def test_metrics_compare_each_description_with_its_own_mask(tmp_path):
    root = dataset(tmp_path, classes=(21, 12, 22))  # liver, fat, gallbladder
    out = validate.validate(FakePipe(["The liver is visible.", "A grasper holds tissue.", "unclear"]), "Olive", root, "video09")
    m = out["metrics"]
    assert out["frames"][0]["recall"] == 1.0 and out["frames"][1]["absent_mentions"] == ["Grasper"] and out["frames"][2]["unclear"]
    assert m["structure_recall"] == pytest.approx(1 / 3)  # frame 0 finds its liver (1.0); the other two name nothing that is there (0.0)
    assert m["absent_mention_rate"] == pytest.approx(1 / 3) and m["unclear_rate"] == pytest.approx(1 / 3)


def test_report_level_metrics(tmp_path):
    root = dataset(tmp_path)
    out = validate.validate(FakePipe(["Liver.", "Fat.", "Gallbladder."]), "Olive", root, "video09")
    m = out["metrics"]
    assert m["status"] == "pending_review" and m["judge_approved"] and m["guardrails_passed"] and m["attempts"] == 2
    assert m["statements"] == 2 and m["guideline_reminders"] == 1 and m["citation_validity"] == 1.0 and m["seconds"] >= 0


def test_a_statement_citing_an_unknown_source_lowers_citation_validity(tmp_path):
    root = dataset(tmp_path)
    pipe = FakePipe(["Liver.", "Fat.", "Gallbladder."], statements=[Statement("ok", ["F0"]), Statement("bad", ["F9"])])
    assert validate.validate(pipe, "Olive", root, "video09")["metrics"]["citation_validity"] == 0.5


def test_a_guardrail_block_is_reported_not_hidden(tmp_path):
    root = dataset(tmp_path)
    m = validate.validate(FakePipe(["Liver.", "Fat.", "Gallbladder."], status="blocked_by_guardrails"), "Olive", root, "video09")["metrics"]
    assert m["judge_approved"] and not m["guardrails_passed"]


def test_the_test_split_is_held_out_until_final_and_every_look_is_logged(tmp_path):
    root = dataset(tmp_path, split="test")
    pipe = FakePipe(["Liver.", "Fat.", "Gallbladder."])
    with pytest.raises(HeldOutSplitError):
        validate.validate(pipe, "Olive", root, "video09", split="test")
    assert pipe.seen == {} and not (root / "test_runs.json").exists()  # nothing ran
    validate.validate(pipe, "Olive", root, "video09", split="test", final=True)
    validate.validate(pipe, "Olive", root, "video09", split="test", final=True)
    assert len(json.loads((root / "test_runs.json").read_text())) == 2


def test_dev_runs_are_not_logged_as_test_looks(tmp_path):
    root = dataset(tmp_path)
    validate.validate(FakePipe(["Liver.", "Fat.", "Gallbladder."]), "Olive", root, "video09")
    assert not (root / "test_runs.json").exists()


def test_at_most_sixteen_frames_are_sent(tmp_path):
    root = dataset(tmp_path, n=20)
    pipe = FakePipe(["Liver."] * 16)
    out = validate.validate(pipe, "Olive", root, "video09")
    assert out["metrics"]["frames_sent"] == 16 and pipe.seen["frames"] == 32


def test_missing_video_or_files_are_clear_errors(tmp_path):
    root = dataset(tmp_path)
    with pytest.raises(DatasetError, match="no dev frames for video99"):
        validate.validate(FakePipe([]), "Olive", root, "video99")
    next((tmp_path / "video09").rglob("*_endo.png")).unlink()
    with pytest.raises(DatasetError, match="not downloaded"):
        validate.validate(FakePipe([]), "Olive", root, "video09")


def test_command_writes_the_result_and_needs_final_for_test(tmp_path, capsys):
    root = dataset(tmp_path / "ds")
    keys = tmp_path / "keys.txt"
    keys.write_text("SURGIDOC_USERS=Ada:admin;Olive:operator\n")
    out = tmp_path / "out" / "run.json"
    lines = []
    res = validate.main(["--root", str(root), "--keys", str(keys), "--out", str(out), "--video", "video09"],
                        build=lambda k: FakePipe(["Liver.", "Fat.", "Gallbladder."]), log=lines.append)
    assert json.loads(out.read_text())["metrics"]["status"] == "pending_review" and res["video"] == "video09"
    assert any(l.startswith("structure_recall") for l in lines)
    with pytest.raises(SystemExit):
        validate.main(["--split", "test", "--root", str(root), "--keys", str(keys)])
    assert "--final" in capsys.readouterr().err


def test_command_needs_an_operator_user(tmp_path, capsys):
    keys = tmp_path / "keys.txt"
    keys.write_text("SURGIDOC_USERS=Ada:admin\n")
    with pytest.raises(SystemExit):
        validate.main(["--root", str(dataset(tmp_path / "ds")), "--keys", str(keys)], build=lambda k: FakePipe([]))
    assert "operator" in capsys.readouterr().err
