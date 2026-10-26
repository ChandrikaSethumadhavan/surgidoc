import base64
import json

import pytest

from surgidoc.config import load_keys
from surgidoc.vision import (EndoViTClient, EndoViTError, Finding, Frame, GeminiVisionDescriber, VideoError,
                             VisionError, describe_frames, describer_from_keys, extract_frames)

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def make_video(path, frames=20, fps=10, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        w.write(np.full((size[1], size[0], 3), i * 10 % 255, dtype=np.uint8))
    w.release()
    return path


# --- frame extraction (real OpenCV on a synthetic video) ----------------------
def test_extract_samples_by_interval_and_returns_jpeg(tmp_path):
    v = make_video(tmp_path / "v.mp4")  # 2.0 s at 10 fps
    frames = extract_frames(v, every_s=0.5)
    assert [round(f.t, 1) for f in frames] == [0.0, 0.5, 1.0, 1.5]
    assert [f.index for f in frames] == [0, 1, 2, 3]
    assert all(f.jpeg[:2] == b"\xff\xd8" for f in frames)


def test_extract_caps_frames(tmp_path):
    v = make_video(tmp_path / "v.mp4", frames=100)
    assert len(extract_frames(v, every_s=0.5, max_frames=3)) == 3


def test_extract_downsizes_large_frames(tmp_path):
    v = make_video(tmp_path / "big.mp4", frames=5, size=(1280, 720))
    f = extract_frames(v, every_s=0.2, max_side=640)[0]
    h, w = cv2.imdecode(np.frombuffer(f.jpeg, np.uint8), cv2.IMREAD_COLOR).shape[:2]
    assert max(h, w) <= 640 and w > h


def test_extract_missing_or_unreadable_file(tmp_path):
    with pytest.raises(VideoError, match="not found"):
        extract_frames(tmp_path / "nope.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(VideoError):
        extract_frames(bad)


@pytest.mark.parametrize("kw", [{"every_s": 0}, {"max_frames": 0}, {"max_side": 10}])
def test_extract_rejects_bad_parameters(tmp_path, kw):
    with pytest.raises(ValueError):
        extract_frames(make_video(tmp_path / "v.mp4"), **kw)


# --- EndoViT endpoint client ---------------------------------------------------
FRAME = Frame(0, 0.0, b"\xff\xd8jpeg")


def test_endovit_request_and_response_mapping():
    calls = []

    def post(url, headers, body):
        calls.append((url, headers, body))
        return {"findings": "mucosal fold", "labels": ["fold"]}

    f = EndoViTClient("https://ev.example/infer", "TOK", post=post).describe([FRAME])[0]
    url, headers, body = calls[0]
    assert url == "https://ev.example/infer" and headers["Authorization"] == "Bearer TOK"
    assert base64.b64decode(body["image_b64"]) == FRAME.jpeg and body["mime"] == "image/jpeg"
    assert (f.text, f.source, f.frame) == ("mucosal fold", "endovit", 0)


def test_endovit_rejects_bad_responses_without_leaking_token():
    for bad in [{"nope": 1}, {"findings": ""}, {"findings": 5}, "str"]:
        with pytest.raises(EndoViTError) as e:
            EndoViTClient("https://ev/x", "TOK", post=lambda u, h, b, bad=bad: bad).describe([FRAME])
        assert "TOK" not in str(e.value)


def test_endovit_transport_failure_becomes_endovit_error():
    def post(u, h, b):
        raise OSError("connection TOK refused")

    with pytest.raises(EndoViTError) as e:
        EndoViTClient("https://ev/x", "TOK", post=post).describe([FRAME])
    assert "TOK" not in str(e.value)


def test_endovit_requires_https():
    with pytest.raises(ValueError):
        EndoViTClient("http://ev/x", "TOK")


# --- Gemini vision describer -----------------------------------------------------
class FakeGateway:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def generate(self, role, prompt, system=None, json=False, temperature=None, images=()):
        self.calls.append({"role": role, "prompt": prompt, "system": system, "json": json, "images": list(images)})

        class R:
            text = self.replies.pop(0)

        return R()


def frames(n):
    return [Frame(i, i * 2.0, b"\xff\xd8" + bytes([i])) for i in range(n)]


def reply(idx, text="finding"):
    return json.dumps([{"frame": i, "finding": f"{text} {i}"} for i in idx])


def test_gemini_batches_frames_and_maps_findings():
    gw = FakeGateway([reply(range(0, 6)), reply(range(6, 12)), reply(range(12, 13))])
    out = GeminiVisionDescriber(gw, batch=6).describe(frames(13))
    assert len(gw.calls) == 3 and [len(c["images"]) for c in gw.calls] == [6, 6, 1]
    assert [f.frame for f in out] == list(range(13)) and out[7].text == "finding 7"
    assert all(f.source == "gemini-vision" for f in out)
    assert out[7].t == 14.0


def test_gemini_request_uses_vision_role_json_and_safety_instruction():
    gw = FakeGateway([reply([0])])
    GeminiVisionDescriber(gw).describe(frames(1))
    c = gw.calls[0]
    assert c["role"] == "vision" and c["json"] is True
    assert c["images"][0] == ("image/jpeg", base64.b64encode(frames(1)[0].jpeg).decode())
    assert "do not diagnose" in c["system"].lower() and "frame 0" in c["prompt"].lower()


def test_gemini_retries_once_on_invalid_json_then_fails():
    gw = FakeGateway(["not json", reply([0])])
    assert GeminiVisionDescriber(gw).describe(frames(1))[0].text == "finding 0"
    gw2 = FakeGateway(["not json", "still not"])
    with pytest.raises(VisionError):
        GeminiVisionDescriber(gw2).describe(frames(1))


def test_gemini_missing_or_unknown_frames_are_errors():
    with pytest.raises(VisionError, match="missing"):
        GeminiVisionDescriber(FakeGateway([reply([0]), reply([0])])).describe(frames(2))
    bad = json.dumps([{"frame": 9, "finding": "x"}, {"frame": 0, "finding": "x"}])
    with pytest.raises(VisionError):
        GeminiVisionDescriber(FakeGateway([bad, bad])).describe(frames(1))


# --- fallback chain -----------------------------------------------------------------
class FakeEndoViT:
    def __init__(self, fail_on=()):
        self.fail_on, self.seen = set(fail_on), []

    def describe(self, fs):
        out = []
        for f in fs:
            self.seen.append(f.index)
            if f.index in self.fail_on:
                raise EndoViTError("down")
            out.append(Finding(f.index, f.t, f"ev {f.index}", "endovit"))
        return out


def test_primary_success_makes_no_gemini_calls():
    gw = FakeGateway([])
    out = describe_frames(frames(3), FakeEndoViT(), GeminiVisionDescriber(gw))
    assert [f.source for f in out] == ["endovit"] * 3 and gw.calls == []


def test_only_failed_frames_fall_back_to_gemini_in_order():
    gw = FakeGateway([reply([1])])
    out = describe_frames(frames(3), FakeEndoViT(fail_on={1}), GeminiVisionDescriber(gw))
    assert [f.source for f in out] == ["endovit", "gemini-vision", "endovit"]
    assert [f.frame for f in out] == [0, 1, 2] and len(gw.calls[0]["images"]) == 1


def test_no_primary_uses_gemini_for_everything():
    gw = FakeGateway([reply(range(3))])
    out = describe_frames(frames(3), None, GeminiVisionDescriber(gw))
    assert [f.source for f in out] == ["gemini-vision"] * 3


def test_empty_frames_returns_empty_without_calls():
    gw = FakeGateway([])
    assert describe_frames([], FakeEndoViT(), GeminiVisionDescriber(gw)) == [] and gw.calls == []


def test_describer_from_keys_endovit_optional(tmp_path):
    p = tmp_path / "keys.txt"
    p.write_text("GOOGLE_AI_STUDIO_API_KEY=k\n")
    assert describer_from_keys(load_keys(p), FakeGateway([])).primary is None
    p.write_text("ENDOVIT_INFERENCE_URL=https://ev.example/infer\nENDOVIT_INFERENCE_TOKEN=t\n")
    assert isinstance(describer_from_keys(load_keys(p), FakeGateway([])).primary, EndoViTClient)
