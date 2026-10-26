"""FR-1, FR-2: video -> frames -> text findings.

Gemini vision (via the gateway) is the working default. EndoViT is an encoder, not a captioner, so it is only used
when a hosted endpoint honouring this contract is configured: POST {image_b64, mime} -> {"findings": str, "labels": [...]}.
Findings are descriptions of what is visible, never diagnoses.
"""
import base64
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SYSTEM = ("You describe frames from an endoscopy video for a clinical documentation assistant. Describe only what is "
          "visible: anatomy, instruments, mucosal appearance, bleeding, lesions. Do not diagnose, name diseases as "
          "conclusions, or recommend treatment. Say 'unclear' when a frame is not interpretable. "
          'Return a JSON array with exactly one object per frame: [{"frame": <int>, "finding": <string>}].')


class VideoError(Exception):
    pass


class EndoViTError(Exception):
    pass


class VisionError(Exception):
    pass


@dataclass(frozen=True)
class Frame:
    index: int
    t: float  # seconds into the video
    jpeg: bytes


@dataclass(frozen=True)
class Finding:
    frame: int
    t: float
    text: str
    source: str  # "endovit" | "gemini-vision"


def extract_frames(video, every_s=2.0, max_frames=16, max_side=768):
    """Uniformly sample one frame every `every_s` seconds, capped at `max_frames`, downsized and JPEG-encoded."""
    if every_s <= 0 or max_frames < 1 or max_side < 64:
        raise ValueError("every_s must be > 0, max_frames >= 1, max_side >= 64")
    path = Path(video)
    if not path.is_file():
        raise VideoError(f"video not found: {path.name}")
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or fps <= 0 or count <= 0:
            raise VideoError(f"cannot read video: {path.name}")
        duration, out, k = count / fps, [], 0
        while len(out) < max_frames and k * every_s < duration:
            t = k * every_s
            k += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
            ok, img = cap.read()
            if not ok:
                continue
            h, w = img.shape[:2]
            if max(h, w) > max_side:
                s = max_side / max(h, w)
                img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                out.append(Frame(len(out), t, buf.tobytes()))
        return out
    finally:
        cap.release()


def _post(url, headers, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


class EndoViTClient:
    def __init__(self, url, token, post=_post):
        if not url.startswith("https://"):
            raise ValueError("EndoViT endpoint must use https")
        self.url, self.token, self.post = url, token, post

    def describe(self, frames):
        out = []
        for f in frames:
            body = {"image_b64": base64.b64encode(f.jpeg).decode(), "mime": "image/jpeg"}
            try:
                resp = self.post(self.url, {"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"}, body)
            except Exception as e:  # message dropped on purpose: it could echo the token or image
                raise EndoViTError(f"EndoViT request failed: {type(e).__name__}") from None
            text = resp.get("findings") if isinstance(resp, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise EndoViTError("EndoViT response has no findings")
            out.append(Finding(f.index, f.t, text.strip(), "endovit"))
        return out


class GeminiVisionDescriber:
    """Several frames per request: the free tier is limited by requests per minute, not by images."""

    def __init__(self, gateway, batch=6, retries=1):
        self.gateway, self.batch, self.retries = gateway, batch, retries

    def describe(self, frames):
        out = []
        for i in range(0, len(frames), self.batch):
            out += self._describe_batch(frames[i:i + self.batch])
        return out

    def _describe_batch(self, frames):
        prompt = "Describe each frame in order.\n" + "\n".join(f"Frame {f.index} (t={f.t:.1f}s)" for f in frames)
        images = [("image/jpeg", base64.b64encode(f.jpeg).decode()) for f in frames]
        err = None
        for _ in range(self.retries + 1):
            text = self.gateway.generate("vision", prompt, system=SYSTEM, json=True, images=images).text
            try:
                return self._parse(text, frames)
            except VisionError as e:
                err = e
        raise err

    @staticmethod
    def _parse(text, frames):
        try:
            items = json.loads(text)
            got = {int(x["frame"]): str(x["finding"]).strip() for x in items}
        except (ValueError, TypeError, KeyError):
            raise VisionError("vision response is not the expected JSON array") from None
        want = {f.index: f for f in frames}
        if set(got) - set(want):
            raise VisionError(f"vision response has unexpected frames: {sorted(set(got) - set(want))}")
        if set(want) - set(got) or not all(got.values()):
            raise VisionError(f"vision response is missing frames: {sorted(set(want) - set(got))}")
        return [Finding(i, want[i].t, got[i], "gemini-vision") for i in sorted(want)]


def describe_frames(frames, primary, fallback):
    """Try the primary describer per frame; only the failures go to the fallback. Result is in frame order."""
    done, rest = {}, list(frames)
    if primary:
        rest = []
        for f in frames:
            try:
                done[f.index] = primary.describe([f])[0]
            except EndoViTError:
                rest.append(f)
    if rest:
        done.update({x.frame: x for x in fallback.describe(rest)})
    return [done[f.index] for f in frames]


class Describer:
    def __init__(self, primary, fallback):
        self.primary, self.fallback = primary, fallback

    def describe(self, frames):
        return describe_frames(frames, self.primary, self.fallback)


def describer_from_keys(keys, gateway, **kw):
    """EndoViT is optional: used first only if ENDOVIT_INFERENCE_URL is set (then the token is required)."""
    url = keys.get("ENDOVIT_INFERENCE_URL")
    primary = EndoViTClient(url, keys.require("ENDOVIT_INFERENCE_TOKEN")) if url else None
    return Describer(primary, GeminiVisionDescriber(gateway, **kw))
