"""FR-8, NFR-4: one door to Gemini (Google AI Studio free tier): routing, rate limits, retry, failover.

Free-tier content may be used by Google to improve products: only ever send public data.
Limits below are conservative placeholders; read your real ones in AI Studio and pass `limits=`.
A 429 from the API is the ground truth and is always retried/failed over.
"""
import json
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass

GEMINI = "https://generativelanguage.googleapis.com/v1beta"
# free of charge per ai.google.dev/gemini-api/docs/pricing, 2026-10-26
ROUTES = {
    "generator": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "judge": ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
    "vision": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
}
DEFAULT_LIMITS = {"rpm": 5, "rpd": 200}
BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}


class GatewayError(Exception):
    pass


class RateLimited(GatewayError):
    pass


class GatewayBlocked(GatewayError):
    pass


class GatewayHTTPError(GatewayError):
    def __init__(self, status, retry_after=None):
        super().__init__(f"HTTP {status}")
        self.status, self.retry_after = status, retry_after


@dataclass(frozen=True)
class Result:
    text: str
    model: str
    prompt_tokens: int
    output_tokens: int
    attempts: int


def _post(url, headers, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:  # body is dropped on purpose: it can echo request content
        ra = e.headers.get("Retry-After")
        raise GatewayHTTPError(e.code, float(ra) if ra and ra.replace(".", "", 1).isdigit() else None) from None
    except (urllib.error.URLError, TimeoutError):
        raise GatewayHTTPError(503) from None


class Gateway:
    def __init__(self, api_key, routes=None, limits=None, default_limits=None, max_retries=3, post=_post,
                 clock=time.time, sleep=time.sleep, jitter=lambda: random.uniform(0, 0.5)):
        self.key, self.routes = api_key, routes or ROUTES
        self.limits, self.default_limits = limits or {}, default_limits or DEFAULT_LIMITS
        self.max_retries, self.post, self.clock, self.sleep, self.jitter = max_retries, post, clock, sleep, jitter
        self._recent = defaultdict(deque)  # model -> send times within the last minute
        self._daily = {}  # model -> (day, count)
        self.stats = defaultdict(lambda: {"requests": 0, "prompt_tokens": 0, "output_tokens": 0})

    @classmethod
    def from_keys(cls, keys, **kw):
        return cls(keys.require("GOOGLE_AI_STUDIO_API_KEY"), **kw)

    def _limit(self, model):
        return {**self.default_limits, **self.limits.get(model, {})}

    def _day(self):
        return int(self.clock() // 86400)  # ponytail: UTC day; Google resets at midnight Pacific, so we may be off by hours

    def _exhausted(self, model):
        day, n = self._daily.get(model, (self._day(), 0))
        return day == self._day() and n >= self._limit(model)["rpd"]

    def _acquire(self, model):
        q, rpm = self._recent[model], self._limit(model)["rpm"]
        while q and self.clock() - q[0] >= 60:
            q.popleft()
        if len(q) >= rpm:
            self.sleep(60 - (self.clock() - q[0]))
            while q and self.clock() - q[0] >= 60:
                q.popleft()
        q.append(self.clock())
        day, n = self._daily.get(model, (self._day(), 0))
        self._daily[model] = (self._day(), n + 1 if day == self._day() else 1)

    def _request(self, model, body):
        self._acquire(model)
        resp = self.post(f"{GEMINI}/models/{model}:generateContent",
                         {"Content-Type": "application/json", "x-goog-api-key": self.key}, body)
        if (resp.get("promptFeedback") or {}).get("blockReason"):
            raise GatewayBlocked("prompt blocked by Gemini safety filters")
        cands = resp.get("candidates") or []
        if cands and cands[0].get("finishReason") in BLOCKED:
            raise GatewayBlocked(f"response blocked: {cands[0]['finishReason']}")
        text = "".join(p.get("text", "") for p in (cands[0].get("content") or {}).get("parts", [])) if cands else ""
        if not text:
            raise GatewayHTTPError(503)  # empty answer: transient, retry
        u = resp.get("usageMetadata") or {}
        return text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)

    def generate(self, role, prompt, system=None, json=False, temperature=None, images=()):
        if role not in self.routes:
            raise ValueError(f"unknown role: {role}")
        parts = [{"text": prompt}] + [{"inlineData": {"mimeType": m, "data": d}} for m, d in images]
        body = {"contents": [{"role": "user", "parts": parts}]}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        cfg = {k: v for k, v in (("temperature", temperature), ("responseMimeType", "application/json" if json else None))
               if v is not None}
        if cfg:
            body["generationConfig"] = cfg
        attempts, last = 0, None
        for model in self.routes[role]:
            if self._exhausted(model):
                continue
            for n in range(self.max_retries + 1):
                attempts += 1
                try:
                    text, pt, ot = self._request(model, body)
                except GatewayBlocked:
                    raise
                except GatewayHTTPError as e:
                    last = e
                    if e.status == 404:
                        break  # model name retired or unavailable: fail over
                    if e.status not in (408, 429) and e.status < 500:
                        raise GatewayError(f"request rejected: {e}") from None
                    if n < self.max_retries:
                        self.sleep(e.retry_after if e.retry_after else min(2.0 ** n, 30.0) + self.jitter())
                    continue
                s = self.stats[model]
                s["requests"] += 1
                s["prompt_tokens"] += pt
                s["output_tokens"] += ot
                return Result(text, model, pt, ot, attempts)
        if last is None:
            raise RateLimited(f"daily limit reached for every model in role {role}")
        raise GatewayError(f"all models failed for role {role}: last error {last}")
