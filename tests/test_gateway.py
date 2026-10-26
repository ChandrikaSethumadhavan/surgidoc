import pytest

from surgidoc.config import MissingKey, load_keys
from surgidoc.gateway import Gateway, GatewayBlocked, GatewayError, GatewayHTTPError, RateLimited

OK = {"candidates": [{"content": {"parts": [{"text": "hel"}, {"text": "lo"}]}, "finishReason": "STOP"}],
      "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}}


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t, self.slept = t, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def make(responses, clock=None, routes=None, **kw):
    """responses: list of dicts (returned) or exceptions (raised), consumed in order."""
    clock = clock or Clock()
    calls = []
    queue = list(responses)

    def post(url, headers, body):
        calls.append((url, headers, body))
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    g = Gateway("SECRET", routes=routes or {"generator": ["m1", "m2"], "judge": ["m2"]}, post=post,
                clock=clock, sleep=clock.sleep, jitter=lambda: 0.0, **kw)
    return g, calls, clock


def test_request_shape_and_result():
    g, calls, _ = make([OK])
    r = g.generate("generator", "write report", system="be careful", json=True, temperature=0.2)
    url, headers, body = calls[0]
    assert url.endswith("/models/m1:generateContent") and "SECRET" not in url
    assert headers["x-goog-api-key"] == "SECRET"
    assert body["contents"][0]["parts"][0]["text"] == "write report"
    assert body["systemInstruction"]["parts"][0]["text"] == "be careful"
    assert body["generationConfig"] == {"temperature": 0.2, "responseMimeType": "application/json"}
    assert (r.text, r.model, r.prompt_tokens, r.output_tokens) == ("hello", "m1", 7, 3)


def test_unknown_role_rejected():
    g, _, _ = make([])
    with pytest.raises(ValueError, match="role"):
        g.generate("nope", "x")


def test_rpm_limit_waits_for_window():
    g, calls, clock = make([OK, OK, OK], limits={"m1": {"rpm": 2, "rpd": 100}})
    g.generate("generator", "a")
    g.generate("generator", "b")
    t0 = clock.t
    g.generate("generator", "c")
    assert len(calls) == 3 and clock.slept and clock.t - t0 >= 59  # waited for the 60s window


def test_retry_with_exponential_backoff_on_429_then_success():
    g, calls, clock = make([GatewayHTTPError(429), GatewayHTTPError(503), OK], default_limits={"rpm": 100, "rpd": 100})
    r = g.generate("generator", "x")
    assert r.model == "m1" and r.attempts == 3
    assert clock.slept == [1.0, 2.0]


def test_retry_after_header_is_honoured():
    g, _, clock = make([GatewayHTTPError(429, retry_after=17), OK], default_limits={"rpm": 100, "rpd": 100})
    g.generate("generator", "x")
    assert clock.slept == [17]


def test_falls_back_to_next_model_after_retries_exhausted():
    errs = [GatewayHTTPError(429)] * 3
    g, calls, _ = make(errs + [OK], max_retries=2, default_limits={"rpm": 100, "rpd": 100})
    r = g.generate("generator", "x")
    assert r.model == "m2" and calls[-1][0].endswith("m2:generateContent")


def test_all_models_failing_raises_gateway_error_without_leaking_key():
    errs = [GatewayHTTPError(500)] * 6
    g, _, _ = make(errs, max_retries=2, default_limits={"rpm": 100, "rpd": 100})
    with pytest.raises(GatewayError) as e:
        g.generate("generator", "x")
    assert "SECRET" not in str(e.value)


def test_client_error_is_not_retried_or_failed_over():
    g, calls, _ = make([GatewayHTTPError(400), OK], default_limits={"rpm": 100, "rpd": 100})
    with pytest.raises(GatewayError):
        g.generate("generator", "x")
    assert len(calls) == 1


def test_unknown_model_404_fails_over():
    g, calls, _ = make([GatewayHTTPError(404), OK], default_limits={"rpm": 100, "rpd": 100})
    assert g.generate("generator", "x").model == "m2"


def test_blocked_prompt_raises_and_is_not_retried():
    blocked = {"promptFeedback": {"blockReason": "SAFETY"}}
    g, calls, _ = make([blocked, OK], default_limits={"rpm": 100, "rpd": 100})
    with pytest.raises(GatewayBlocked):
        g.generate("generator", "x")
    assert len(calls) == 1


def test_blocked_candidate_finish_reason():
    resp = {"candidates": [{"finishReason": "SAFETY"}]}
    g, _, _ = make([resp], default_limits={"rpm": 100, "rpd": 100})
    with pytest.raises(GatewayBlocked):
        g.generate("generator", "x")


def test_empty_response_is_an_error():
    g, _, _ = make([{"candidates": []}] * 6, max_retries=1, default_limits={"rpm": 100, "rpd": 100})
    with pytest.raises(GatewayError):
        g.generate("generator", "x")


def test_daily_limit_skips_model_and_resets_next_day():
    clock = Clock()
    g, calls, _ = make([OK, OK, OK], clock=clock, routes={"generator": ["m1", "m2"]},
                       limits={"m1": {"rpm": 100, "rpd": 1}, "m2": {"rpm": 100, "rpd": 1}})
    assert g.generate("generator", "a").model == "m1"
    assert g.generate("generator", "b").model == "m2"
    with pytest.raises(RateLimited):
        g.generate("generator", "c")
    assert len(calls) == 2  # exhausted: no HTTP call made
    clock.t += 86400
    assert g.generate("generator", "d").model == "m1"


def test_stats_count_requests_and_tokens():
    g, _, _ = make([OK, OK], default_limits={"rpm": 100, "rpd": 100})
    g.generate("generator", "a")
    g.generate("generator", "b")
    assert g.stats["m1"] == {"requests": 2, "prompt_tokens": 14, "output_tokens": 6}


def test_from_keys_requires_google_key(tmp_path):
    p = tmp_path / "keys.txt"
    p.write_text("X=1\n")
    with pytest.raises(MissingKey, match="GOOGLE_AI_STUDIO_API_KEY"):
        Gateway.from_keys(load_keys(p))


def test_images_are_sent_as_inline_data_for_vision():
    g, calls, _ = make([OK], routes={"vision": ["m1"]})
    g.generate("vision", "describe", images=[("image/png", "QUJD")])
    parts = calls[0][2]["contents"][0]["parts"]
    assert parts[0] == {"text": "describe"}
    assert parts[1] == {"inlineData": {"mimeType": "image/png", "data": "QUJD"}}
