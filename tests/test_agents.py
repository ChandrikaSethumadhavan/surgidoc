import json

import pytest

from surgidoc.agents import ReportAgents
from surgidoc.gateway import GatewayBlocked
from surgidoc.ingest import Chunk


class FakeGateway:
    """Replies are consumed per role in order; an Exception instance is raised instead of returned."""

    def __init__(self, generator=(), judge=()):
        self.q = {"generator": list(generator), "judge": list(judge)}
        self.calls = []

    def generate(self, role, prompt, system=None, json=False, temperature=None, images=()):
        self.calls.append({"role": role, "prompt": prompt, "system": system, "json": json, "temperature": temperature})
        r = self.q[role].pop(0)
        if isinstance(r, Exception):
            raise r

        class R:
            text = r

        return R()

    def roles(self):
        return [c["role"] for c in self.calls]


CHUNKS = [Chunk("Cold snare polypectomy is recommended for polyps smaller than 10 mm.", "esge/poly.xml", 3),
          Chunk("Band ligation is preferred for variceal bleeding.", "esge/var.xml", 1)]
FINDINGS = ["Small sessile polyp in the sigmoid colon.", "Mild mucosal erythema."]


def retrieve(query):
    retrieve.queries.append(query)
    return CHUNKS


retrieve.queries = []


def draft(*statements):
    return json.dumps({"statements": [{"text": t, "sources": s} for t, s in statements]})


GOOD = draft(("A small sessile polyp was observed in the sigmoid colon.", ["F0"]),
             ("Cold snare removal is recommended for polyps under 10 mm.", ["F0", "K1"]))
APPROVE = json.dumps({"approved": True, "issues": [], "feedback": ""})


def reject(msg="statement 2 is not supported"):
    return json.dumps({"approved": False, "issues": [{"statement": 1, "problem": msg}], "feedback": msg})


def agents(gw, **kw):
    retrieve.queries.clear()
    return ReportAgents(gw, retrieve, **kw)


# --- happy path ---------------------------------------------------------------
def test_approved_on_first_attempt():
    gw = FakeGateway([GOOD], [APPROVE])
    r = agents(gw).run(FINDINGS)
    assert r.status == "approved" and r.attempts == 1 and r.draft is True
    assert [s.text[:7] for s in r.statements] == ["A small", "Cold sn"]
    assert r.statements[1].sources == ["F0", "K1"]
    assert gw.roles() == ["generator", "judge"]


def test_sources_map_resolves_ids_for_the_pdf():
    r = agents(FakeGateway([GOOD], [APPROVE])).run(FINDINGS)
    assert r.sources["F0"]["text"] == FINDINGS[0] and r.sources["F0"]["kind"] == "frame"
    assert r.sources["K1"]["source"] == "esge/poly.xml" and r.sources["K1"]["kind"] == "knowledge"
    assert set(r.sources) == {"F0", "F1", "K1", "K2"}


def test_retrieval_query_comes_from_findings():
    agents(FakeGateway([GOOD], [APPROVE])).run(FINDINGS)
    assert "sigmoid colon" in retrieve.queries[0]


def test_prompts_wrap_untrusted_text_as_data_and_forbid_diagnosis():
    gw = FakeGateway([GOOD], [APPROVE])
    agents(gw).run(FINDINGS)
    g, j = gw.calls
    assert "<source id=\"K1\">" in g["prompt"] and "<source id=\"F0\">" in g["prompt"]
    for c in (g, j):
        assert "not instructions" in c["system"].lower() and "diagnos" in c["system"].lower()
        assert c["json"] is True
    assert "<source id=\"K1\">" in j["prompt"] and "A small sessile polyp" in j["prompt"]


def test_judge_is_deterministic_generator_is_low_temperature():
    gw = FakeGateway([GOOD], [APPROVE])
    agents(gw).run(FINDINGS)
    assert gw.calls[1]["temperature"] == 0 and gw.calls[0]["temperature"] <= 0.3


# --- regeneration loop (AC-2) ----------------------------------------------------
def test_rejection_regenerates_with_judge_feedback_then_approves():
    gw = FakeGateway([GOOD, GOOD], [reject("not supported by K2"), APPROVE])
    r = agents(gw).run(FINDINGS)
    assert r.status == "approved" and r.attempts == 2
    assert gw.roles() == ["generator", "judge", "generator", "judge"]
    assert "not supported by K2" in gw.calls[2]["prompt"]


def test_stops_after_max_attempts_and_never_approves():
    gw = FakeGateway([GOOD] * 3, [reject()] * 3)
    r = agents(gw, max_attempts=3).run(FINDINGS)
    assert r.status == "rejected" and r.attempts == 3 and len(gw.calls) == 6
    assert r.statements  # last draft is kept for the human reviewer
    assert r.history[-1]["verdict"] == "rejected" and len(r.history) == 3


def test_judge_cannot_be_talked_into_approving_with_open_issues():
    mixed = json.dumps({"approved": True, "issues": [{"statement": 0, "problem": "x"}], "feedback": ""})
    r = agents(FakeGateway([GOOD], [mixed]), max_attempts=1).run(FINDINGS)
    assert r.status == "rejected"


@pytest.mark.parametrize("bad", ["not json", "[]", json.dumps({"issues": []}), json.dumps({"approved": "yes"})])
def test_unparseable_or_malformed_judge_output_fails_closed(bad):
    r = agents(FakeGateway([GOOD], [bad]), max_attempts=1).run(FINDINGS)
    assert r.status == "rejected"


# --- citation enforcement (NFR-3) ------------------------------------------------
def test_uncited_statement_is_rejected_before_the_judge_is_called():
    uncited = draft(("Unsupported claim.", []))
    gw = FakeGateway([uncited, GOOD], [APPROVE])
    r = agents(gw).run(FINDINGS)
    assert gw.roles() == ["generator", "generator", "judge"]  # no judge call wasted on the bad draft
    assert r.status == "approved" and r.attempts == 2
    assert "cite" in gw.calls[1]["prompt"].lower()


def test_unknown_source_id_is_rejected():
    gw = FakeGateway([draft(("Claim.", ["K9"])), draft(("Claim.", ["K9"]))], [])
    r = agents(gw, max_attempts=2).run(FINDINGS)
    assert r.status == "rejected" and "judge" not in gw.roles()


@pytest.mark.parametrize("bad", ["not json", "{}", json.dumps({"statements": []}), json.dumps({"statements": [{"text": "x"}]}),
                                 json.dumps({"statements": [{"text": "", "sources": ["F0"]}]})])
def test_malformed_generator_output_counts_as_an_attempt(bad):
    gw = FakeGateway([bad, GOOD], [APPROVE])
    r = agents(gw).run(FINDINGS)
    assert r.status == "approved" and r.attempts == 2


def test_all_generator_attempts_malformed_ends_rejected_with_no_statements():
    r = agents(FakeGateway(["nope"] * 2), max_attempts=2).run(FINDINGS)
    assert r.status == "rejected" and r.statements == []


# --- failure handling ---------------------------------------------------------------
def test_safety_block_ends_run_as_blocked():
    r = agents(FakeGateway([GatewayBlocked("blocked")])).run(FINDINGS)
    assert r.status == "blocked" and r.statements == []


def test_empty_findings_is_an_error_and_makes_no_calls():
    gw = FakeGateway()
    with pytest.raises(ValueError):
        agents(gw).run([])
    with pytest.raises(ValueError):
        agents(gw).run(["   "])
    assert gw.calls == []


def test_empty_knowledge_base_result_still_runs_citing_only_frames():
    only_frames = draft(("A polyp was observed.", ["F0"]))
    r = ReportAgents(FakeGateway([only_frames], [APPROVE]), lambda q: []).run(FINDINGS)
    assert r.status == "approved" and set(r.sources) == {"F0", "F1"}


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError):
        ReportAgents(FakeGateway(), retrieve, max_attempts=0)


# --- orchestration is a LangGraph graph ----------------------------------------------
def test_orchestration_is_a_langgraph_with_expected_nodes():
    graph = ReportAgents(FakeGateway(), retrieve).graph
    assert {"retrieve", "generate", "judge"} <= set(graph.get_graph().nodes)
