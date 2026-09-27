"""Patient explainer: de-identified facts, safe Gemini rewrite with template fallback, ElevenLabs audio, the page."""
import json

import httpx
import pytest

from test_workflow import client_as, fundus_bytes, new_case, sign, upload  # noqa: E402  (seeds a temp workspace)

from insightrx.app import explainer  # noqa: E402

RESULT = {"overall": "Referable DR signal",
          "eyes": {"OD": {"status": "Referable DR signal", "edema_signal": True, "score": 0.91},
                   "OS": {"status": "No model finding", "edema_signal": False, "score": 0.08}}}
SNAPSHOT = [{"key": "eyes", "state": "signal"}, {"key": "heart", "state": "signal"},
            {"key": "kidneys", "state": "exploratory"}, {"key": "nerves", "state": "clear"}]


class Review:
    next_action = "refer_retina"


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ELEVENLABS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    explainer._summaries.clear()
    explainer._audio.clear()


def test_facts_are_categorical_and_carry_no_scores():
    f = explainer.facts(RESULT, SNAPSHOT, Review())
    assert f == {"eyes": {"OD": "signs", "OS": "clear"}, "edema": True, "organs": ["heart", "kidneys"],
                 "next": "refer_retina", "signed": True}
    assert "0.9" not in json.dumps(f) and explainer.facts(None, SNAPSHOT, None) is None


@pytest.mark.parametrize("lang, words", [("en", ["right eye", "left eye", "retina specialist", "does not make a diagnosis"]),
                                         ("pt", ["olho direito", "olho esquerdo", "especialista em retina", "não faz diagnóstico"])])
def test_template_in_both_languages(lang, words):
    text = explainer.summary(explainer.facts(RESULT, SNAPSHOT, Review()), lang)
    assert text["source"] == "template" and all(w in text["text"] for w in words)


def test_same_state_in_both_eyes_reads_naturally():
    f = {"eyes": {"OD": "clear", "OS": "clear"}, "edema": False, "organs": [], "next": None, "signed": False}
    assert "The photos of your eyes did not show" in explainer.template(f, "en")
    assert "As fotos dos seus dois olhos não mostraram" in explainer.template(f, "pt")


class FakeGemini:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []
        self.models = self

    def generate_content(self, model, contents, **kw):
        self.prompts.append(contents)
        return type("R", (), {"text": self.reply})()


def use_gemini(monkeypatch, reply):
    from google import genai
    fake = FakeGemini(reply)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(genai, "Client", lambda api_key: fake)
    return fake


def test_gemini_rewrite_is_used_when_safe(monkeypatch):
    f = explainer.facts(RESULT, SNAPSHOT, Review())
    base = explainer.template(f, "en")
    fake = use_gemini(monkeypatch, "We looked at the back of your eyes today. " + base)
    out = explainer.summary(f, "en")
    assert out["source"].startswith("Gemini") and fake.prompts[0].endswith(base)
    explainer.summary(f, "en")
    assert len(fake.prompts) == 1                                     # cached


@pytest.mark.parametrize("reply", ["Your risk is 91% so see a doctor in 2 weeks.", "", "Patient ID RL-P0042 is fine."])
def test_gemini_output_that_adds_facts_falls_back_to_template(monkeypatch, reply):
    use_gemini(monkeypatch, reply)
    f = explainer.facts(RESULT, SNAPSHOT, Review())
    out = explainer.summary(f, "en")
    assert out["text"] == explainer.template(f, "en") and "safety check" in out["source"]


def test_speech_calls_elevenlabs_and_caches(monkeypatch):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        assert request.headers["xi-api-key"] == "k"
        return httpx.Response(200, content=b"ID3mp3")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    client = httpx.Client(transport=httpx.MockTransport(handle))
    assert explainer.speech("Olá", client) == b"ID3mp3" and explainer.speech("Olá", client) == b"ID3mp3"
    assert len(calls) == 1 and calls[0]["model_id"] == explainer.ELEVEN_MODEL


def test_speech_without_key_or_on_error(monkeypatch):
    with pytest.raises(explainer.ExplainerUnavailable):
        explainer.speech("hello")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    bad = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    with pytest.raises(explainer.ExplainerUnavailable):
        explainer.speech("hello", bad)


def test_explainer_page_and_audio_route(monkeypatch):
    op, pcp = client_as("Sam Rivera"), client_as("Dr. Alex Morgan")
    cid = new_case(op, "RL-EXPLAIN1")
    upload(op, cid, "OD", fundus_bytes(41))
    op.post(f"/cases/{cid}/analyze")
    assert sign(pcp, cid).status_code == 303
    assert "Explain to patient" in pcp.get(f"/cases/{cid}").text
    page = pcp.get(f"/cases/{cid}/explain?lang=pt")
    assert page.status_code == 200 and "Próximo passo" in page.text and "RL-EXPLAIN1" in page.text
    assert "ELEVENLABS_API_KEY" in page.text and pcp.get(f"/cases/{cid}/explain.mp3").status_code == 503
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setattr(explainer, "speech", lambda text, client=None: b"ID3mp3")
    audio = pcp.get(f"/cases/{cid}/explain.mp3?lang=en")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/mpeg"
    assert op.get(f"/cases/{cid}/explain").status_code == 403                  # operators do not see it


def test_busy_gemini_model_falls_back_to_the_next(monkeypatch):
    f = explainer.facts(RESULT, SNAPSHOT, Review())
    base = explainer.template(f, "en")
    fake = use_gemini(monkeypatch, "Hello. " + base)
    tried = []

    def generate(model, contents, **kw):
        tried.append(model)
        if len(tried) == 1:
            raise RuntimeError("503 UNAVAILABLE")
        return type("R", (), {"text": fake.reply})()
    fake.generate_content = generate
    out = explainer.summary(f, "en")
    assert len(tried) == 2 and out["source"] == f"Gemini ({tried[1]}), checked"
