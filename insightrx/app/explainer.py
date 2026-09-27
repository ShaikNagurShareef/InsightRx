"""
Patient explainer: the screening result in plain words, in English or Brazilian Portuguese, read aloud.

  1. facts()      de-identified facts from the analysis and the signed review (no names, refs, dates, ids or scores)
  2. summary()    Gemini rewrites the deterministic template for a patient (MLH "Best Use of Gemini API"); the output
                  is rejected, and the template used instead, if it adds numbers, identifiers or runs long
  3. speech()     ElevenLabs reads it aloud with a multilingual voice (MLH "Best Use of ElevenLabs")

Both services are optional: without GEMINI_API_KEY the template is shown, without ELEVENLABS_API_KEY there is no audio.
"""
import hashlib
import logging
import os
import re
from collections import OrderedDict

import httpx

from .llm import FORBIDDEN, GEMINI_MODEL, RestrictedPayload

LANGS = {"en": "English", "pt": "Português (Brasil)"}
ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}"
ELEVEN_VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")      # premade "Rachel"
ELEVEN_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")
MAX_CHARS, CACHE_SIZE = 1200, 64
_DIGIT = re.compile(r"\d")
log = logging.getLogger("insightrx.explainer")
_summaries: OrderedDict = OrderedDict()
_audio: OrderedDict = OrderedDict()

EYE_STATE = {"Referable DR signal": "signs", "No model finding": "clear", "Uncertain quality": "unclear",
             "Unable to assess": "unclear", "Not provided": "missing"}
NEXT_STEPS = ("refer_retina", "routine_rescreen", "manual_exam", "recapture", "systemic_review")
ORGAN_WORDS = {"heart": ("heart and circulation", "o coração e a circulação"), "kidneys": ("kidneys", "os rins"),
               "nerves": ("nerves", "os nervos"), "metabolism": ("metabolism", "o metabolismo")}

TEXT = {
    "en": {
        "intro": "Today we took photos of the back of your eyes to look for diabetic eye disease.",
        "eye": {"signs": "The photos of your {eye} show signs of diabetic eye disease that an eye specialist should look at.",
                "clear": "The photos of your {eye} did not show signs of diabetic eye disease.",
                "unclear": "The photos of your {eye} were not clear enough to check.",
                "missing": "We do not have photos of your {eye}."},
        "eyes": {"OD": "right eye", "OS": "left eye"},
        "edema": "There may also be swelling near the centre of vision, so this should not wait.",
        "organs": "Your eyes can also show early clues about your {organs}. Please talk about these with your doctor.",
        "next": {"refer_retina": "Next step: we will refer you to a retina specialist.",
                 "routine_rescreen": "Next step: repeat this eye check at your next routine visit.",
                 "manual_exam": "Next step: a full eye examination with an eye doctor.",
                 "recapture": "Next step: we need to take the photos again.",
                 "systemic_review": "Next step: your doctor will review your general health.",
                 None: "Next step: your clinician will review these photos with you."},
        "reviewed": "Your clinician has reviewed and signed these results.",
        "caveat": "A computer program helps your care team read the photos. It does not make a diagnosis.",
        "and": "and",
    },
    "pt": {
        "intro": "Hoje tiramos fotos do fundo dos seus olhos para procurar retinopatia diabética.",
        "eye": {"signs": "As fotos do seu {eye} mostram sinais de doença diabética nos olhos que um especialista deve avaliar.",
                "clear": "As fotos do seu {eye} não mostraram sinais de doença diabética nos olhos.",
                "unclear": "As fotos do seu {eye} não ficaram nítidas o suficiente para avaliar.",
                "missing": "Não temos fotos do seu {eye}."},
        "eyes": {"OD": "olho direito", "OS": "olho esquerdo"},
        "edema": "Também pode haver inchaço perto do centro da visão, por isso isto não deve esperar.",
        "organs": "Os olhos também podem mostrar sinais precoces sobre {organs}. Converse sobre isso com o seu médico.",
        "next": {"refer_retina": "Próximo passo: vamos encaminhar você a um especialista em retina.",
                 "routine_rescreen": "Próximo passo: repetir este exame na sua próxima consulta de rotina.",
                 "manual_exam": "Próximo passo: um exame completo com um oftalmologista.",
                 "recapture": "Próximo passo: precisamos tirar as fotos novamente.",
                 "systemic_review": "Próximo passo: o seu médico vai avaliar a sua saúde geral.",
                 None: "Próximo passo: o seu médico vai revisar estas fotos com você."},
        "reviewed": "O seu médico revisou e assinou estes resultados.",
        "caveat": "Um programa de computador ajuda a equipe de saúde a ler as fotos. Ele não faz diagnóstico.",
        "and": "e",
    },
}


class ExplainerUnavailable(Exception):
    pass


def gemini_enabled() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def voice_enabled() -> bool:
    return bool(os.environ.get("ELEVENLABS_API_KEY"))


def facts(result: dict | None, snapshot: list, review) -> dict | None:
    """Only categorical, non-identifying facts: eye states, edema, organ keys with signals, next step, signed."""
    if not result or result.get("overall") is None:
        return None
    eyes = result.get("eyes", {})
    return {
        "eyes": {e: EYE_STATE.get((eyes.get(e) or {}).get("status"), "missing") for e in ("OD", "OS")},
        "edema": any((eyes.get(e) or {}).get("edema_signal") for e in ("OD", "OS")),
        "organs": [s["key"] for s in snapshot if s["key"] in ORGAN_WORDS and s["state"] in ("signal", "exploratory")],
        "next": review.next_action if review and review.next_action in NEXT_STEPS else None,
        "signed": bool(review),
    }


def _join(words: list, conj: str) -> str:
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} {conj} {words[-1]}"


def template(f: dict, lang: str) -> str:
    t = TEXT[lang]
    parts = [t["intro"]]
    if f["eyes"]["OD"] == f["eyes"]["OS"]:
        both = "eyes" if lang == "en" else "dois olhos"
        parts.append(t["eye"][f["eyes"]["OD"]].format(eye=both).replace("do seu dois", "dos seus dois"))
    else:
        parts += [t["eye"][f["eyes"][e]].format(eye=t["eyes"][e]) for e in ("OD", "OS")]
    if f["edema"]:
        parts.append(t["edema"])
    if f["organs"]:
        parts.append(t["organs"].format(organs=_join([ORGAN_WORDS[o][lang == "pt"] for o in f["organs"]], t["and"])))
    parts.append(t["next"][f["next"]])
    if f["signed"]:
        parts.append(t["reviewed"])
    parts.append(t["caveat"])
    return " ".join(parts)


def check_outbound(text: str) -> str:
    for p in FORBIDDEN:
        if p.search(text):
            raise RestrictedPayload(f"text matches restricted pattern {p.pattern}")
    return text


def _acceptable(text: str, base: str) -> bool:
    """Gemini may reword, never add: no digits, no identifiers, bounded length, the caveat kept."""
    return (bool(text) and len(text) <= MAX_CHARS and not _DIGIT.search(text)
            and not any(p.search(text) for p in FORBIDDEN) and len(text) >= len(base) // 3)


def _remember(cache: OrderedDict, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > CACHE_SIZE:
        cache.popitem(last=False)
    return value


def summary(f: dict, lang: str = "en") -> dict:
    """{text, source}. The template is sent to Gemini, never the case record."""
    lang = lang if lang in LANGS else "en"
    base = template(f, lang)
    if not gemini_enabled():
        return {"text": base, "source": "template"}
    key = hashlib.sha256(f"{lang}|{base}".encode()).hexdigest()
    if key in _summaries:
        return _summaries[key]
    prompt = (f"Rewrite this note for a patient in warm, simple {LANGS[lang]} at a 6th-grade reading level, as if a "
              "caring nurse is speaking. Keep every fact and the final sentence about the computer program. Do not add "
              "any fact, number, drug, test, time frame or advice that is not in the note. At most 120 words. Reply "
              "with the rewritten note only.\n\nNote: " + check_outbound(base))
    try:
        from google import genai
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        text = (client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text or "").strip()
    except Exception as e:                                   # noqa: BLE001 - the template is always a safe answer
        log.warning("gemini patient summary failed: %s", e)
        return {"text": base, "source": "template (Gemini unavailable)"}
    if not _acceptable(text, base):
        return {"text": base, "source": "template (Gemini output failed the safety check)"}
    return _remember(_summaries, key, {"text": text, "source": f"Gemini ({GEMINI_MODEL}), checked"})


def speech(text: str, client: httpx.Client | None = None) -> bytes:
    """MP3 of the text read by ElevenLabs; cached by text."""
    if not voice_enabled():
        raise ExplainerUnavailable("Audio needs ELEVENLABS_API_KEY.")
    check_outbound(text)
    key = hashlib.sha256(f"{ELEVEN_VOICE}|{ELEVEN_MODEL}|{text}".encode()).hexdigest()
    if key in _audio:
        return _audio[key]
    own = client is None
    client = client or httpx.Client()
    try:
        r = client.post(ELEVEN_URL.format(voice=ELEVEN_VOICE), params={"output_format": "mp3_44100_128"},
                        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "accept": "audio/mpeg"},
                        json={"text": text, "model_id": ELEVEN_MODEL,
                              "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}, timeout=45)
        if r.status_code != 200:
            raise ExplainerUnavailable(f"ElevenLabs returned {r.status_code}.")
        return _remember(_audio, key, r.content)
    except httpx.HTTPError as e:
        raise ExplainerUnavailable("ElevenLabs is unreachable.") from e
    finally:
        if own:
            client.close()

