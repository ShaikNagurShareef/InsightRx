"""
Consultation drafting and evidence briefs.

Case packages are assembled deterministically from case data (every statement traces to a field).
Gemini is only ever sent a generic clinical question plus curated public passages; the processing-policy guard
rejects any payload that contains case, patient, image or model-run content (spec SEC02).
"""
import json
import os
import re

from .evidence import retrieve

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
PROMPT_VERSION = "evidence-brief-v1"


class RestrictedPayload(Exception):
    pass


FORBIDDEN = [re.compile(p, re.I) for p in (r"\bpatient[_ ]?(id|ref)\b", r"\bRL-P\d+", r"\bcase\s*#?\d+", r"\.jpe?g\b",
                                          r"\bmbrset[_ ]?(id|patient|file)", r"\bmodel[_ ]run\b", r"\bdob\b",
                                          r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")]


def guard(payload: dict):
    """Only {'question': str, 'passages': [...]} with no identifiers may leave the research environment."""
    if set(payload) - {"question", "passages"}:
        raise RestrictedPayload(f"field(s) not allowed: {sorted(set(payload) - {'question', 'passages'})}")
    text = json.dumps(payload)
    for p in FORBIDDEN:
        if p.search(text):
            raise RestrictedPayload(f"payload matches restricted pattern {p.pattern}")
    return True


def evidence_brief(question: str, topics=()):
    refs = retrieve(question, topics)
    if not refs:
        return {"answer": "No supporting passage in the approved reference set. No evidence-based answer is given.",
                "references": [], "source": "template"}
    template = " ".join(f"[{i + 1}] {r['passage']}" for i, r in enumerate(refs))
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return {"answer": template, "references": refs, "source": "template (no GEMINI_API_KEY)"}
    payload = {"question": question, "passages": [{"n": i + 1, "text": r["passage"]} for i, r in enumerate(refs)]}
    try:
        guard(payload)
        from google import genai
        client = genai.Client(api_key=key)
        prompt = ("You answer a clinician's generic question using ONLY the numbered passages. Cite passages as [n] "
                  "after each sentence. If the passages do not answer the question, reply exactly "
                  "'Not supported by the approved references.' Do not give patient-specific advice, diagnoses, "
                  "treatment orders or urgency. Max 90 words.\n\n" + json.dumps(payload))
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = (resp.text or "").strip()
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        if not text or (not cited and "Not supported" not in text) or any(c < 1 or c > len(refs) for c in cited):
            return {"answer": template, "references": refs, "source": "template (LLM output failed citation check)"}
        return {"answer": text, "references": refs, "source": f"gemini:{GEMINI_MODEL}:{PROMPT_VERSION}"}
    except RestrictedPayload as e:
        return {"answer": template, "references": refs, "source": f"template (blocked: {e})"}
    except Exception as e:  # network / quota / SDK failure -> manual template
        return {"answer": template, "references": refs, "source": f"template (LLM unavailable: {type(e).__name__})"}


TOPIC_QUESTIONS = {
    "retinal": "Which diabetic retinopathy findings warrant referral to an ophthalmologist?",
    "systemic_hypertension": "What is known about retinal microvascular signs and hypertension?",
    "nephropathy": "How is kidney involvement in diabetes assessed?",
    "vascular_disease": "What is known about retinal microvascular signs and cardiovascular disease?",
    "diabetic_foot": "How often should people with diabetes receive a foot evaluation?",
    "neuropathy": "How often should people with diabetes receive a foot evaluation?",
}


def draft_package(case, patient, review, run, images, topic, question):
    """Deterministic, traceable package. Each fact lists the field it came from."""
    res = (run.result if run else {}) or {}
    facts = [
        {"text": f"Encounter {case.encounter_date}; device: {case.device}.", "source": "case.encounter_date, case.device"},
        {"text": f"Images: {sum(1 for i in images if i.laterality == 'OD')} right eye, "
                 f"{sum(1 for i in images if i.laterality == 'OS')} left eye.", "source": "images.laterality"},
    ]
    if run:
        for eye, name in (("OD", "Right eye"), ("OS", "Left eye")):
            e = res.get("eyes", {}).get(eye, {})
            facts.append({"text": f"{name}: {e.get('status', 'not assessed')} "
                                  f"({e.get('n_assessable', 0)} assessable of {e.get('n_images', 0)} images).",
                          "source": f"model_run #{run.id} ({run.source})"})
        facts.append({"text": f"Model screening summary: {res.get('overall')} [{run.source.upper()}, {run.model_version}, "
                              f"thresholds {run.threshold_version}].", "source": f"model_run #{run.id}"})
        thr = res.get("thresholds", {}).get("dr_referable")
        why = [f"{eye} max score {e['score']:.2f} vs threshold {thr:.2f}" + (f", ICDR estimate {e['max_icdr']}" if "max_icdr" in e else "")
               for eye, e in res.get("eyes", {}).items() if "score" in e and thr is not None]
        if why:
            facts.append({"text": "Model explanation: " + "; ".join(why) + ". Attention maps available in the case "
                                  "workspace (model attention, not lesion segmentation).", "source": f"model_run #{run.id}"})
        sig = [f"{v['label']} ({v['score']:.2f}, {v['inputs']}, CV AUROC {v['cv_auroc']})"
               for v in res.get("systemic", {}).values() if v.get("status") == "Research signal"]
        if sig:
            facts.append({"text": "Research association signals (not diagnoses): " + "; ".join(sig) + ".",
                          "source": f"model_run #{run.id}"})
    facts.append({"text": f"Referring clinician interpretation: {review.interpretation}", "source": f"review #{review.id} (signed)"})
    if review.override_reason:
        facts.append({"text": f"Clinician override reason: {review.override_reason}", "source": f"review #{review.id}"})
    hist = []
    for code, c in (patient.conditions or {}).items():
        hist.append(f"{code.replace('_', ' ')}: {c.get('value', 'unknown')} ({c.get('source', 'unspecified source')})")
    facts.append({"text": "Recorded history - " + ("; ".join(hist) if hist else "none recorded") + ".",
                  "source": "patient.conditions (reported, not adjudicated)"})
    if case.symptoms:
        facts.append({"text": f"Clinician-entered symptoms/context: {case.symptoms}", "source": "case.symptoms"})
    gaps = []
    if run and not res.get("complete"):
        gaps.append("Eye coverage incomplete - one or both eyes lack an assessable image.")
    if not run:
        gaps.append("No model result available; package relies on clinician review only.")
    for code, c in (patient.conditions or {}).items():
        if c.get("value") == "unknown":
            gaps.append(f"{code.replace('_', ' ')} history unknown.")
    brief = evidence_brief(TOPIC_QUESTIONS.get(topic, question), topics=[topic])
    return {"topic": topic, "question": question, "facts": facts, "gaps": gaps,
            "limitations": res.get("limitations", []), "evidence": brief,
            "requested_response": "Please acknowledge, request any missing information, and return a signed response "
                                  "with your recommended next action."}
