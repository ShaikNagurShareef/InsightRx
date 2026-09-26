"""
Oculomics view model: the retina as a non-invasive window onto systemic health.

Maps RetiLink's evidence (recorded history, validated retinal models, research association models) onto organ
systems for a single patient (whole-body snapshot) and for a clinician's screened panel. Every organ carries its
evidence tier so the UI can be compelling without over-claiming: nothing here is a diagnosis.
"""
from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Case, ModelRun

TIER_STATE = {"established": "clear", "research": "recorded", "exploratory": "exploratory", "evaluating": "unknown",
              "future": "na"}

# Organ systems in display order. `conditions` are mBRSET history fields / systemic model heads.
ORGANS = [
    {"key": "eyes", "label": "Eyes", "what": "Diabetic retinopathy and macular edema", "conditions": [],
     "feature": "Microaneurysms, haemorrhages and exudates seen directly in the photo"},
    {"key": "heart", "label": "Heart and blood vessels", "what": "Hypertension, vascular disease, prior heart attack",
     "conditions": ["systemic_hypertension", "vascular_disease", "acute_myocardial_infarction"],
     "feature": "Arteriolar narrowing and vessel changes mirror blood pressure and vascular health"},
    {"key": "kidneys", "label": "Kidneys", "what": "Diabetic kidney involvement", "conditions": ["nephropathy"],
     "feature": "Retinal and kidney small vessels share the same diabetic microvascular damage"},
    {"key": "nerves", "label": "Nerves and feet", "what": "Diabetic neuropathy and foot complications",
     "conditions": ["neuropathy", "diabetic_foot"],
     "feature": "The retina is neural tissue; microvascular disease tracks peripheral nerve damage"},
    {"key": "metabolism", "label": "Metabolism", "what": "Obesity, diabetes history and treatment", "conditions": ["obesity"],
     "feature": "Diabetes duration and control shape every other organ's risk"},
    {"key": "brain", "label": "Brain", "what": "Stroke and cognitive decline", "conditions": [],
     "feature": "The optic nerve extends from the brain; associations are active research"},
]
# Positions of organ markers in the body-map SVG (viewBox 0 0 320 470)
POS = {"eyes": (160, 60), "brain": (160, 30), "heart": (176, 148), "metabolism": (140, 178),
       "kidneys": (166, 212), "nerves": (160, 440)}
STATE_TEXT = {"signal": "Signal to review", "exploratory": "Exploratory signal", "recorded": "Known condition",
              "clear": "No finding recorded", "unknown": "History unknown", "na": "Not evaluated"}


def evidence_tier(organ, metrics):
    """Tier and headline number for an organ, from the measured metrics (never assumed)."""
    im = (metrics or {}).get("image_metrics.json") or {}
    cv = (metrics or {}).get("systemic_cv.json") or {}
    if organ["key"] == "eyes":
        auc = (im.get("patient_dr_referable") or {}).get("auroc")
        return {"tier": "established", "label": "Validated in RetiLink",
                "detail": f"Referable DR AUROC {auc:.3f} on held-out patients" if auc else "Validated on held-out patients"}
    if organ["key"] == "brain":
        return {"tier": "future", "label": "Future research", "detail": "No brain endpoint in mBRSET, so RetiLink makes no claim"}
    heads = [cv[c] for c in organ["conditions"] if c in cv]
    if organ["key"] == "heart" and "cardiovascular" in cv:
        heads.append(cv["cardiovascular"])
    if not heads:
        return {"tier": "future", "label": "Not yet modelled", "detail": "Recorded history only"}
    best = max(heads, key=lambda h: h["chosen_metrics"]["auroc"])
    auc = best["chosen_metrics"]["auroc"]
    if any(h["enabled"] for h in heads):
        return {"tier": "research", "label": "Research signal", "detail": f"Best cross-validated AUROC {auc:.2f}"}
    lo = best["chosen_metrics"]["auroc_ci95"][0]
    if lo > 0.5:
        return {"tier": "exploratory", "label": "Exploratory",
                "detail": f"Best cross-validated AUROC {auc:.2f}, below the release gate"}
    return {"tier": "evaluating", "label": "Near chance", "detail": f"Cross-validated AUROC {auc:.2f}; interval includes chance"}


def _systemic_state(organ, conditions, systemic):
    recorded = [c for c in organ["conditions"] if (conditions.get(c) or {}).get("value") == "present"]
    absent = [c for c in organ["conditions"] if (conditions.get(c) or {}).get("value") == "absent"]
    extra = ["cardiovascular"] if organ["key"] == "heart" else ["dm_complications"] if organ["key"] in ("kidneys", "nerves") else []
    # a model signal only matters for what is not already known: recorded conditions take precedence
    fresh = [c for c in organ["conditions"] + extra if c not in recorded]
    signals = [c for c in fresh if (systemic.get(c) or {}).get("status") == "Research signal"]
    exploratory = [c for c in fresh if (systemic.get(c) or {}).get("status") == "Exploratory signal"]
    # precedence: research-grade signal on something not yet known > known condition > exploratory signal
    if signals:
        state = "signal"
    elif recorded:
        state, signals = "recorded", exploratory
    elif exploratory:
        state, signals = "exploratory", exploratory
    elif organ["conditions"] and len(absent) == len(organ["conditions"]):
        state = "clear"
    elif organ["conditions"]:
        state = "unknown"
    else:
        state = "na"
    return state, recorded, signals


def patient_snapshot(patient, result, labels, metrics):
    """One entry per organ for a patient: state, what is recorded, what the models flag."""
    conditions = patient.conditions or {}
    systemic = (result or {}).get("systemic", {})
    out = []
    for o in ORGANS:
        if o["key"] == "eyes":
            overall = (result or {}).get("overall")
            state = {"Referable DR signal": "signal", "No model finding": "clear"}.get(
                overall, "unknown" if overall else "na")
            note = overall or "Not analysed yet"
            if state == "unknown":
                out.append({**o, "state": "unknown", "state_text": overall, "note": "One or both eyes lack an assessable photo",
                            "pos": POS[o["key"]], "evidence": evidence_tier(o, metrics)})
                continue
            recorded, signals = [], (["Referable DR"] if state == "signal" else [])
        elif o["key"] == "metabolism":
            state, recorded, signals = _systemic_state(o, conditions, systemic)
            bits = []
            if patient.dm_time is not None:
                bits.append(f"diabetes {patient.dm_time:g} years")
            if patient.insulin == "yes":
                bits.append("on insulin")
            note = ", ".join(bits) or "No metabolic history recorded"
        else:
            state, recorded, signals = _systemic_state(o, conditions, systemic)
            note = "; ".join(filter(None, [
                ("Recorded: " + ", ".join(labels.get(c, c) for c in recorded)) if recorded else "",
                (("Model signal: " if state == "signal" else "Exploratory model signal: ")
                 + ", ".join(labels.get(c, c) for c in signals)) if signals else ""])) or STATE_TEXT[state]
        out.append({**o, "state": state, "state_text": STATE_TEXT[state], "note": note, "pos": POS[o["key"]],
                    "evidence": evidence_tier(o, metrics)})
    return out


def panel_stats(db: Session, cases: list[Case], labels, metrics):
    """Population view over the cases this clinician may see (latest completed analysis per case)."""
    latest = {}
    ids = [c.id for c in cases]
    if ids:
        for r in db.scalars(select(ModelRun).where(ModelRun.case_id.in_(ids), ModelRun.status == "completed")
                            .order_by(ModelRun.id)):
            latest[r.case_id] = r.result or {}
    analysed = [c for c in cases if c.id in latest]
    dr = Counter(latest[c.id].get("overall", "Not analysed") for c in analysed)
    snaps = {c.id: patient_snapshot(c.patient, latest.get(c.id), labels, metrics) for c in cases}
    organs = []
    review = []
    for i, o in enumerate(ORGANS):
        tally = Counter(snaps[c.id][i]["state"] for c in cases)
        counts = {k: tally[k] for k in STATE_TEXT}          # plain dict: every state present, no Counter methods
        organs.append({**o, "counts": counts, "pos": POS[o["key"]], "evidence": evidence_tier(o, metrics),
                       "flag": counts["signal"] + counts["exploratory"] + counts["recorded"]})
    for c in analysed:
        snap = snaps[c.id]
        flags = [s for s in snap if s["state"] in ("signal", "exploratory")]
        unknown = [s for s in snap if s["state"] == "unknown"]
        if flags or unknown:
            review.append({"case": c, "flags": flags, "unknown": unknown, "snap": snap})
    review.sort(key=lambda r: (-len(r["flags"]), -len(r["unknown"])))
    return {"n_cases": len(cases), "n_analysed": len(analysed), "dr": dr, "organs": organs, "review": review[:10],
            "n_review": len(review)}


def evidence_map(metrics):
    """Organs coloured by evidence tier (for the 'How it works' body map)."""
    out = []
    for o in ORGANS:
        ev = evidence_tier(o, metrics)
        out.append({**o, "pos": POS[o["key"]], "evidence": ev, "state": TIER_STATE[ev["tier"]], "state_text": ev["label"]})
    return out
