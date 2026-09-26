"""Guided 'Consult a specialist': suggested topics, matching specialists and draft wording from the screening."""

TOPICS = {
    "retinal": {"label": "Diabetic retinopathy", "organ": "eyes", "match": ("retina", "ophthalm"),
                "question": "Please assess for referable diabetic retinopathy and advise on management and follow-up."},
    "systemic_hypertension": {"label": "Heart and blood vessels", "organ": "heart", "match": ("cardio",),
                              "question": "Retinal screening and history suggest cardiovascular risk. Please review blood pressure and cardiovascular risk management."},
    "nephropathy": {"label": "Kidneys", "organ": "kidneys", "match": ("nephro",),
                    "question": "Retinal microvascular findings and diabetes history raise kidney concerns. Please advise on renal assessment (eGFR, urine albumin) and follow-up."},
    "neuropathy": {"label": "Nerves and feet", "organ": "nerves", "match": ("neuro", "podiatr", "foot"),
                   "question": "Please assess for diabetic neuropathy and foot risk, and advise on follow-up."},
}


def suggested_topics(result, snapshot):
    """Topics worth consulting on, most important first, each with the reason it is suggested."""
    out = []
    overall = (result or {}).get("overall")
    if overall == "Referable DR signal":
        out.append(("retinal", "Referable diabetic retinopathy signal"))
    elif overall in ("Assessment incomplete", "Unable to assess"):
        out.append(("retinal", "Photos could not fully assess the retina"))
    by_organ = {o["key"]: o for o in snapshot or []}
    for key, t in TOPICS.items():
        if key == "retinal":
            continue
        o = by_organ.get(t["organ"])
        if o and o["state"] in ("signal", "recorded", "exploratory"):
            out.append((key, f"{o['label']}: {o['state_text'].lower()}"))
    if not any(k == "retinal" for k, _ in out):
        out.append(("retinal", "Routine retinal opinion"))
    return out


def specialists_for(topic, specialists):
    """(specialist, suggested) pairs, suggested ones first."""
    keys = TOPICS.get(topic, TOPICS["retinal"])["match"]
    tagged = [(s, any(k in (s.specialty or "").lower() for k in keys)) for s in specialists]
    return sorted(tagged, key=lambda t: not t[1])


def draft_interpretation(result):
    overall = (result or {}).get("overall")
    eyes = (result or {}).get("eyes", {})
    pos = [n for e, n in (("OD", "right"), ("OS", "left")) if eyes.get(e, {}).get("status") == "Referable DR signal"]
    if overall == "Referable DR signal":
        where = " and ".join(pos) + (" eyes" if len(pos) > 1 else " eye") if pos else "at least one photo"
        return f"I reviewed the photos: findings consistent with referable diabetic retinopathy in the {where}. I agree with the screening result."
    if overall == "No model finding":
        return "I reviewed the photos: no referable diabetic retinopathy seen. Requesting specialist opinion for the question below."
    return "Photos do not allow a complete assessment; requesting specialist examination."
