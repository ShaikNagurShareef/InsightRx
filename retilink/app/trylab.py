"""'Try an image': ad-hoc analysis of up to 4 photos with optional patient details, re-runnable for 1 hour."""
import secrets
from datetime import timedelta
from types import SimpleNamespace

from sqlalchemy.orm import Session

from .models import TryImage, now

KEEP = timedelta(hours=1)
CONDITIONS = ["systemic_hypertension", "nephropathy", "vascular_disease", "acute_myocardial_infarction", "neuropathy",
              "diabetic_foot", "obesity"]


def _num(v, lo, hi):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if lo <= x <= hi else None


def parse_details(form) -> dict:
    """Validated patient details from a form; unknown stays None (never coerced to 'no')."""
    d = {"age": _num(form.get("age"), 0, 120), "sex": form.get("sex") if form.get("sex") in ("female", "male") else None,
         "dm_time": _num(form.get("dm_time"), 0, 90),
         "insulin": form.get("insulin") if form.get("insulin") in ("yes", "no") else "unknown",
         "oral_treatment": form.get("oral_treatment") if form.get("oral_treatment") in ("yes", "no") else "unknown",
         "conditions": {c: (form.get(f"cond_{c}") if form.get(f"cond_{c}") in ("present", "absent") else "unknown")
                        for c in CONDITIONS},
         "medications": _meds(form)}
    return d


def _meds(form) -> list:
    """Current medicines: picker values plus a comma-separated 'other' field, validated to generic-name form."""
    from .therapeutics import normalize_meds
    picked = form.getlist("med") if hasattr(form, "getlist") else list(form.get("med") or [])
    return normalize_meds(list(picked) + [form.get("med_other") or ""])


def model_inputs(d: dict) -> dict:
    """The metadata vector the systemic models read (NaN = unknown)."""
    nan = float("nan")
    yn = {"yes": 1.0, "no": 0.0}
    return {"age": d["age"] if d["age"] is not None else nan, "sex": {"male": 1.0, "female": 0.0}.get(d["sex"], nan),
            "dm_time": d["dm_time"] if d["dm_time"] is not None else nan, "insulin": yn.get(d["insulin"], nan),
            "oraltreatment_dm": yn.get(d["oral_treatment"], nan)}


def as_patient(d: dict):
    """A patient-like object for the whole-body snapshot."""
    conds = {c: {"value": v, "source": "entered in Try an image" if v != "unknown" else "not entered",
                 "date": "", "verification": "reported"} for c, v in d["conditions"].items()}
    return SimpleNamespace(conditions=conds, dm_time=d["dm_time"], insulin=d["insulin"], medications=d["medications"])


EYES = ("OD", "OS", "unknown")


def store(db: Session, user, blobs) -> str:
    """blobs: [(idx, name, data, eye)]. The eye is kept in the name column as 'EYE::name' (no schema change)."""
    purge(db)
    token = secrets.token_urlsafe(16)
    for i, name, data, eye in blobs:
        db.add(TryImage(token=token, tenant_id=user.tenant_id, user_id=user.id, idx=i,
                        name=f"{eye}::{name}"[:200], data=data))
    db.commit()
    return token


def load(db: Session, user, token: str):
    rows = (db.query(TryImage).filter(TryImage.token == token, TryImage.user_id == user.id,
                                      TryImage.created_at >= now() - KEEP).order_by(TryImage.idx).all())
    out = []
    for r in rows:
        eye, _, name = r.name.partition("::") if "::" in r.name else ("unknown", "", r.name)
        out.append((r.idx, name, r.data, eye if eye in EYES else "unknown"))
    return out


def purge(db: Session):
    db.query(TryImage).filter(TryImage.created_at < now() - KEEP).delete()


EVIDENCE_TOPICS = {  # finding -> curated reference topics
    "dr": (["dr", "referral", "referable"], "Referable diabetic retinopathy"),
    "edema": (["macular", "edema"], "Macular edema"),
    "heart": (["hypertension", "cardiovascular"], "Heart and blood vessels"),
    "kidneys": (["nephropathy", "kidney"], "Kidneys"),
    "nerves": (["diabetic_foot", "foot", "neuropathy"], "Nerves and feet"),
}


def evidence_for(overall, any_edema, snapshot):
    """Guideline passages relevant to what this screening found (curated set only)."""
    from .evidence import retrieve
    wanted = []
    if overall == "Referable DR signal":
        wanted.append("dr")
    if any_edema:
        wanted.append("edema")
    wanted += [o["key"] for o in snapshot if o["key"] in EVIDENCE_TOPICS and o["state"] in ("signal", "recorded", "exploratory")]
    cards, seen = [], set()
    for k in wanted:
        topics, label = EVIDENCE_TOPICS[k]
        for ref in retrieve(" ".join(topics), topics, k=1):
            if ref["id"] not in seen:
                seen.add(ref["id"])
                cards.append({"finding": label, **ref})
    return cards
