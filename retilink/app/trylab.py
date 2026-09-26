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
                        for c in CONDITIONS}}
    return d


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
    return SimpleNamespace(conditions=conds, dm_time=d["dm_time"], insulin=d["insulin"])


def store(db: Session, user, blobs) -> str:
    purge(db)
    token = secrets.token_urlsafe(16)
    for i, name, data in blobs:
        db.add(TryImage(token=token, tenant_id=user.tenant_id, user_id=user.id, idx=i, name=name[:200], data=data))
    db.commit()
    return token


def load(db: Session, user, token: str):
    rows = (db.query(TryImage).filter(TryImage.token == token, TryImage.user_id == user.id,
                                      TryImage.created_at >= now() - KEEP).order_by(TryImage.idx).all())
    return [(r.idx, r.name, r.data) for r in rows]


def purge(db: Session):
    db.query(TryImage).filter(TryImage.created_at < now() - KEEP).delete()
