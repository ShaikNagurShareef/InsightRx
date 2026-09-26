"""Server-side clinical and referral state rules, signatures, audit, tasks and notifications."""
import hashlib
import json
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import AuditEvent, Case, Image, Notification, Referral, Review, Task, User, now

# ------------------------------------------------------------------ referral state machine (spec §13)
REFERRAL_TRANSITIONS = {
    "Sent": {"Acknowledged", "Needs information", "Declined", "Cancelled"},
    "Needs information": {"Acknowledged", "Declined", "Cancelled"},
    "Acknowledged": {"Needs information", "Scheduled", "Response received", "Unreachable", "Patient declined", "Cancelled"},
    "Scheduled": {"Visit recorded", "Scheduled", "Acknowledged", "Unreachable", "Patient declined", "Cancelled"},
    "Visit recorded": {"Response received", "Cancelled"},
    "Response received": {"Closed", "Needs information"},
    "Closed": set(), "Declined": set(), "Cancelled": set(), "Unreachable": set(), "Patient declined": set(),
}
TERMINAL = {"Closed", "Declined", "Cancelled", "Unreachable", "Patient declined"}
ALTERNATIVE_DISPOSITIONS = {"Declined", "Cancelled", "Unreachable", "Patient declined"}
FUNNEL = ["Sent", "Acknowledged", "Scheduled", "Visit recorded", "Response received", "Closed"]

# SIMULATED demo policy (spec: due dates are operational simulation settings, not clinical timing)
DEMO_DUE = {"acknowledge": timedelta(hours=24), "schedule": timedelta(hours=48), "review": timedelta(hours=24),
            "respond": timedelta(hours=72)}


def error(status, code, msg, retryable=False):
    raise HTTPException(status_code=status, detail={"code": code, "safe_message": msg, "retryable": retryable})


def transition(db: Session, ref: Referral, new: str, actor: User, detail=""):
    if new not in REFERRAL_TRANSITIONS.get(ref.stage, set()):
        error(409, "illegal_transition", f"Cannot move referral from {ref.stage} to {new}.")
    if new == "Closed":
        from .models import Message
        signed = db.scalar(select(Message).where(Message.referral_id == ref.id, Message.kind == "response",
                                                 Message.signature != ""))
        if not signed:
            error(409, "no_signed_response", "Closing requires a signed specialist response.")
        ref.closure_code = "completed"
    if new in ALTERNATIVE_DISPOSITIONS:
        ref.closure_code = new.lower().replace(" ", "_")
    old, ref.stage = ref.stage, new
    if new == "Acknowledged" and not ref.acknowledged_at:
        ref.acknowledged_at = now()
    audit(db, ref.tenant_id, ref.case_id, actor.id, "referral_stage", f"#{ref.id} {old} -> {new}. {detail}".strip())
    if new in TERMINAL:
        for t in db.scalars(select(Task).where(Task.referral_id == ref.id, Task.status == "open")):
            t.status = "cancelled" if new != "Closed" else "done"


# ------------------------------------------------------------------ signatures / hashes
def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def review_signature(case: Case, images, run_id, summary, signer: User, ts: datetime) -> str:
    return digest({"case": case.id, "version": case.version, "images": sorted(i.sha256 for i in images),
                   "model_run": run_id, "summary": summary, "signer": signer.id, "at": ts.isoformat()})


def current_review(db: Session, case: Case):
    """Latest signed review, only if it is bound to the current case version."""
    r = db.scalar(select(Review).where(Review.case_id == case.id).order_by(Review.id.desc()))
    return r if r and r.case_version == case.version else None


def active_images(db: Session, case: Case):
    return list(db.scalars(select(Image).where(Image.case_id == case.id, Image.superseded_by.is_(None))
                           .order_by(Image.laterality, Image.id)))


def bump_version(db: Session, case: Case, actor: User, reason: str):
    """New images / laterality / intake edits invalidate the unsigned draft and any pending package."""
    had_review = current_review(db, case) is not None
    case.version += 1
    if case.status not in ("Draft",):
        case.status = "Images ready"
    audit(db, case.tenant_id, case.id, actor.id, "case_version", f"v{case.version}: {reason}"
          + (" (previous signed review is now superseded; re-review required)" if had_review else ""), case.version)


# ------------------------------------------------------------------ audit / tasks / notifications
def audit(db: Session, tenant_id, case_id, actor_id, action, detail="", version=None):
    db.add(AuditEvent(tenant_id=tenant_id, case_id=case_id, actor_id=actor_id, action=action, detail=detail,
                      version=version))


def add_task(db: Session, case: Case, kind, title, assignee_id, referral_id=None, due=None):
    exists = db.scalar(select(Task).where(Task.case_id == case.id, Task.kind == kind, Task.referral_id == referral_id,
                                          Task.status == "open"))
    if exists:
        return exists
    t = Task(tenant_id=case.tenant_id, case_id=case.id, referral_id=referral_id, kind=kind, title=title,
             assignee_id=assignee_id, due_at=(now() + due) if due else None)
    db.add(t)
    return t


def close_tasks(db: Session, case_id, kind, referral_id=None):
    for t in db.scalars(select(Task).where(Task.case_id == case_id, Task.kind == kind, Task.status == "open",
                                           Task.referral_id == referral_id)):
        t.status = "done"


def in_quiet_hours(user: User, at: datetime) -> bool:
    p = user.prefs or {}
    qs, qe = p.get("quiet_start"), p.get("quiet_end")
    if qs is None or qe is None:
        return False
    h = at.astimezone().hour
    return (qs <= h or h < qe) if qs > qe else (qs <= h < qe)


def notify(db: Session, user: User, body: str, link: str):
    """In-app notification without clinical detail; digest users / quiet hours batch instead of pinging."""
    digest_mode = (user.prefs or {}).get("mode") == "digest" or in_quiet_hours(user, now())
    db.add(Notification(tenant_id=user.tenant_id, user_id=user.id, body=body, link=link, digest=digest_mode))


def is_overdue(due_at) -> bool:
    if not due_at:
        return False
    if due_at.tzinfo is None:
        due_at = due_at.replace(tzinfo=timezone.utc)
    return due_at < now()
