"""Per-user statistics and a readable activity feed for the inbox, scoped to what the user may see."""
from datetime import timedelta, timezone
from statistics import median

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import AuditEvent, Barrier, Case, Image, ModelRun, Referral, Review, Task, User, now

PHRASES = {
    "case_created": "opened a screening for {ref}",
    "model_run": "ran the screening models on {ref}",
    "review_signed": "signed an interpretation for {ref}",
    "referral_sent": "sent a consultation for {ref}",
    "referral_opened": "opened the consultation for {ref}",
    "referral_stage": "moved the consultation for {ref} to {stage}",
    "info_provided": "answered an information request for {ref}",
    "barrier_added": "recorded an access barrier for {ref}",
    "barrier_resolved": "resolved an access barrier for {ref}",
    "owner_reassigned": "reassigned the consultation for {ref}",
}


def _aware(d):
    return d.replace(tzinfo=timezone.utc) if d is not None and d.tzinfo is None else d


def ago(d):
    s = (now() - _aware(d)).total_seconds()
    if s < 3600:
        return f"{max(1, int(s // 60))} min ago"
    if s < 86400:
        return f"{int(s // 3600)} h ago"
    days = int(s // 86400)
    return "yesterday" if days == 1 else f"{days} days ago"


def feed(db: Session, user: User, case_ids, limit=12):
    """Recent workflow events on cases the user can see, as sentences."""
    if not case_ids:
        return []
    actors = {u.id: u for u in db.scalars(select(User).where(User.tenant_id == user.tenant_id))}
    refs = dict(db.execute(select(Case.id, Case.patient_id).where(Case.id.in_(case_ids))).all())
    from .models import Patient
    pref = dict(db.execute(select(Patient.id, Patient.ref).where(Patient.id.in_(set(refs.values())))).all())
    rows = db.scalars(select(AuditEvent).where(AuditEvent.case_id.in_(case_ids), AuditEvent.action.in_(PHRASES))
                      .order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc()).limit(limit)).all()
    out = []
    for e in rows:
        a = actors.get(e.actor_id)
        stage = e.detail.split("-> ")[-1].rstrip(". ") if e.action == "referral_stage" else ""
        out.append({"who": a.name if a else "System", "you": bool(a and a.id == user.id),
                    "text": PHRASES[e.action].format(ref=pref.get(refs.get(e.case_id), "a patient"), stage=stage.lower()),
                    "case_id": e.case_id, "when": ago(e.created_at), "action": e.action})
    return out


def daily(db: Session, user: User, case_ids, days=14):
    """Events per day for the last `days` days (the user's own actions, or all visible ones for operators)."""
    if not case_ids:
        return [0] * days
    since = now() - timedelta(days=days)
    q = select(AuditEvent.created_at).where(AuditEvent.case_id.in_(case_ids), AuditEvent.created_at >= since,
                                            AuditEvent.action.in_(PHRASES))
    if user.role != "operator":
        q = q.where(AuditEvent.actor_id == user.id)
    counts = [0] * days
    for (t,) in db.execute(q):
        i = days - 1 - int((now() - _aware(t)).total_seconds() // 86400)
        if 0 <= i < days:
            counts[i] += 1
    return counts


def week_stats(db: Session, user: User, case_ids):
    """Four role-specific numbers for the last 7 days (plus context)."""
    wk = now() - timedelta(days=7)
    ids = case_ids or [-1]
    ev = lambda action, mine=True: db.query(AuditEvent).filter(
        AuditEvent.case_id.in_(ids), AuditEvent.action == action, AuditEvent.created_at >= wk,
        *( [AuditEvent.actor_id == user.id] if mine else [])).count()
    if user.role == "operator":
        runs = db.scalars(select(ModelRun).where(ModelRun.case_id.in_(ids), ModelRun.created_at >= wk)).all()
        unable = sum((r.result or {}).get("overall") in ("Unable to assess", "Assessment incomplete") for r in runs)
        photos = db.query(Image).filter(Image.case_id.in_(ids), Image.created_at >= wk).count()
        return [("Screenings opened", ev("case_created"), "this week"), ("Photos captured", photos, "this week"),
                ("Analyses run", len(runs), "this week"),
                ("Needed recapture", unable, f"of {len(runs)} analyses" if runs else "no analyses yet")]
    if user.role == "referring":
        closed = db.query(Referral).filter(Referral.sender_id == user.id, Referral.stage == "Closed").count()
        sent_all = db.query(Referral).filter(Referral.sender_id == user.id).count()
        return [("Results reviewed", ev("review_signed"), "signed this week"),
                ("Consultations sent", ev("referral_sent"), "this week"),
                ("Loops closed", closed, f"of {sent_all} consultations sent"),
                ("Waiting on you", db.query(Task).filter(Task.assignee_id == user.id, Task.status == "open").count(), "open actions")]
    if user.role == "specialist":
        mine = db.scalars(select(Referral).where(Referral.recipient_id == user.id)).all()
        lat = [(_aware(r.acknowledged_at) - _aware(r.sent_at)).total_seconds() / 3600 for r in mine if r.acknowledged_at]
        responded = sum(r.stage in ("Response received", "Closed") for r in mine)
        return [("Consultations received", sum(_aware(r.sent_at) >= wk for r in mine), "this week"),
                ("Median time to accept", f"{median(lat):.1f} h" if lat else "None yet", f"over {len(lat)} accepted"),
                ("Signed responses", responded, f"of {len(mine)} received"),
                ("Awaiting your acceptance", sum(r.stage in ("Sent", "Needs information") for r in mine), "right now")]
    if user.role == "coordinator":
        refs = db.scalars(select(Referral).where(Referral.tenant_id == user.tenant_id)).all()
        barriers = db.scalars(select(Barrier).where(Barrier.tenant_id == user.tenant_id)).all()
        return [("Appointments booked", sum(bool(r.appointment.get("date")) for r in refs), "consultations with a visit booked"),
                ("Barriers resolved", sum(b.resolved for b in barriers), f"of {len(barriers)} recorded"),
                ("To schedule", db.query(Task).filter(Task.assignee_id == user.id, Task.status == "open").count(), "open tasks"),
                ("Closed with an answer", sum(r.stage == "Closed" for r in refs), f"of {len(refs)} consultations")]
    runs = db.query(ModelRun).filter(ModelRun.tenant_id == user.tenant_id).count()
    return [("Screenings analysed", runs, "all time"),
            ("Consultations", db.query(Referral).filter(Referral.tenant_id == user.tenant_id).count(), "all time"),
            ("Signed reviews", db.query(Review).filter(Review.tenant_id == user.tenant_id).count(), "all time"),
            ("Active users", db.query(User).filter(User.tenant_id == user.tenant_id, User.active.is_(True)).count(), "in this organisation")]
