"""
Demo caseload from REAL mBRSET test-split patients (never used for training), analysed by the real models.

Four weeks of backdated workflow (reviews, consultations, scheduling, responses, tasks, notifications) around real
photographs and real model output. Patient references are synthetic (RL-P02xx); recorded history comes from the
mBRSET record of that patient. mBRSET is credentialed PhysioNet data: this caseload is for the local research
workspace only and must not be loaded into a public deployment without a data-use review.
"""
import hashlib
import json
import os
import random
from datetime import timedelta, timezone

from . import workflow as wf
from .db import APP_ROOT, IMAGE_STORE
from .models import AuditEvent, Barrier, Case, Image, Message, ModelRun, Notification, Patient, Referral, Review, Task, now

SPLIT = "/data/users3/nshaik3/Projects/Oculomics/OculoMoE/cache/mbrset/splits_protocol_P1.json"
CONDITIONS = ["systemic_hypertension", "nephropathy", "vascular_disease", "acute_myocardial_infarction", "neuropathy",
              "diabetic_foot", "obesity"]


def available():
    from retilink.ml import config as C
    return os.path.exists(SPLIT) and os.path.exists(C.MBRSET_LABELS)


def _aware(d):
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d


def _past(t):
    """Demo timestamps never land in the future."""
    return min(_aware(t), now() - timedelta(minutes=10))


def _event(db, tenant, case_id, actor, action, detail, at, version=None):
    db.add(AuditEvent(tenant_id=tenant, case_id=case_id, actor_id=actor, action=action, detail=detail, version=version,
                      created_at=at))


def select_patients(exclude=()):
    """Test-split patients by category, from the dataset's own labels. -> (dict category -> [patient ids], images, patients)"""
    from retilink.ml.data import load_tables
    images, patients = load_tables()
    test = set(json.load(open(SPLIT))["splits"]["0"]["test"]) - set(exclude)
    t = images[images["patient_id"].isin(test)]
    g = {pid: d for pid, d in t.groupby("patient_id")}
    good = [p for p, d in g.items() if (d["quality_poor"] == 0).all() and d["icdr"].notna().all()]
    rng = random.Random(21)
    cats = {
        "referable": [p for p in good if (g[p]["icdr"] >= 2).any()],
        "negative": [p for p in good if (g[p]["icdr"] == 0).all()],
        "mild": [p for p in good if g[p]["icdr"].max() == 1],
        "poor": [p for p, d in g.items() if (d["quality_poor"] == 1).sum() >= 2],
    }
    for v in cats.values():
        rng.shuffle(v)
    return cats, images, patients.set_index("patient_id")


def _history(row):
    out = {}
    for c in CONDITIONS:
        v = row.get(c)
        val = "unknown" if v != v or v is None else ("present" if float(v) == 1 else "absent")
        out[c] = {"value": val, "source": "mBRSET record" if val != "unknown" else "not recorded",
                  "date": "2026-08-14" if val != "unknown" else "", "verification": "reported"}
    return out


def populate(db, U, tenant, exclude=()):
    """Add ~24 backdated cases built from real test-split patients. Skips (no fake data) if mBRSET is unreachable."""
    if not available():
        return 0
    from .vision import get_service
    svc = get_service()
    rng = random.Random(11)
    cats, images, pat = select_patients(exclude)
    used = set()

    def take(cat, want=None):
        for pid in cats[cat]:
            if pid in used:
                continue
            if want and pat.loc[pid].get(want) != 1:
                continue
            used.add(pid)
            return pid
        return take(cat) if want else None

    # (owner, category, one_eye, reviewed, referral stage, specialist key, history field to prefer)
    plan = ([("pcp", "referable", False, True, s, "ret", None) for s in
             ("Closed", "Closed", "Response received", "Scheduled", "Acknowledged", "Sent", "Sent")]
            + [("pcp", "negative", False, True, None, None, None)] * 4
            + [("pcp", "negative", False, False, None, None, None), ("pcp", "mild", False, False, None, None, None),
               ("pcp", "referable", False, False, None, None, None), ("pcp", "poor", False, False, None, None, None),
               ("pcp", "negative", True, False, None, None, None)]
            + [("pcp2", "referable", False, True, s, "ret2", None) for s in ("Closed", "Scheduled", "Sent")]
            + [("pcp2", "referable", False, True, "Sent", "neph", "nephropathy"),
               ("pcp2", "negative", False, True, "Acknowledged", "cardio", "systemic_hypertension"),
               ("pcp2", "mild", False, True, None, None, None), ("pcp2", "negative", False, True, None, None, None),
               ("pcp2", "poor", True, False, None, None, None)])
    random.Random(4).shuffle(plan)            # interleave referrals, reviews and new screenings over the period
    start = now() - timedelta(days=20)
    made = 0
    for n, (owner, cat, one_eye, reviewed, stage, spk, want) in enumerate(plan):
        pid = take(cat, want)
        if pid is None:
            continue
        rec = pat.loc[pid]
        rows = images[images["patient_id"] == pid].sort_values("image_id")
        if one_eye:
            rows = rows[rows["eye"] == "OD"]
        t0 = _past(start + timedelta(days=n * 20 / len(plan) + rng.uniform(0, .5), hours=rng.randint(0, 6)))
        yn = lambda v: "unknown" if v != v or v is None else ("yes" if float(v) == 1 else "no")
        p = Patient(tenant_id=tenant.id, ref=f"RL-P{210 + n:04d}", age=float(rec["age"]) if rec["age"] == rec["age"] else None,
                    sex={1.0: "male", 0.0: "female"}.get(rec["sex"]), dm_time=float(rec["dm_time"]) if rec["dm_time"] == rec["dm_time"] else None,
                    insulin=yn(rec["insulin"]), oral_treatment=yn(rec["oraltreatment_dm"]), conditions=_history(rec))
        db.add(p)
        db.flush()
        c = Case(tenant_id=tenant.id, patient_id=p.id, owner_id=U[owner].id, created_by=U["op"].id,
                 encounter_date=t0.date().isoformat(), device="Portable smartphone fundus camera (dilated)",
                 symptoms=rng.choice(["", "", "Reports blurred reading vision (clinician-entered).", "Annual diabetes review."]),
                 created_at=t0)
        db.add(c)
        db.flush()
        _event(db, tenant.id, c.id, U["op"].id, "case_created", f"patient {p.ref}", t0, 1)
        imgs = []
        for _, r in rows.iterrows():
            data = open(r["path"], "rb").read()
            sha = hashlib.sha256(data).hexdigest()
            path = ""
            if IMAGE_STORE == "fs":
                path = os.path.join(APP_ROOT, "images", f"t{tenant.id}", f"{sha}.jpg")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                if not os.path.exists(path):
                    with open(path, "wb") as fh:
                        fh.write(data)
            im = Image(tenant_id=tenant.id, case_id=c.id, sha256=sha, path=path, laterality=r["eye"], view="macula-centred",
                       source=c.device, width=1600, height=1600, uploaded_by=U["op"].id, data=None if path else data,
                       created_at=t0)
            db.add(im)
            imgs.append((im, r))
        db.flush()
        c.version, c.status = 2, "Images ready"
        _event(db, tenant.id, c.id, U["op"].id, "case_version", f"v2: {len(imgs)} image(s) uploaded", t0 + timedelta(minutes=3), 2)
        made += 1
        if svc.mode != "live":
            continue                      # no model available: leave the case ready for analysis, never invent output
        yv = lambda v: float(v) if v == v and v is not None else float("nan")
        res = svc.analyze([{"id": im.id, "path": im.path, "laterality": im.laterality,
                            **({} if im.path else {"bytes": im.data})} for im, _ in imgs],
                          {"age": yv(rec["age"]), "sex": yv(rec["sex"]), "dm_time": yv(rec["dm_time"]),
                           "insulin": yv(rec["insulin"]), "oraltreatment_dm": yv(rec["oraltreatment_dm"])})
        t1 = t0 + timedelta(minutes=6)
        run = ModelRun(tenant_id=tenant.id, case_id=c.id, case_version=2, image_ids=[im.id for im, _ in imgs],
                       model_version=res["model_version"], threshold_version=res["threshold_version"], source=res["source"],
                       status="completed", result=res, latency_ms=res.get("latency_ms", 0), created_at=t1)
        db.add(run)
        db.flush()
        _event(db, tenant.id, c.id, U["op"].id, "model_run", f"run #{run.id} [{run.source}] {res['overall']}", t1, 2)
        c.status = "Unable to assess" if res["overall"] == "Unable to assess" else "HCP review"
        if not reviewed:
            db.add(Task(tenant_id=tenant.id, case_id=c.id, kind="review", title=f"Review screening result for {p.ref}",
                        assignee_id=U[owner].id, due_at=t1 + wf.DEMO_DUE["review"], created_at=t1))
            db.add(Notification(tenant_id=tenant.id, user_id=U[owner].id, body=f"Screening result ready to review ({p.ref})",
                                link=f"/cases/{c.id}", created_at=t1, read=rng.random() < .4))
            continue
        # the clinician's reading follows the dataset grader (ground truth), so disagreements with the model are real
        grade = int(rows["icdr"].max()) if rows["icdr"].notna().any() else None
        truth_ref = grade is not None and grade >= 2
        model_ref = res["overall"] == "Referable DR signal"
        agree = truth_ref == model_ref
        names = ["no apparent DR", "mild NPDR", "moderate NPDR", "severe NPDR", "proliferative DR"]
        interp = f"On my review of both eyes: {names[grade] if grade is not None else 'not gradable'}."
        t2 = _past(t1 + timedelta(hours=rng.uniform(1, 12)))
        rv = Review(tenant_id=tenant.id, case_id=c.id, case_version=2, model_run_id=run.id, hcp_id=U[owner].id,
                    decision="accept" if agree else "disagree", interpretation=interp,
                    override_reason="" if agree else "My grading of the photos differs from the model output.",
                    next_action="refer_retina" if (truth_ref or stage) else "routine_rescreen",
                    systemic_notes={"systemic_hypertension": "reviewed_known"} if p.conditions["systemic_hypertension"]["value"] == "present" else {},
                    signature=wf.digest({"demo": n, "pid": int(pid)}), signed_at=t2)
        db.add(rv)
        db.flush()
        c.status = "Signed"
        _event(db, tenant.id, c.id, U[owner].id, "review_signed", f"{rv.decision}; sig {rv.signature[:12]}", t2, 2)
        if stage:
            _referral(db, rng, tenant, U, c, p, rv, owner, spk, stage, _past(t2 + timedelta(minutes=rng.randint(5, 50))), n, grade)
    _backdate_history(db, rng)
    return made


def _referral(db, rng, tenant, U, c, p, rv, owner, spk, stage, sent, n, grade):
    spec, sender, coord = U[spk], U[owner], U["coord"]
    topic = {"neph": "nephropathy", "cardio": "systemic_hypertension"}.get(spk, "retinal")
    q = {"retinal": "Please assess for referable diabetic retinopathy and advise on management and follow-up.",
         "nephropathy": "Recorded diabetic kidney involvement with retinal microvascular changes. Please advise on renal follow-up.",
         "systemic_hypertension": "Known hypertension on the record. Please review cardiovascular risk management."}[topic]
    pkg = {"topic": topic, "question": q, "facts": [{"text": f"Signed review: {rv.interpretation}", "source": f"review #{rv.id} (signed)"}],
           "gaps": [], "limitations": [], "evidence": {"answer": "", "references": []},
           "image_ids": [im.id for im in db.query(Image).filter(Image.case_id == c.id)]}
    r = Referral(tenant_id=tenant.id, case_id=c.id, review_id=rv.id, sender_id=sender.id, recipient_id=spec.id,
                 owner_id=spec.id, topic=topic, question=q, package=pkg, package_hash=wf.digest(pkg),
                 signature=wf.digest({"dr": n}), idempotency_key=f"demo-{n}", stage="Sent", sent_at=sent,
                 due_at=sent + wf.DEMO_DUE["acknowledge"])
    db.add(r)
    db.flush()
    _event(db, tenant.id, c.id, sender.id, "referral_sent", f"#{r.id} to {spec.name} ({topic})", sent, 2)
    order = ["Sent", "Acknowledged", "Scheduled", "Visit recorded", "Response received", "Closed"]
    t = sent
    if rng.random() < .7:
        r.opened_at = _past(sent + timedelta(hours=rng.uniform(.3, 6)))
        _event(db, tenant.id, c.id, spec.id, "referral_opened", f"#{r.id}", r.opened_at)
    for s in order[1:order.index(stage) + 1]:
        t = _past(t + timedelta(hours=rng.uniform(2, 16) if s == "Acknowledged" else rng.uniform(8, 36)))
        actor = spec if s in ("Acknowledged", "Visit recorded", "Response received") else sender if s == "Closed" else coord
        if s == "Acknowledged":
            r.acknowledged_at = t
        if s == "Scheduled":
            appt = t + timedelta(days=rng.randint(4, 12))
            r.appointment = {"date": appt.date().isoformat(), "time": rng.choice(["09:30", "11:00", "14:15"]),
                             "timezone": "America/New_York", "facility": "Peachtree Eye Clinic",
                             "confirmation": "phone call with patient"}
            if rng.random() < .5:
                db.add(Barrier(tenant_id=tenant.id, referral_id=r.id, category=rng.choice(["transport", "language", "contact"]),
                               note=rng.choice(["Needs a ride on appointment day", "Prefers Portuguese-speaking staff",
                                                "Only reachable after 5 pm"]), owner_id=coord.id, resolved=stage != "Scheduled"))
        if s == "Response received":
            finding = {2: "Moderate NPDR confirmed on examination, no centre-involving edema. Re-image in 6 months.",
                       3: "Severe NPDR confirmed. Starting treatment evaluation at our clinic.",
                       4: "Proliferative DR confirmed. Treatment scheduled at our clinic."}.get(grade, "Findings reviewed; continue primary-care management and annual screening.")
            db.add(Message(tenant_id=tenant.id, referral_id=r.id, author_id=spec.id, kind="response", body=finding,
                           recommendation="Treatment evaluation at specialist clinic" if (grade or 0) >= 3 else "Specialist follow-up visit",
                           signature=wf.digest({"resp": n}), created_at=t))
        _event(db, tenant.id, c.id, actor.id, "referral_stage", f"#{r.id} {order[order.index(s) - 1]} -> {s}.", t)
    r.stage = stage
    if stage == "Closed":
        r.closure_code = "completed"
    r.owner_id = {"Sent": spec.id, "Acknowledged": coord.id, "Scheduled": spec.id, "Response received": sender.id,
                  "Closed": sender.id}[stage]
    task = {"Sent": ("acknowledge", f"Acknowledge consultation RL-{r.id}", spec, wf.DEMO_DUE["acknowledge"]),
            "Acknowledged": ("schedule", f"Schedule appointment for RL-{r.id}", coord, wf.DEMO_DUE["schedule"]),
            "Scheduled": ("respond", f"Record visit and sign response for RL-{r.id}", spec, wf.DEMO_DUE["respond"]),
            "Response received": ("close", f"Acknowledge specialist response on RL-{r.id}", sender, wf.DEMO_DUE["review"])}.get(stage)
    if task:
        kind, title, who, due = task
        db.add(Task(tenant_id=tenant.id, case_id=c.id, referral_id=r.id, kind=kind, title=title, assignee_id=who.id,
                    due_at=t + due, created_at=t))
        db.add(Notification(tenant_id=tenant.id, user_id=who.id, body=title, link=f"/referrals/{r.id}", created_at=t,
                            read=rng.random() < .3))


def _backdate_history(db, rng):
    """Spread the earlier historical referral events over their real timeline instead of 'now'."""
    for r in db.query(Referral).filter(Referral.idempotency_key.like("seed-%")):
        t = _aware(r.sent_at)
        for e in db.query(AuditEvent).filter(AuditEvent.case_id == r.case_id).order_by(AuditEvent.id):
            t = t + timedelta(hours=rng.uniform(3, 30))
            e.created_at = min(t, now() - timedelta(hours=1))
