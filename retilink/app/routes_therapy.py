"""
Therapeutics layer routes: therapy options per case, target explorer (AlphaFold / PDB), trials, the CMS NPI Registry
(refer outside the network), CMS quality and billing, and the manufacturer medical-information channel.

Included by main.py after its helpers are defined.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import external
from . import oculomics as oc
from . import therapeutics as tx
from . import workflow as wf
from .consult import TOPICS
from .db import get_db
from .main import (COMPOSITE_LABELS, SYSTEMIC_LABELS, TZ, case_for, ctx, current_user, get_service, latest_run,
                   patient_line, require, templates, visible_cases)
from .models import MedInfoRequest, Referral, Tenant, User, now

router = APIRouter()
CLINICIANS = ("referring", "specialist")
ANSWERED_STAGES = ("Response received", "Closed")


def _fresh_result(db, case):
    run = latest_run(db, case)
    return run.result if run and run.status == "completed" and run.case_version == case.version else None


def _snapshot(patient, res):
    return oc.patient_snapshot(patient, res, {**SYSTEMIC_LABELS, **COMPOSITE_LABELS}, get_service().metrics)


def case_findings(db, case):
    res = _fresh_result(db, case)
    return res, tx.findings(res, _snapshot(case.patient, res))


# ------------------------------------------------------------------ therapy for one patient
@router.get("/cases/{case_id}/therapy", response_class=HTMLResponse)
def case_therapy(case_id: int, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    case = case_for(db, user, case_id)
    res, fnd = case_findings(db, case)
    meds = case.patient.medications or []
    keys = [f["key"] for f in fnd]
    signed = wf.current_review(db, case) is not None
    answered = db.scalar(select(Referral).where(Referral.case_id == case.id, Referral.stage.in_(ANSWERED_STAGES))) is not None
    gap, gap_text = tx.care_gap(case.status, signed, answered)
    return templates.TemplateResponse(request, "case_therapy.html", ctx(
        request, user, db, active="patients", case=case, res=res, fnd=fnd, meds=meds,
        options=tx.options_for(fnd, meds), current_alerts=tx.merge_alerts(tx.check_interactions(meds, (), keys)),
        sponsored=tx.sponsored_for(fnd), targets=tx.targets_for(fnd),
        trials=tx.trial_matches(fnd, case.patient.age, case.patient.sex, per_topic=2), cms=tx.cms(), gap=gap, gap_text=gap_text,
        signed=signed, patient_line=patient_line(case.patient), topics=TOPICS))


# ------------------------------------------------------------------ target explorer
def _panel_counts(db, user):
    """How many of this clinician's screened patients carry each finding (drives 'patients this target touches')."""
    counts, members = {}, {}
    for c in visible_cases(db, user):
        _, fnd = case_findings(db, c)
        for f in fnd:                                   # 'metabolism' (diabetes) applies to every screened patient
            counts[f["key"]] = counts.get(f["key"], 0) + 1
            members.setdefault(f["key"], []).append(c)
    return counts, members


@router.get("/therapeutics", response_class=HTMLResponse)
def therapeutics_page(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    counts, _ = _panel_counts(db, user)
    groups = []
    for o in oc.ORGANS:
        ts = [tx.target(t["gene"]) for t in tx.targets() if t["organ"] == o["key"]]
        if ts:
            groups.append({"organ": o, "targets": ts})
    return templates.TemplateResponse(request, "therapeutics.html", ctx(
        request, user, db, active="therapeutics", groups=groups, counts=counts,
        findings=tx.catalogue()["findings"], classes=tx.catalogue()["classes"]))


@router.get("/therapeutics/targets/{gene}", response_class=HTMLResponse)
def target_page(gene: str, request: Request, view: str = "", user: User = Depends(current_user),
                db: Session = Depends(get_db)):
    t = tx.target(gene.upper())
    if not t:
        wf.error(404, "not_found", "Unknown target.")
    view = view if view in ("model", "complex") and (view != "complex" or t["has_complex"]) else \
        ("complex" if t["has_complex"] else "model")
    _, members = _panel_counts(db, user)
    patients = {c.id: c for k in t["findings"] for c in members.get(k, [])}
    fnd = [{"key": k, "label": tx.catalogue()["findings"][k]["label"]} for k in t["findings"]]
    classes = [c for c in tx.catalogue()["classes"] if gene.upper() in c["targets"]]
    return templates.TemplateResponse(request, "target.html", ctx(
        request, user, db, active="therapeutics", t=t, view=view, patients=list(patients.values()),
        classes=classes, trials=tx.trial_matches(fnd, per_topic=3), findings=tx.catalogue()["findings"]))


# ------------------------------------------------------------------ CMS NPI Registry: refer outside the network
@router.get("/cases/{case_id}/refer-out", response_class=HTMLResponse)
def refer_out(case_id: int, request: Request, topic: str = "retinal", zip: str = "", user: User = Depends(current_user),
              db: Session = Depends(get_db)):
    case = case_for(db, user, case_id)
    require(user, "referring")
    topic = topic if topic in TOPICS else "retinal"
    taxonomy = external.taxonomy_for(topic)
    z = external.clean_zip(zip)
    found = external.nppes(taxonomy, z)
    return templates.TemplateResponse(request, "refer_out.html", ctx(
        request, user, db, active="patients", case=case, topic=topic, topics=TOPICS, taxonomy=taxonomy, zip=z,
        providers=found["items"], source=found["source"], nppes_url=tx.cms()["nppes_url"]))


@router.get("/cases/{case_id}/letter", response_class=HTMLResponse)
def referral_letter(case_id: int, request: Request, npi: str = "", topic: str = "retinal",
                    user: User = Depends(current_user), db: Session = Depends(get_db)):
    case = case_for(db, user, case_id)
    require(user, "referring")
    provider = external.provider(npi)
    if not provider:
        wf.error(404, "not_found", "No provider with that NPI in the registry.")
    res, fnd = case_findings(db, case)
    review = wf.current_review(db, case)
    wf.audit(db, case.tenant_id, case.id, user.id, "referral_letter", f"out-of-network letter for NPI {provider['npi']}",
             case.version)
    db.commit()
    return templates.TemplateResponse(request, "letter.html", ctx(
        request, user, db, active="patients", case=case, provider=provider, res=res, fnd=fnd, review=review,
        topic=TOPICS.get(topic, TOPICS["retinal"]), today=datetime.now(TZ).strftime("%B %-d, %Y"),
        patient_line=patient_line(case.patient), meds=case.patient.medications or []))


# ------------------------------------------------------------------ CMS quality and billing roll-up
@router.get("/quality", response_class=HTMLResponse)
def quality_page(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    year = str(datetime.now(TZ).year)
    rows = {"open": 0, "pending": 0, "closed": 0}
    screened = gradable = 0
    for c in visible_cases(db, user):
        if not c.encounter_date.startswith(year):
            continue
        screened += 1
        res = _fresh_result(db, c)
        gradable += bool(res and res.get("overall") not in ("Unable to assess", None))
        answered = db.scalar(select(Referral).where(Referral.case_id == c.id, Referral.stage.in_(ANSWERED_STAGES))) is not None
        gap, _ = tx.care_gap(c.status, wf.current_review(db, c) is not None, answered)
        rows[gap] += 1
    return templates.TemplateResponse(request, "quality.html", ctx(
        request, user, db, active="quality", cms=tx.cms(), year=year, screened=screened, gradable=gradable, gaps=rows))


# ------------------------------------------------------------------ manufacturer medical-information channel
def _desk(db):
    return db.scalar(select(User).where(User.role == "medinfo", User.active.is_(True)).order_by(User.id))


@router.post("/medinfo")
async def medinfo_create(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    require(user, *CLINICIANS)
    f = await request.form()
    case = case_for(db, user, int(f.get("case_id") or 0))
    known = {d for c in tx.catalogue()["classes"] for d in c["drugs"]}
    drug = (f.get("drug") or "").strip().lower()
    question = (f.get("question") or "").strip()[:1500]
    if drug not in known:
        wf.error(422, "unknown_drug", "Choose a drug from the therapy options.")
    if len(question) < 10:
        wf.error(422, "missing_question", "Write the question for the medical-information team.")
    desk = _desk(db)
    if not desk:
        wf.error(503, "no_desk", "No medical-information desk is connected.")
    _, fnd = case_findings(db, case)
    req = MedInfoRequest(tenant_id=user.tenant_id, requester_id=user.id, desk_tenant_id=desk.tenant_id, drug=drug,
                         therapy_class=(f.get("therapy_class") or "")[:60], question=question,
                         context=tx.deidentified_context(case.patient.age, fnd))
    db.add(req)
    db.flush()
    wf.audit(db, user.tenant_id, case.id, user.id, "medinfo_request",
             f"MI-{req.id} to {db.get(User, desk.id).name}: {drug} (de-identified context only)", case.version)
    for u in db.scalars(select(User).where(User.tenant_id == desk.tenant_id, User.role == "medinfo")):
        wf.notify(db, u, f"New medical-information request MI-{req.id} ({drug})", "/medinfo")
    db.commit()
    return RedirectResponse(f"/medinfo?msg=Request+MI-{req.id}+sent.+You+will+be+notified+when+it+is+answered.", 303)


@router.get("/medinfo", response_class=HTMLResponse)
def medinfo_page(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    q = select(MedInfoRequest)
    q = q.where(MedInfoRequest.desk_tenant_id == user.tenant_id) if user.role == "medinfo" else \
        q.where(MedInfoRequest.requester_id == user.id)
    reqs = db.scalars(q.order_by(MedInfoRequest.status.desc(), MedInfoRequest.id.desc())).all()
    return templates.TemplateResponse(request, "medinfo.html", ctx(request, user, db, active="medinfo", reqs=reqs,
                                                                     tenants=_tenant_names(db)))


def _tenant_names(db):
    return {t.id: t.name for t in db.scalars(select(Tenant))}


@router.post("/medinfo/{rid}/answer")
async def medinfo_answer(rid: int, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    require(user, "medinfo")
    req = db.get(MedInfoRequest, rid)
    if not req or req.desk_tenant_id != user.tenant_id:
        wf.error(404, "not_found", "Request not found.")
    if req.status == "Answered":
        wf.error(409, "already_answered", "This request was already answered.")
    f = await request.form()
    answer, source = (f.get("answer") or "").strip()[:4000], (f.get("answer_source") or "").strip()[:200]
    if len(answer) < 10 or not source:
        wf.error(422, "incomplete_answer", "Write the answer and cite the label section or reference it is based on.")
    req.answer, req.answer_source, req.status = answer, source, "Answered"
    req.answered_by, req.answered_at = user.id, now()
    wf.audit(db, user.tenant_id, None, user.id, "medinfo_answer", f"MI-{req.id} answered")
    wf.notify(db, req.requester, f"Medical information answered (MI-{req.id}, {req.drug})", "/medinfo")
    db.commit()
    return RedirectResponse("/medinfo?msg=Answer+sent+to+the+clinician.", 303)
