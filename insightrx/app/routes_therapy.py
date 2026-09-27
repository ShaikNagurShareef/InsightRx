"""
Therapeutics layer routes: therapy options per case, target explorer (AlphaFold / PDB), trials, the CMS NPI Registry
(refer outside the network), CMS quality and billing, and the manufacturer medical-information channel.

Included by main.py after its helpers are defined.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import copilot
from . import external
from . import oculomics as oc
from . import personalize
from . import therapeutics as tx
from . import trylab
from . import workflow as wf
from .consult import TOPICS
from .db import get_db
from .main import (COMPOSITE_LABELS, SYSTEMIC_LABELS, TZ, case_for, ctx, current_user, get_service, image_bytes,
                   latest_run, patient_line, require, templates, thumbnail, visible_cases)
from .models import MedInfoRequest, Referral, ScreenResult, Tenant, User, now

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
    b = case_bundle(db, case)
    gap, gap_text, signed = case_gap(db, case)
    return templates.TemplateResponse(request, "case_therapy.html", ctx(
        request, user, db, active="patients", case=case, res=b["res"], fnd=b["fnd"], meds=b["meds"],
        options=b["options"], current_alerts=b["current_alerts"], sponsored=b["sponsored"], priorities=b["priorities"],
        trials=b["trials"], cms=tx.cms(), gap=gap, gap_text=gap_text, signed=signed,
        patient_line=patient_line(case.patient), topics=TOPICS,
        copilot_enabled=copilot.enabled(), copilot_turns=copilot.transcript(db, user, f"case:{case.id}")))


def case_bundle(db, case, per_topic=2):
    res = _fresh_result(db, case)
    b = personalize.bundle(res, _snapshot(case.patient, res), case.patient.medications or [], case.patient.age,
                           case.patient.sex, per_topic=per_topic)
    return {**b, "res": res}


def case_gap(db, case):
    signed = wf.current_review(db, case) is not None
    answered = db.scalar(select(Referral).where(Referral.case_id == case.id, Referral.stage.in_(ANSWERED_STAGES))) is not None
    gap, gap_text = tx.care_gap(case.status, signed, answered)
    return gap, gap_text, signed


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


def patient_context(db, user, gene, case_id=0, screen=""):
    """This target's personalised priority for one patient (a saved case or a held Screen result), or None."""
    if case_id:
        case = case_for(db, user, case_id)
        b = personalize.bundle(_fresh_result(db, case), _snapshot(case.patient, _fresh_result(db, case)),
                               case.patient.medications or [], with_trials=False)
        ref, back = case.patient.ref, f"/cases/{case.id}/therapy"
    elif screen:
        sr = _screen_result(db, user, screen)
        d = sr.details
        b = personalize.bundle(sr.result, _snapshot(trylab.as_patient(d), sr.result), d.get("medications") or [],
                               with_trials=False)
        ref, back = "this screening", None
    else:
        return None
    hit = next((p for p in b["priorities"] if p["gene"] == gene), None)
    rank = [p["gene"] for p in b["priorities"]].index(gene) + 1 if hit else None
    return {**hit, "ref": ref, "rank": rank, "of": len(b["priorities"]), "back": back} if hit else \
        {"ref": ref, "priority": None, "why": "This target is not linked to the patient's current findings.", "back": back}


def _screen_result(db, user, token):
    sr = db.get(ScreenResult, token or "")
    if not sr or sr.user_id != user.id or sr.created_at.replace(tzinfo=sr.created_at.tzinfo or now().tzinfo) < now() - trylab.KEEP:
        wf.error(410, "screen_expired", "That screening is no longer held (kept for 1 hour). Screen the photos again.")
    return sr


@router.get("/therapeutics/targets/{gene}", response_class=HTMLResponse)
def target_page(gene: str, request: Request, view: str = "", case: int = 0, screen: str = "",
                user: User = Depends(current_user), db: Session = Depends(get_db)):
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
        classes=classes, trials=tx.trial_matches(fnd, per_topic=3), findings=tx.catalogue()["findings"],
        pc=patient_context(db, user, t["gene"], case, screen), case_id=case, screen=screen,
        si=personalize.structure_insights().get(t["gene"]) or {}))


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


# ------------------------------------------------------------------ PDF reports
def _pdf(data: bytes, name: str):
    return Response(data, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "private, no-store"})


def _eye_rows(res):
    rows = []
    for e, n in (("OD", "Right eye"), ("OS", "Left eye")):
        ev = (res or {}).get("eyes", {}).get(e)
        if ev:
            rows.append([n, ev.get("status", ""), f"{ev['score']:.2f}" if ev.get("score") is not None else "-",
                         str(ev.get("max_icdr", "-")), "yes" if ev.get("edema_signal") else "no"])
    return rows


def _image_caption(eye, r):
    name = {"OD": "Right eye", "OS": "Left eye"}.get(eye, "Eye not set")
    bits = [name, f"quality {r.get('quality', '?')}"]
    if r.get("p_dr") is not None and r.get("quality") != "unassessable":
        bits.append(f"DR score {r['p_dr']:.2f}")
    return ", ".join(bits)


def _patient_pdf(ref, subtitle, meta, res, snapshot, meds, age, sex, images, gap, gap_text, user):
    from . import reports
    b = personalize.bundle(res, snapshot, meds, age, sex, per_topic=3)
    return reports.patient_report({"ref": ref, "subtitle": subtitle, "meta": meta, "res": res, "eyes": _eye_rows(res),
                                   "systemic": (res or {}).get("systemic") or {}, "images": images, "bundle": b,
                                   "cms": tx.cms(), "gap": gap, "gap_text": gap_text, "clinician": user.name})


@router.get("/cases/{case_id}/report.pdf")
def case_report(case_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    case = case_for(db, user, case_id)
    res = _fresh_result(db, case)
    per = (res or {}).get("images") or {}
    images = [(thumbnail(image_bytes(i), 420), _image_caption(i.laterality, per.get(str(i.id)) or {}))
              for i in wf.active_images(db, case)[:4]]
    gap, gap_text, _ = case_gap(db, case)
    p = case.patient
    meta = [("Patient", f"{p.ref}. {patient_line(p)}".strip()), ("Encounter", f"{case.encounter_date}, {case.device}"),
            ("Owner", case.owner.name), ("Current medicines", ", ".join(p.medications or []) or "none recorded"),
            ("Model", f"{(res or {}).get('model_version', '-')} ({(res or {}).get('source', 'no current analysis')})")]
    pdf = _patient_pdf(p.ref, "Retinal screening to therapy options, target priorities and trials", meta, res,
                       _snapshot(p, res), p.medications or [], p.age, p.sex, images, gap, gap_text, user)
    wf.audit(db, case.tenant_id, case.id, user.id, "report_pdf", "personalised therapy and target report", case.version)
    db.commit()
    return _pdf(pdf, f"InsightRx_{p.ref}_therapy_report.pdf")


@router.get("/screen/report.pdf")
def screen_report(token: str = "", user: User = Depends(current_user), db: Session = Depends(get_db)):
    sr = _screen_result(db, user, token)
    d, res = sr.details, sr.result
    per = res.get("images") or {}
    images = [(thumbnail(data, 420), _image_caption(eye, per.get(str(i)) or {})) for i, _, data, eye in
              trylab.load(db, user, token)[:4]]
    bits = [f"{int(d['age'])} years" if d.get("age") is not None else "", d.get("sex") or "",
            f"diabetes for {d['dm_time']:g} years" if d.get("dm_time") is not None else ""]
    meta = [("Patient", ", ".join(b for b in bits if b) or "details not entered"),
            ("Current medicines", ", ".join(d.get("medications") or []) or "none entered"),
            ("Model", f"{res.get('model_version', '-')} ({res.get('source', '-')})"),
            ("Status", "Unsaved screening (held 1 hour). Save the patient to consult and track.")]
    gap, gap_text = tx.care_gap("", False, False)
    pdf = _patient_pdf("Screening report", "Retinal screening to therapy options, target priorities and trials", meta, res,
                       _snapshot(trylab.as_patient(d), res), d.get("medications") or [], d.get("age"), d.get("sex"),
                       images, gap, gap_text, user)
    wf.audit(db, user.tenant_id, None, user.id, "report_pdf", "screening therapy and target report (unsaved)")
    db.commit()
    return _pdf(pdf, "InsightRx_screening_therapy_report.pdf")


@router.get("/therapeutics/targets/{gene}/report.pdf")
def target_report(gene: str, case: int = 0, screen: str = "", user: User = Depends(current_user),
                  db: Session = Depends(get_db)):
    from . import reports
    t = tx.target(gene.upper())
    if not t:
        wf.error(404, "not_found", "Unknown target.")
    counts, _ = _panel_counts(db, user)
    panel_n = len(visible_cases(db, user))
    demand = max([counts.get(k, 0) for k in t["findings"]] + [0])
    fnd = [{"key": k, "label": tx.catalogue()["findings"][k]["label"]} for k in t["findings"]]
    classes = [c for c in tx.catalogue()["classes"] if t["gene"] in c["targets"]]
    pc = patient_context(db, user, t["gene"], case, screen)
    pdf = reports.target_report(t, demand, panel_n, tx.trial_matches(fnd, per_topic=4), classes,
                                pc if pc and pc.get("priority") is not None else None)
    return _pdf(pdf, f"InsightRx_{t['gene']}_target_dossier.pdf")


@router.get("/therapeutics/report.pdf")
def portfolio_report(user: User = Depends(current_user), db: Session = Depends(get_db)):
    from . import reports
    counts, _ = _panel_counts(db, user)
    panel_n = len(visible_cases(db, user))
    return _pdf(reports.portfolio_report(reports.portfolio_rows(counts, panel_n), panel_n), "InsightRx_target_portfolio.pdf")
