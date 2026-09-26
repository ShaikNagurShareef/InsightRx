"""Insight Rx FastAPI application (server-rendered HCP workspace)."""
import hashlib
import io
import json
import os
import secrets
import uuid
from collections import Counter
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image as PILImage
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from . import workflow as wf
from .db import APP_ROOT, IMAGE_STORE, Base, SessionLocal, engine, ensure_columns, get_db
from . import activity as act
from . import consult
from . import oculomics as oc
from . import therapeutics as tx
from . import trylab
from .llm import draft_package, evidence_brief
from .models import (AppSetting, AuditEvent, Barrier, Case, Image, Message, ModelRun, Notification, Patient, Referral, Review,
                     Task, Tenant, User, now)
from .vision import COMPOSITE_LABELS, ICDR_NAMES, META_LABELS, SYSTEMIC_LABELS, get_service

HERE = os.path.dirname(__file__)
MAX_BYTES = int(os.environ.get("INSIGHTRX_MAX_IMAGE_MB", "10")) * 1024 * 1024   # Vercel: 4 (request body limit)
MAX_IMAGES = 8
CLINICAL_ROLES = {"referring"}

Base.metadata.create_all(engine)
ensure_columns()
if os.environ.get("INSIGHTRX_AUTOSEED", "1") == "1":         # fresh database -> synthetic demo workspace, no images
    from .seed import seed
    with SessionLocal() as _db:
        try:
            seed(_db, with_images=False, verbose=False)
        except IntegrityError:                                # another instance seeded concurrently
            _db.rollback()
app = FastAPI(title="Insight Rx")
ACCESS_CODE = os.environ.get("INSIGHTRX_ACCESS_CODE", "")
OPEN_PATHS = ("/static/", "/access", "/api/health", "/api/vision/register", "/favicon")


@app.middleware("http")
async def access_gate(request: Request, call_next):
    """Optional site-wide access code (keeps credentialed dataset images off the open web). Registered before the
    session middleware so it runs inside it and can read the session."""
    if ACCESS_CODE and not request.url.path.startswith(OPEN_PATHS) and not request.session.get("access_ok"):
        return RedirectResponse("/access?next=" + request.url.path, 303)
    return await call_next(request)


app.add_middleware(SessionMiddleware, secret_key=os.environ.get("INSIGHTRX_SECRET", secrets.token_hex(16)),
                   same_site="lax")
# static assets live in public/static so Vercel serves them from its CDN; locally FastAPI serves the same files
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "public", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
ASSET_V = hashlib.sha1(b"".join(open(os.path.join(STATIC_DIR, f), "rb").read()
                                for f in ("app.css", "app.js", "icons.svg", "molview.js"))).hexdigest()[:8]
TZ = ZoneInfo(os.environ.get("INSIGHTRX_TZ", "America/New_York"))
ROLE_LABELS = {"operator": "Screening operator", "referring": "Referring clinician", "specialist": "Specialist",
               "coordinator": "Care coordinator", "admin": "Administrator", "medinfo": "Medical information desk"}


def role_label(u):
    base = ROLE_LABELS.get(u.role, u.role)
    return f"{base}, {u.specialty}" if u.specialty and u.specialty.lower() not in base.lower() else base


def _aware(d):
    from datetime import timezone
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d


def fmt_dt(d):
    if not d:
        return "Not yet"
    return _aware(d).astimezone(TZ).strftime("%b %-d, %-I:%M %p %Z")


def fmt_due(d):
    delta = _aware(d) - now()
    hrs = delta.total_seconds() / 3600
    if hrs < 0:
        return "Overdue"
    return f"Due in {int(hrs)} h" if hrs < 48 else f"Due {fmt_dt(d)}"


templates = Jinja2Templates(directory=os.path.join(HERE, "templates"))
templates.env.globals.update(ICDR_NAMES=ICDR_NAMES, SYSTEMIC_LABELS=SYSTEMIC_LABELS, COMPOSITE_LABELS=COMPOSITE_LABELS,
                             META_LABELS=META_LABELS, is_overdue=wf.is_overdue, asset_v=ASSET_V, role_label=role_label,
                             COMMON_MEDS=tx.common_meds())
templates.env.filters["dt"] = fmt_dt
templates.env.filters["due"] = fmt_due


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    resp.headers.setdefault("Content-Security-Policy",
                            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
                            "script-src 'self'; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; "
                            "base-uri 'self'; form-action 'self'; object-src 'none'")
    if request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return resp


class LoginRequired(Exception):
    pass


@app.exception_handler(LoginRequired)
async def _login(request, exc):
    return RedirectResponse("/login", 303)


@app.exception_handler(HTTPException)
async def _http(request: Request, exc: HTTPException):
    if request.headers.get("accept", "").startswith("application/json") or request.url.path.startswith("/api/"):
        body = exc.detail if isinstance(exc.detail, dict) else {"code": "error", "safe_message": str(exc.detail)}
        body["request_id"] = uuid.uuid4().hex[:12]
        return JSONResponse(body, status_code=exc.status_code)
    msg = exc.detail["safe_message"] if isinstance(exc.detail, dict) else str(exc.detail)
    return templates.TemplateResponse(request, "error.html", {"user": None, "status": exc.status_code, "message": msg},
                                      status_code=exc.status_code)


# ------------------------------------------------------------------ auth / access
def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    uid = request.session.get("uid")
    user = db.get(User, uid) if uid else None
    if not user or not user.active:
        raise LoginRequired()
    return user


def require(user: User, *roles):
    if user.role not in roles:
        wf.error(403, "forbidden_role", f"The {user.role} role cannot perform this action.")


def case_for(db: Session, user: User, case_id: int) -> Case:
    case = db.get(Case, case_id)
    if not case or case.tenant_id != user.tenant_id:
        wf.error(404, "not_found", "Case not found.")       # never reveal other tenants' cases
    if user.role in ("operator", "coordinator"):
        return case
    if user.role == "referring" and case.owner_id == user.id:
        return case
    if user.role == "specialist" and db.scalar(select(Referral).where(
            Referral.case_id == case.id, Referral.recipient_id == user.id, Referral.stage != "Cancelled")):
        return case
    wf.error(403, "no_case_access", "You are not assigned to this case.")


def referral_for(db: Session, user: User, rid: int) -> Referral:
    ref = db.get(Referral, rid)
    if not ref or ref.tenant_id != user.tenant_id:
        wf.error(404, "not_found", "Referral not found.")
    if user.role == "coordinator" or user.id in (ref.sender_id, ref.recipient_id):
        return ref
    wf.error(403, "no_referral_access", "You are not a party to this referral.")


def ctx(request, user, db, **kw):
    unread = db.query(Notification).filter(Notification.user_id == user.id, Notification.read.is_(False)).count()
    open_tasks = db.query(Task).filter(Task.assignee_id == user.id, Task.status == "open").count()
    base = {"user": user, "unread": unread, "open_tasks": open_tasks, "vision_mode": get_service().mode,
            "msg": request.query_params.get("msg", ""), "active": ""}
    return {**base, **kw}


# ------------------------------------------------------------------ login
@app.get("/access", response_class=HTMLResponse)
def access_page(request: Request, next: str = "/login", error: str = ""):
    return templates.TemplateResponse(request, "access.html", {"user": None, "next": next if next.startswith("/") else "/login",
                                                               "error": error})


@app.post("/access")
def access_check(request: Request, code: str = Form(""), next: str = Form("/login")):
    if ACCESS_CODE and secrets.compare_digest(code.strip(), ACCESS_CODE):
        request.session["access_ok"] = True
        return RedirectResponse(next if next.startswith("/") and not next.startswith("//") else "/login", 303)
    return RedirectResponse("/access?error=1&next=" + (next if next.startswith("/") else "/login"), 303)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, db: Session = Depends(get_db)):
    users = db.scalars(select(User).where(User.active.is_(True)).order_by(User.tenant_id, User.id)).all()
    tenants = {t.id: t.name for t in db.scalars(select(Tenant))}
    svc = get_service()
    auc = ((svc.metrics.get("image_metrics.json") or {}).get("patient_dr_referable") or {}).get("auroc")
    return templates.TemplateResponse(request, "login.html", {"user": None, "users": users, "tenants": tenants,
                                                              "tiers": oc.evidence_map(svc.metrics), "auc": auc})


@app.post("/login")
def login(request: Request, user_id: int = Form(...), db: Session = Depends(get_db)):
    u = db.get(User, user_id)
    if not u:
        wf.error(404, "not_found", "Unknown user.")
    request.session["uid"] = u.id
    wf.audit(db, u.tenant_id, None, u.id, "login")
    db.commit()
    return RedirectResponse(home_for(u), 303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", 303)


def home_for(u: User) -> str:
    return {"operator": "/screen", "referring": "/screen", "admin": "/performance", "medinfo": "/medinfo"}.get(u.role, "/consults")


@app.get("/")
def root(request: Request, db: Session = Depends(get_db)):
    u = db.get(User, request.session.get("uid") or 0)
    return RedirectResponse(home_for(u) if u else "/login", 303)


@app.get("/inbox")
def inbox_redirect(request: Request):
    return RedirectResponse("/consults" + (f"?{request.url.query}" if request.url.query else ""), 303)


# ------------------------------------------------------------------ inbox (FR20)
@app.get("/consults", response_class=HTMLResponse)
def inbox(request: Request, filter: str = "", user: User = Depends(current_user), db: Session = Depends(get_db)):
    tasks = db.scalars(select(Task).where(Task.tenant_id == user.tenant_id, Task.assignee_id == user.id,
                                          Task.status == "open").order_by(Task.due_at.is_(None), Task.due_at)).all()
    if filter == "overdue":
        tasks = [t for t in tasks if wf.is_overdue(t.due_at)]
    q = select(Referral).where(Referral.tenant_id == user.tenant_id)
    if user.role != "coordinator":
        q = q.where(or_(Referral.sender_id == user.id, Referral.recipient_id == user.id))
    refs = db.scalars(q.order_by(Referral.sent_at.desc())).all()
    if filter == "awaiting":
        refs = [r for r in refs if r.stage in ("Sent", "Needs information")]
    if user.role in ("operator", "coordinator"):
        cases = db.scalars(select(Case).where(Case.tenant_id == user.tenant_id).order_by(Case.id.desc())).all()
    elif user.role == "referring":
        cases = db.scalars(select(Case).where(Case.owner_id == user.id).order_by(Case.id.desc())).all()
    else:
        cases = []
    if filter == "incomplete":
        cases = [c for c in cases if c.status in ("Draft", "Images ready", "Unable to assess")]
    results = {}
    if cases:
        for r in db.scalars(select(ModelRun).where(ModelRun.case_id.in_([c.id for c in cases]), ModelRun.status == "completed")
                            .order_by(ModelRun.id)):
            results[r.case_id] = (r.result or {}).get("overall")
    hour = datetime.now(TZ).hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    awaiting = sum(r.stage in ("Sent", "Needs information") and (user.role != "specialist" or r.recipient_id == user.id)
                   for r in refs)
    seen = [c.id for c in visible_cases(db, user)]
    return templates.TemplateResponse(request, "inbox.html", ctx(
        request, user, db, tasks=tasks, refs=refs, cases=cases, filter=filter, results=results, greeting=greeting,
        week=act.week_stats(db, user, seen), activity=act.feed(db, user, seen), spark=act.daily(db, user, seen),
        overdue=sum(wf.is_overdue(t.due_at) for t in tasks), awaiting=awaiting,
        in_review=sum(c.status == "HCP review" for c in cases), active="consults"))


# ------------------------------------------------------------------ case creation (FR01, FR02)
@app.get("/cases/new")
def new_case_redirect():
    return RedirectResponse("/screen", 303)


@app.get("/cases/new/form", response_class=HTMLResponse)
def new_case_page(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    require(user, "operator", "referring")
    hcps = db.scalars(select(User).where(User.tenant_id == user.tenant_id, User.role == "referring")).all()
    patients = db.scalars(select(Patient).where(Patient.tenant_id == user.tenant_id)).all()
    return templates.TemplateResponse(request, "case_new.html", ctx(request, user, db, hcps=hcps, patients=patients,
                                                                      conditions=SYSTEMIC_LABELS, active="new",
                                                                      today=datetime.now(TZ).date().isoformat()))


@app.post("/cases")
async def create_case(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    require(user, "operator", "referring")
    f = await request.form()
    if not f.get("device") or not f.get("encounter_date") or not (f.get("patient_id") or f.get("patient_ref")):
        wf.error(422, "missing_identity", "Patient reference, encounter date and device/source are required.")
    owner = db.get(User, int(f.get("owner_id") or user.id))
    if not owner or owner.tenant_id != user.tenant_id or owner.role != "referring":
        wf.error(422, "bad_owner", "Choose a referring HCP in your organization.")
    if f.get("patient_id"):
        patient = db.get(Patient, int(f["patient_id"]))
        if not patient or patient.tenant_id != user.tenant_id:
            wf.error(404, "not_found", "Patient not found.")
    else:
        num = lambda k: float(f[k]) if f.get(k) not in (None, "") else None
        conds = {c: {"value": f.get(f"cond_{c}", "unknown"), "source": "clinician-entered at intake",
                     "date": f.get("encounter_date"), "verification": "reported"} for c in SYSTEMIC_LABELS}
        patient = Patient(tenant_id=user.tenant_id, ref=f["patient_ref"].strip(), age=num("age"),
                          sex=f.get("sex") or None, dm_time=num("dm_time"), insulin=f.get("insulin", "unknown"),
                          oral_treatment=f.get("oral_treatment", "unknown"), conditions=conds,
                          medications=trylab.parse_details(f)["medications"])
        db.add(patient)
        db.flush()
    dup = db.scalar(select(Case).where(Case.patient_id == patient.id, Case.encounter_date == f["encounter_date"]))
    case = Case(tenant_id=user.tenant_id, patient_id=patient.id, owner_id=owner.id, created_by=user.id,
                encounter_date=f["encounter_date"], device=f["device"], symptoms=f.get("symptoms", ""),
                urgent_concern=bool(f.get("urgent_concern")))
    db.add(case)
    db.flush()
    wf.audit(db, user.tenant_id, case.id, user.id, "case_created",
             f"patient {patient.ref}" + (f"; WARNING duplicate encounter of case #{dup.id} (not merged)" if dup else ""), 1)
    # photos attached in the same step (validated before anything is stored)
    blobs = []
    for eye in ("OD", "OS"):
        for uf in f.getlist(f"files_{eye}"):
            if hasattr(uf, "read") and getattr(uf, "filename", ""):
                data = await uf.read()
                if data:
                    blobs.append((eye, data, _validate_upload(data)))
    if len(blobs) > MAX_IMAGES:
        wf.error(413, "too_many_images", f"At most {MAX_IMAGES} images per case.")
    seen = set()
    for eye, data, img in blobs:
        sha, path = _store(case, data, img)
        if sha in seen:
            continue
        seen.add(sha)
        db.add(Image(tenant_id=case.tenant_id, case_id=case.id, sha256=sha, path=path, laterality=eye,
                     view="macula-centred", source=case.device, width=img.size[0], height=img.size[1],
                     uploaded_by=user.id, data=None if path else data))
    if seen:
        case.status = "Images ready"
        wf.bump_version(db, case, user, f"{len(seen)} image(s) uploaded at intake")
        db.flush()
        run_analysis(db, case, user)
    db.commit()
    return RedirectResponse(f"/cases/{case.id}" + ("?msg=Case+created+and+analysed" if seen else "?msg=Case+created"), 303)


@app.post("/cases/{case_id}/intake")
async def edit_intake(case_id: int, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    case = case_for(db, user, case_id)
    require(user, "operator", "referring")
    f = await request.form()
    case.symptoms = f.get("symptoms", case.symptoms)
    if user.role == "referring":
        case.urgent_concern = bool(f.get("urgent_concern"))
    p = case.patient
    if "age" in f:                                   # model inputs (unknown stays unknown)
        d = trylab.parse_details(f)
        p.age, p.sex, p.dm_time, p.insulin, p.oral_treatment = d["age"], d["sex"], d["dm_time"], d["insulin"], d["oral_treatment"]
        p.medications = d["medications"]
    for c in SYSTEMIC_LABELS:
        v = f.get(f"cond_{c}")
        if v and v != (p.conditions or {}).get(c, {}).get("value"):
            conds = dict(p.conditions or {})
            conds[c] = {"value": v, "source": f"updated by {user.name}", "date": now().date().isoformat(),
                        "verification": "reported"}
            p.conditions = conds
    wf.bump_version(db, case, user, "clinical inputs edited")
    if f.get("rerun") and wf.active_images(db, case):
        run_analysis(db, case, user)
        db.commit()
        return RedirectResponse(f"/cases/{case_id}?tab=systemic&msg=Inputs+saved+and+analysis+re-run", 303)
    db.commit()
    return RedirectResponse(f"/cases/{case_id}?tab={f.get('tab', 'retinal')}&msg=Inputs+saved", 303)


# ------------------------------------------------------------------ images (FR03-FR05, FR08, FR10)
def _validate_upload(data: bytes):
    if len(data) > MAX_BYTES:
        wf.error(413, "too_large", "Images are limited to 10 MB.")
    if not (data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"):
        wf.error(415, "bad_type", "Only JPEG or PNG images are accepted.")
    PILImage.MAX_IMAGE_PIXELS = 40_000_000       # decompression-bomb bound
    try:
        img = PILImage.open(io.BytesIO(data))
        img.verify()
        img = PILImage.open(io.BytesIO(data))
        img.load()
    except Exception:
        wf.error(422, "corrupt_image", "The file could not be decoded as an image.")
    return img


def _store(case, data, img):
    """-> (sha256, path). Database store keeps bytes on the Image row instead (path '')."""
    sha = hashlib.sha256(data).hexdigest()
    if IMAGE_STORE == "db":
        return sha, ""
    ext = ".png" if img.format == "PNG" else ".jpg"
    path = os.path.join(APP_ROOT, "images", f"t{case.tenant_id}", f"{sha}{ext}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):      # content-addressed, immutable original (model input is never re-encoded)
        with open(path, "wb") as fh:
            fh.write(data)
    return sha, path


@app.post("/cases/{case_id}/images")
async def upload_images(case_id: int, request: Request, files: list[UploadFile] = File(...),
                        laterality: str = Form("unknown"), view: str = Form("macula-centred"),
                        user: User = Depends(current_user), db: Session = Depends(get_db)):
    case = case_for(db, user, case_id)
    require(user, "operator", "referring")
    existing = wf.active_images(db, case)
    if len(existing) + len(files) > MAX_IMAGES:
        wf.error(413, "too_many_images", f"At most {MAX_IMAGES} active images per case.")
    blobs = []
    for uf in files:                          # validate everything before storing anything (no partial uploads)
        data = await uf.read()
        blobs.append((uf.filename, data, _validate_upload(data)))
    added, dups = 0, []
    for name, data, img in blobs:
        sha, path = _store(case, data, img)
        if any(i.sha256 == sha for i in existing):
            dups.append(name)
            continue
        im = Image(tenant_id=case.tenant_id, case_id=case.id, sha256=sha, path=path,
                   laterality=laterality if laterality in ("OD", "OS") else "unknown", view=view,
                   source=case.device, width=img.size[0], height=img.size[1], uploaded_by=user.id,
                   data=None if path else data)
        db.add(im)
        existing.append(im)
        added += 1
    if added:
        if case.status == "Draft":
            case.status = "Images ready"
        wf.bump_version(db, case, user, f"{added} image(s) uploaded ({laterality})")
    if dups:
        wf.audit(db, case.tenant_id, case.id, user.id, "duplicate_upload", f"exact duplicate(s) reused: {', '.join(dups)}")
    db.commit()
    return RedirectResponse(f"/cases/{case_id}?msg=" + (f"Duplicate ignored: {', '.join(dups)}" if dups else "Uploaded"), 303)


@app.post("/images/{image_id}/laterality")
def set_laterality(image_id: int, laterality: str = Form(...), user: User = Depends(current_user),
                   db: Session = Depends(get_db)):
    im = db.get(Image, image_id)
    if not im:
        wf.error(404, "not_found", "Image not found.")
    case = case_for(db, user, im.case_id)
    require(user, "operator", "referring")
    if laterality not in ("OD", "OS", "unknown") or laterality == im.laterality:
        return RedirectResponse(f"/cases/{case.id}", 303)
    old, im.laterality = im.laterality, laterality
    wf.bump_version(db, case, user, f"image #{im.id} laterality {old} -> {laterality} (results invalidated)")
    db.commit()
    return RedirectResponse(f"/cases/{case.id}", 303)


@app.post("/images/{image_id}/recapture")
async def recapture(image_id: int, file: UploadFile = File(...), user: User = Depends(current_user),
                    db: Session = Depends(get_db)):
    im = db.get(Image, image_id)
    if not im:
        wf.error(404, "not_found", "Image not found.")
    case = case_for(db, user, im.case_id)
    require(user, "operator", "referring")
    data = await file.read()
    img = _validate_upload(data)
    sha, path = _store(case, data, img)
    new = Image(tenant_id=case.tenant_id, case_id=case.id, sha256=sha, path=path, laterality=im.laterality, view=im.view,
                source=case.device, width=img.size[0], height=img.size[1], uploaded_by=user.id,
                data=None if path else data)
    db.add(new)
    db.flush()
    im.superseded_by = new.id
    wf.bump_version(db, case, user, f"image #{im.id} recaptured as #{new.id} (earlier image superseded, kept)")
    db.commit()
    return RedirectResponse(f"/cases/{case.id}", 303)


THUMB_WIDTHS = (160, 240, 900)


def thumbnail(src: bytes, w: int) -> bytes:
    from .vision import _to_rgb
    img = PILImage.open(io.BytesIO(src))
    img.draft("RGB", (w * 2, w * 2))
    img = _to_rgb(img)
    img.thumbnail((w, w), PILImage.LANCZOS)
    out = io.BytesIO()
    img.save(out, "JPEG", quality=82, optimize=True, progressive=True)
    return out.getvalue()


@app.get("/images/{image_id}")
def image_file(image_id: int, request: Request, w: int = 0, user: User = Depends(current_user),
               db: Session = Depends(get_db)):
    im = db.get(Image, image_id)
    if not im:
        wf.error(404, "not_found", "Image not found.")
    case_for(db, user, im.case_id)
    if w:
        w = min(THUMB_WIDTHS, key=lambda x: abs(x - w))
        etag = f'"{im.sha256[:16]}-{w}"'
        headers = {"Cache-Control": "private, max-age=86400", "ETag": etag}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        path = os.path.join(APP_ROOT, "thumbs", f"{im.sha256}_{w}.jpg")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(thumbnail(image_bytes(im), w))
        return FileResponse(path, media_type="image/jpeg", headers=headers)
    if im.path:
        return FileResponse(im.path, headers={"Cache-Control": "private, max-age=300"})
    mt = "image/png" if im.data[:4] == b"\x89PNG" else "image/jpeg"
    return Response(im.data, media_type=mt, headers={"Cache-Control": "private, max-age=300"})


def image_bytes(im: Image) -> bytes:
    if im.path:
        with open(im.path, "rb") as fh:
            return fh.read()
    return im.data


@app.get("/images/{image_id}/explain")
def image_explain(image_id: int, head: str = "dr_referable", user: User = Depends(current_user),
                  db: Session = Depends(get_db)):
    """Model-attention overlay (FR19), computed on demand and cached per image x head x model version."""
    im = db.get(Image, image_id)
    if not im:
        wf.error(404, "not_found", "Image not found.")
    case_for(db, user, im.case_id)
    svc = get_service()
    if not head.replace("_", "").isalnum():
        wf.error(422, "bad_head", "Unknown head.")
    path = os.path.join(APP_ROOT, "explain", f"{im.sha256}_{head}_{hashlib.sha1(svc.version.encode()).hexdigest()[:10]}.png")
    if not os.path.exists(path):
        png = svc.explain(im.path or image_bytes(im), head)
        if png is None:
            wf.error(404, "no_explanation", "No explanation available for this output.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(png)
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=300"})


# ------------------------------------------------------------------ analysis (FR06-FR14)
def patient_meta(p: Patient):
    yn = {"yes": 1.0, "no": 0.0}
    return {"age": p.age, "sex": {"male": 1.0, "female": 0.0}.get(p.sex or ""), "dm_time": p.dm_time,
            "insulin": yn.get(p.insulin), "oraltreatment_dm": yn.get(p.oral_treatment)}


def run_analysis(db: Session, case: Case, user: User) -> ModelRun:
    imgs = wf.active_images(db, case)
    svc = get_service()
    try:
        res = svc.analyze([{"id": i.id, "path": i.path, "laterality": i.laterality,
                            **({} if i.path else {"bytes": image_bytes(i)})} for i in imgs],
                          {k: (v if v is not None else float("nan")) for k, v in patient_meta(case.patient).items()})
        status = "completed"
    except Exception as e:
        res, status = {"error": type(e).__name__, "overall": "Model unavailable - manual review"}, "failed"
    run = ModelRun(tenant_id=case.tenant_id, case_id=case.id, case_version=case.version, image_ids=[i.id for i in imgs],
                   model_version=res.get("model_version", svc.version), threshold_version=res.get("threshold_version", "-"),
                   source=res.get("source", svc.mode), status=status, result=res, latency_ms=res.get("latency_ms", 0))
    db.add(run)
    db.flush()
    case.status = "Unable to assess" if res.get("overall") == "Unable to assess" else "HCP review"
    wf.audit(db, case.tenant_id, case.id, user.id, "model_run",
             f"run #{run.id} [{run.source}] {res.get('overall')} ({run.model_version})", case.version)
    wf.add_task(db, case, "review", f"Review screening result for {case.patient.ref}", case.owner_id,
                due=wf.DEMO_DUE["review"])
    wf.notify(db, case.owner, f"Screening result ready to review ({case.patient.ref})", f"/cases/{case.id}")
    return run


@app.post("/cases/{case_id}/analyze")
def analyze(case_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    case = case_for(db, user, case_id)
    require(user, "operator", "referring")
    if not wf.active_images(db, case):
        wf.error(409, "no_images", "Upload at least one image before analysis.")
    if not case.patient.ref:
        wf.error(409, "missing_identity", "Resolve patient identity before analysis.")
    run_analysis(db, case, user)
    db.commit()
    return RedirectResponse(f"/cases/{case_id}?tab=retinal", 303)


def latest_run(db, case):
    return db.scalar(select(ModelRun).where(ModelRun.case_id == case.id).order_by(ModelRun.id.desc()))


# ------------------------------------------------------------------ case page
def relay_legs(db: Session, case: Case, ref: Referral | None = None):
    """Who has held / holds / will hold this case, for the relay strip."""
    if ref is None:
        ref = db.scalar(select(Referral).where(Referral.case_id == case.id, Referral.stage != "Cancelled")
                        .order_by(Referral.id.desc()))
    creator = db.get(User, case.created_by)
    has_imgs = bool(wf.active_images(db, case))
    signed = wf.current_review(db, case) is not None
    closed = bool(ref and ref.stage == "Closed")
    legs = []
    if creator.id != case.owner_id:
        legs.append({"name": creator.name, "label": "Captured the photos",
                     "state": "done" if has_imgs or signed or ref else "now"})
    ref_open = ref and ref.stage not in wf.TERMINAL
    owner_now = (not ref and has_imgs) or (ref_open and ref.owner_id == case.owner_id)
    legs.append({"name": case.owner.name, "label": "Referring clinician",
                 "state": "done" if closed or (signed and not owner_now) else "now" if owner_now else ""})
    if ref:
        legs.append({"name": ref.recipient.name, "label": ref.recipient.specialty or "Specialist",
                     "state": "done" if ref.stage in ("Response received", "Closed") else
                     "now" if ref_open and ref.owner_id == ref.recipient_id else ""})
        coord = db.scalar(select(User).where(User.tenant_id == case.tenant_id, User.role == "coordinator"))
        if coord:
            legs.append({"name": coord.name, "label": "Care coordinator",
                         "state": "now" if ref_open and ref.owner_id == coord.id else
                         "done" if ref.appointment.get("date") or closed else ""})
    else:
        legs.append({"name": "Specialist", "label": "Chosen when a consultation is sent", "state": ""})
    return legs


def patient_line(p: Patient):
    bits = []
    if p.age:
        bits.append(f"{int(p.age)} years")
    if p.sex:
        bits.append(p.sex)
    if p.dm_time is not None:
        bits.append(f"diabetes for {p.dm_time:g} years")
    if p.insulin == "yes":
        bits.append("on insulin")
    return (", ".join(bits).capitalize() + ". ") if bits else ""


@app.get("/cases/{case_id}", response_class=HTMLResponse)
def case_page(case_id: int, request: Request, tab: str = "retinal", msg: str = "", topic: str = "retinal",
              user: User = Depends(current_user), db: Session = Depends(get_db)):
    case = case_for(db, user, case_id)
    imgs = wf.active_images(db, case)
    superseded = db.scalars(select(Image).where(Image.case_id == case.id, Image.superseded_by.is_not(None))).all()
    run = latest_run(db, case)
    run_stale = bool(run and run.case_version != case.version)
    review = wf.current_review(db, case)
    reviews = db.scalars(select(Review).where(Review.case_id == case.id).order_by(Review.id.desc())).all()
    refs = db.scalars(select(Referral).where(Referral.case_id == case.id).order_by(Referral.id)).all()
    if user.role == "specialist":
        refs = [r for r in refs if r.recipient_id == user.id]
    specialists = db.scalars(select(User).where(User.tenant_id == user.tenant_id, User.role == "specialist",
                                                User.active.is_(True))).all()
    draft, token, idem = None, None, None
    if tab == "consult" and review and user.role == "referring":
        question = request.query_params.get("question") or ""
        recipient_id = int(request.query_params.get("recipient_id") or 0)
        draft = draft_package(case, case.patient, review, run if run and not run_stale else None, imgs, topic, question)
        token = approval_token(case, review, recipient_id, topic, question)
        idem = uuid.uuid4().hex
    if run and run.status == "completed":
        per = (run.result or {}).get("images", {})
        score = lambda i: (per.get(str(i.id)) or {}).get("p_dr", -1) if (per.get(str(i.id)) or {}).get("quality") != "unassessable" else -2
        imgs = sorted(imgs, key=lambda i: (i.laterality, -score(i)))
    events = db.scalars(select(AuditEvent).where(AuditEvent.case_id == case.id).order_by(AuditEvent.id)).all()
    actors = {u.id: u for u in db.scalars(select(User).where(User.tenant_id == user.tenant_id))}
    return templates.TemplateResponse(request, "case.html", ctx(
        request, user, db, case=case, imgs=imgs, superseded=superseded, run=run, run_stale=run_stale, review=review,
        reviews=reviews, refs=refs, specialists=specialists, tab=tab, msg=msg, draft=draft, token=token, idem=idem,
        topic=topic, events=events, actors=actors, conditions=SYSTEMIC_LABELS, legs=relay_legs(db, case),
        inputs={"age": case.patient.age, "sex": case.patient.sex, "dm_time": case.patient.dm_time,
                "insulin": case.patient.insulin, "oral_treatment": case.patient.oral_treatment,
                "conditions": {c: ((case.patient.conditions or {}).get(c) or {}).get("value", "unknown") for c in SYSTEMIC_LABELS},
                "medications": case.patient.medications or []},
        snapshot=oc.patient_snapshot(case.patient, run.result if run and run.status == "completed" and not run_stale else None,
                                     {**SYSTEMIC_LABELS, **COMPOSITE_LABELS}, get_service().metrics),
        patient_line=patient_line(case.patient),
        q_question=request.query_params.get("question", ""),
        q_recipient=int(request.query_params.get("recipient_id") or 0)))


# ------------------------------------------------------------------ review & sign (FR15-FR17, FR39-FR45)
@app.post("/cases/{case_id}/reviews")
async def sign_review(case_id: int, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    case = case_for(db, user, case_id)
    require(user, "referring")
    f = await request.form()
    sign_review_core(db, case, user, f)
    db.commit()
    return RedirectResponse(f"/cases/{case_id}?tab=review", 303)


def sign_review_core(db: Session, case: Case, user: User, f) -> Review:
    """Validate and sign the referring clinician's interpretation (bound to case version and model run)."""
    decision = f.get("decision")
    if decision not in ("accept", "disagree", "recapture", "manual_review"):
        wf.error(422, "bad_decision", "Choose a review decision.")
    if decision == "disagree" and not f.get("override_reason", "").strip():
        wf.error(422, "reason_required", "Disagreeing with the model requires a reason.")
    if not f.get("attest"):
        wf.error(422, "attestation_required", "Confirm the attestation to sign.")
    run = latest_run(db, case)
    run_id = run.id if run and run.case_version == case.version else None
    imgs = wf.active_images(db, case)
    ts = now()
    interp = f.get("interpretation", "").strip() or "(no free-text interpretation)"
    sysnotes = {c: f.get(f"sys_{c}") for c in SYSTEMIC_LABELS if f.get(f"sys_{c}")}
    sig = wf.review_signature(case, imgs, run_id, interp + json.dumps(sysnotes, sort_keys=True), user, ts)
    rv = Review(tenant_id=case.tenant_id, case_id=case.id, case_version=case.version, model_run_id=run_id, hcp_id=user.id,
                decision=decision, interpretation=interp, override_reason=f.get("override_reason", ""),
                next_action=f.get("next_action", "none"), systemic_notes=sysnotes, signature=sig, signed_at=ts)
    db.add(rv)
    case.status = "Signed"
    wf.close_tasks(db, case.id, "review")
    wf.audit(db, case.tenant_id, case.id, user.id, "review_signed",
             f"{decision}; next action: {rv.next_action}; sig {sig[:12]}", case.version)
    if decision == "recapture":
        wf.add_task(db, case, "recapture", f"Recapture images for {case.patient.ref}", case.created_by)
    for c, note in sysnotes.items():
        if note == "request_context":
            wf.add_task(db, case, f"context_{c}", f"Clarify {SYSTEMIC_LABELS[c]} history for {case.patient.ref}",
                        case.owner_id)
    db.flush()
    return rv


# ------------------------------------------------------------------ consultation & dispatch (FR21-FR25, FR43)
def approval_token(case, review, recipient_id, topic, question):
    return wf.digest({"case": case.id, "v": case.version, "review": review.id if review else None,
                      "recipient": recipient_id, "topic": topic, "q": question})


@app.post("/referrals")
async def create_referral(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    require(user, "referring")
    f = await request.form()
    idem = f.get("idempotency_key") or request.headers.get("Idempotency-Key")
    if not idem:
        wf.error(422, "missing_idempotency_key", "Idempotency key required.")
    prior = db.scalar(select(Referral).where(Referral.tenant_id == user.tenant_id, Referral.sender_id == user.id,
                                             Referral.idempotency_key == idem))
    if prior:                                                   # retry / double click -> same referral
        return RedirectResponse(f"/referrals/{prior.id}?msg=Already+sent", 303)
    case = case_for(db, user, int(f["case_id"]))
    question, topic = f.get("question", "").strip(), f.get("topic", "retinal")
    recipient = db.get(User, int(f.get("recipient_id") or 0))
    if not question:
        wf.error(422, "missing_question", "A focused consultation question is required.")
    if not recipient or recipient.tenant_id != user.tenant_id or recipient.role != "specialist" or not recipient.active:
        wf.error(422, "recipient_not_granted", "Recipient is not an active specialist in the approved directory.")
    review = wf.current_review(db, case)
    if not review:
        wf.error(409, "stale_signature", "No current signed review: the case changed after signing. Re-review first.")
    if f.get("approval_token") != approval_token(case, review, recipient.id, topic, question):
        wf.error(409, "approval_invalidated", "Package, recipient or case changed since preview. Preview again before signing.")
    if not f.get("attest"):
        wf.error(422, "attestation_required", "Confirm the attestation to sign and send.")
    ref = dispatch_referral(db, user, case, review, recipient, topic, question, f, idem)
    if ref is None:
        prior = db.scalar(select(Referral).where(Referral.sender_id == user.id, Referral.idempotency_key == idem))
        return RedirectResponse(f"/referrals/{prior.id}", 303)
    db.commit()
    return RedirectResponse(f"/referrals/{ref.id}", 303)


def dispatch_referral(db: Session, user: User, case: Case, review: Review, recipient: User, topic: str, question: str,
                      f, idem: str):
    """Build the traced package from case data and send it. Returns None if the idempotency key was already used."""
    run = latest_run(db, case)
    imgs = wf.active_images(db, case)
    package = draft_package(case, case.patient, review, run if run and run.case_version == case.version else None,
                            imgs, topic, question)
    package["sender_note"] = f.get("sender_note", "")
    package["image_ids"] = [i.id for i in imgs]
    package["unavailable"] = f.get("unavailable", "")
    phash = wf.digest(package)
    ref = Referral(tenant_id=user.tenant_id, case_id=case.id, review_id=review.id, sender_id=user.id,
                   recipient_id=recipient.id, owner_id=recipient.id, topic=topic, question=question, package=package,
                   package_hash=phash, signature=wf.digest({"pkg": phash, "signer": user.id, "at": now().isoformat()}),
                   idempotency_key=idem, stage="Sent", due_at=now() + wf.DEMO_DUE["acknowledge"])
    db.add(ref)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        return None
    wf.audit(db, case.tenant_id, case.id, user.id, "referral_sent",
             f"#{ref.id} to {recipient.name} ({topic}); package {phash[:12]}", case.version)
    wf.add_task(db, case, "acknowledge", f"Acknowledge consultation RL-{ref.id}", recipient.id, ref.id, wf.DEMO_DUE["acknowledge"])
    wf.notify(db, recipient, f"New consultation request RL-{ref.id}", f"/referrals/{ref.id}")
    return ref


# ------------------------------------------------------------------ guided consult: sign + send in one step
def consult_token(case: Case) -> str:
    return wf.digest({"consult": case.id, "v": case.version})


@app.get("/cases/{case_id}/consult", response_class=HTMLResponse)
def consult_page(case_id: int, request: Request, topic: str = "", user: User = Depends(current_user),
                 db: Session = Depends(get_db)):
    case = case_for(db, user, case_id)
    require(user, "referring")
    run = latest_run(db, case)
    res = run.result if run and run.status == "completed" and run.case_version == case.version else None
    snapshot = oc.patient_snapshot(case.patient, res, {**SYSTEMIC_LABELS, **COMPOSITE_LABELS}, get_service().metrics)
    topics = consult.suggested_topics(res, snapshot)
    topic = topic if topic in consult.TOPICS else topics[0][0]
    specialists = db.scalars(select(User).where(User.tenant_id == user.tenant_id, User.role == "specialist",
                                                User.active.is_(True))).all()
    review = wf.current_review(db, case)
    imgs = wf.active_images(db, case)
    preview = draft_package(case, case.patient, review or Review(id=0, interpretation="(your interpretation, signed on send)",
                                                                override_reason=""),
                            run if res else None, imgs, topic, consult.TOPICS[topic]["question"])
    sent = db.scalars(select(Referral).where(Referral.case_id == case.id).order_by(Referral.id.desc())).all()
    return templates.TemplateResponse(request, "consult.html", ctx(
        request, user, db, active="patients", case=case, res=res, run=run, review=review, topics=topics, topic=topic,
        topic_info=consult.TOPICS[topic], specialists=consult.specialists_for(topic, specialists),
        draft=consult.draft_interpretation(res), preview=preview, token=consult_token(case), idem=uuid.uuid4().hex,
        sent=sent, TOPICS=consult.TOPICS))


@app.post("/cases/{case_id}/consult")
async def consult_send(case_id: int, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    case = case_for(db, user, case_id)
    require(user, "referring")
    f = await request.form()
    idem = f.get("idempotency_key", "")
    prior = db.scalar(select(Referral).where(Referral.tenant_id == user.tenant_id, Referral.sender_id == user.id,
                                             Referral.idempotency_key == idem)) if idem else None
    if prior:                                              # double click / retry -> same consultation
        return RedirectResponse(f"/referrals/{prior.id}?msg=Already+sent", 303)
    if f.get("token") != consult_token(case):
        wf.error(409, "case_changed", "The case changed while you were writing. Reopen Consult a specialist and check again.")
    topic = f.get("topic") if f.get("topic") in consult.TOPICS else "retinal"
    question = (f.get("question") or "").strip()
    if not question:
        wf.error(422, "missing_question", "Write the question you want the specialist to answer.")
    recipient = db.get(User, int(f.get("recipient_id") or 0))
    if not recipient or recipient.tenant_id != user.tenant_id or recipient.role != "specialist" or not recipient.active:
        wf.error(422, "recipient_not_granted", "Choose a specialist from the directory.")
    if not f.get("attest"):
        wf.error(422, "attestation_required", "Tick the box to sign your interpretation and the package.")
    review = wf.current_review(db, case) or sign_review_core(db, case, user, f)
    ref = dispatch_referral(db, user, case, review, recipient, topic, question, f, idem or uuid.uuid4().hex)
    if ref is None:
        wf.error(409, "duplicate", "This consultation was already sent.")
    db.commit()
    return RedirectResponse(f"/referrals/{ref.id}?msg=Sent+to+{recipient.name.replace(' ', '+')}.+You+will+be+notified+when+they+accept.", 303)


@app.get("/referrals/{rid}", response_class=HTMLResponse)
def referral_page(rid: int, request: Request, msg: str = "", user: User = Depends(current_user),
                  db: Session = Depends(get_db)):
    ref = referral_for(db, user, rid)
    if user.id == ref.recipient_id and not ref.opened_at:
        ref.opened_at = now()
        wf.audit(db, ref.tenant_id, ref.case_id, user.id, "referral_opened", f"#{ref.id}")
        db.commit()
    msgs = db.scalars(select(Message).where(Message.referral_id == ref.id).order_by(Message.id)).all()
    barriers = db.scalars(select(Barrier).where(Barrier.referral_id == ref.id)).all()
    tasks = db.scalars(select(Task).where(Task.referral_id == ref.id).order_by(Task.id)).all()
    coords = db.scalars(select(User).where(User.tenant_id == user.tenant_id, User.role == "coordinator")).all()
    specialists = db.scalars(select(User).where(User.tenant_id == user.tenant_id, User.role == "specialist",
                                                User.active.is_(True), User.id != ref.recipient_id)).all()
    imgs = db.scalars(select(Image).where(Image.id.in_(ref.package.get("image_ids", [])))).all()
    return templates.TemplateResponse(request, "referral.html", ctx(
        request, user, db, ref=ref, msgs=msgs, barriers=barriers, tasks=tasks, coords=coords, specialists=specialists,
        imgs=imgs, msg=msg, allowed=sorted(wf.REFERRAL_TRANSITIONS.get(ref.stage, set())),
        legs=relay_legs(db, ref.case, ref)))


@app.post("/referrals/{rid}/action")
async def referral_action(rid: int, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    ref = referral_for(db, user, rid)
    case = ref.case
    f = await request.form()
    a = f.get("action")
    body = f.get("body", "").strip()
    is_rec, is_send, is_coord = user.id == ref.recipient_id, user.id == ref.sender_id, user.role == "coordinator"
    coord = db.scalar(select(User).where(User.tenant_id == user.tenant_id, User.role == "coordinator"))

    def msg(kind, sign=False, rec=""):
        m = Message(tenant_id=ref.tenant_id, referral_id=ref.id, author_id=user.id, kind=kind, body=body, recommendation=rec)
        if sign:
            m.signature = wf.digest({"ref": ref.id, "body": body, "rec": rec, "author": user.id, "at": now().isoformat()})
        db.add(m)

    if a == "acknowledge" and is_rec:
        wf.transition(db, ref, "Acknowledged", user)
        wf.close_tasks(db, case.id, "acknowledge", ref.id)
        ref.owner_id = coord.id if coord else ref.recipient_id
        if coord:
            wf.add_task(db, case, "schedule", f"Schedule appointment for RL-{ref.id}", coord.id, ref.id, wf.DEMO_DUE["schedule"])
            wf.notify(db, coord, f"Consultation RL-{ref.id} accepted - scheduling needed", f"/referrals/{ref.id}")
        wf.notify(db, ref.sender, f"Consultation RL-{ref.id} acknowledged", f"/referrals/{ref.id}")
    elif a == "decline" and is_rec:
        if not body:
            wf.error(422, "reason_required", "Declining requires a reason.")
        msg("decline")
        wf.transition(db, ref, "Declined", user, body)
        ref.owner_id = ref.sender_id
        wf.add_task(db, case, "reassign", f"RL-{ref.id} declined - choose another specialist", ref.sender_id, ref.id)
        wf.notify(db, ref.sender, f"Consultation RL-{ref.id} declined - action needed", f"/referrals/{ref.id}")
    elif a == "request_info" and is_rec:
        if not body:
            wf.error(422, "question_required", "Say what information is needed.")
        msg("info_request")
        wf.transition(db, ref, "Needs information", user)
        wf.close_tasks(db, case.id, "acknowledge", ref.id)
        wf.add_task(db, case, "provide_info", f"Provide information requested on RL-{ref.id}", ref.sender_id, ref.id,
                    wf.DEMO_DUE["review"])
        wf.notify(db, ref.sender, f"Information requested on RL-{ref.id}", f"/referrals/{ref.id}")
    elif a == "provide_info" and is_send and ref.stage == "Needs information":
        msg("info_reply")
        wf.close_tasks(db, case.id, "provide_info", ref.id)
        wf.add_task(db, case, "acknowledge", f"Review new information on RL-{ref.id}", ref.recipient_id, ref.id,
                    wf.DEMO_DUE["acknowledge"])
        wf.audit(db, ref.tenant_id, case.id, user.id, "info_provided", f"#{ref.id}")
        wf.notify(db, ref.recipient, f"Information provided on RL-{ref.id}", f"/referrals/{ref.id}")
    elif a == "schedule" and (is_coord or is_rec):
        appt = {"date": f.get("date"), "time": f.get("time"), "timezone": f.get("tz", "America/New_York"),
                "facility": f.get("facility", ""), "confirmation": f.get("confirmation", ""),
                "history": (ref.appointment or {}).get("history", [])}
        if not appt["date"]:
            wf.error(422, "date_required", "Appointment date required.")
        if ref.appointment.get("date"):
            appt["history"] = appt["history"] + [{k: ref.appointment.get(k) for k in ("date", "time", "facility")}
                                                 | {"changed_at": now().isoformat(), "reason": body or "rescheduled"}]
        ref.appointment = appt
        wf.transition(db, ref, "Scheduled", user, f"{appt['date']} {appt['time']} {appt['timezone']} @ {appt['facility']}")
        wf.close_tasks(db, case.id, "schedule", ref.id)
        ref.owner_id = ref.recipient_id
        wf.add_task(db, case, "respond", f"Record visit and sign response for RL-{ref.id}", ref.recipient_id, ref.id,
                    wf.DEMO_DUE["respond"])
    elif a == "missed" and (is_coord or is_rec):
        wf.transition(db, ref, "Acknowledged", user, f"appointment missed/cancelled: {body}")
        wf.close_tasks(db, case.id, "respond", ref.id)
        hist = (ref.appointment or {}).get("history", []) + [{**{k: ref.appointment.get(k) for k in ("date", "time", "facility")},
                                                              "changed_at": now().isoformat(), "reason": body or "missed"}]
        ref.appointment = {"history": hist}
        if coord:
            ref.owner_id = coord.id
            wf.add_task(db, case, "schedule", f"Re-schedule RL-{ref.id} (missed/cancelled)", coord.id, ref.id, wf.DEMO_DUE["schedule"])
    elif a == "visit" and (is_coord or is_rec):
        wf.transition(db, ref, "Visit recorded", user)
    elif a == "respond" and is_rec:
        if not body or not f.get("recommendation"):
            wf.error(422, "response_incomplete", "A signed response needs text and a next-action recommendation.")
        if not f.get("attest"):
            wf.error(422, "attestation_required", "Confirm the attestation to sign.")
        msg("response", sign=True, rec=f.get("recommendation"))
        wf.transition(db, ref, "Response received", user)
        wf.close_tasks(db, case.id, "respond", ref.id)
        wf.close_tasks(db, case.id, "acknowledge", ref.id)
        ref.owner_id = ref.sender_id
        wf.add_task(db, case, "close", f"Acknowledge specialist response on RL-{ref.id}", ref.sender_id, ref.id,
                    wf.DEMO_DUE["review"])
        wf.notify(db, ref.sender, f"Specialist response received on RL-{ref.id}", f"/referrals/{ref.id}")
    elif a == "close" and is_send:
        wf.transition(db, ref, "Closed", user, "referrer acknowledged signed response")
        wf.close_tasks(db, case.id, "close", ref.id)
    elif a in ("unreachable", "patient_declined") and (is_coord or is_send):
        wf.transition(db, ref, "Unreachable" if a == "unreachable" else "Patient declined", user, body)
    elif a == "cancel" and is_send:
        wf.transition(db, ref, "Cancelled", user, body)
    elif a == "reassign_owner" and (is_coord or is_send):
        new = db.get(User, int(f.get("owner_id") or 0))
        if not new or new.tenant_id != user.tenant_id or not new.active or new.role not in ("coordinator", "specialist", "referring"):
            wf.error(422, "ineligible_assignee", "Choose an active eligible assignee.")
        ref.owner_id = new.id
        wf.audit(db, ref.tenant_id, case.id, user.id, "owner_reassigned", f"#{ref.id} -> {new.name}")
    else:
        wf.error(403, "action_not_permitted", f"'{a}' is not available to you at stage {ref.stage}.")
    db.commit()
    return RedirectResponse(f"/referrals/{rid}", 303)


@app.post("/referrals/{rid}/barriers")
def add_barrier(rid: int, category: str = Form(...), note: str = Form(""), user: User = Depends(current_user),
                db: Session = Depends(get_db)):
    ref = referral_for(db, user, rid)
    require(user, "coordinator", "referring")
    if category not in ("transport", "language", "affordability", "contact", "preference", "other"):
        wf.error(422, "bad_category", "Unknown barrier category.")
    db.add(Barrier(tenant_id=ref.tenant_id, referral_id=ref.id, category=category, note=note, owner_id=user.id))
    wf.audit(db, ref.tenant_id, ref.case_id, user.id, "barrier_added", f"#{ref.id} {category} (does not change medical priority)")
    db.commit()
    return RedirectResponse(f"/referrals/{rid}", 303)


@app.post("/barriers/{bid}/resolve")
def resolve_barrier(bid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    b = db.get(Barrier, bid)
    if not b:
        wf.error(404, "not_found", "Barrier not found.")
    ref = referral_for(db, user, b.referral_id)
    b.resolved = True
    wf.audit(db, ref.tenant_id, ref.case_id, user.id, "barrier_resolved", f"{b.category}")
    db.commit()
    return RedirectResponse(f"/referrals/{ref.id}", 303)


@app.post("/tasks/{tid}/done")
def task_done(tid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    t = db.get(Task, tid)
    if not t or t.tenant_id != user.tenant_id or t.assignee_id != user.id:
        wf.error(404, "not_found", "Task not found.")
    t.status = "done"
    wf.audit(db, t.tenant_id, t.case_id, user.id, "task_done", t.title)
    db.commit()
    return RedirectResponse("/inbox", 303)


# ------------------------------------------------------------------ try an image (no case, nothing stored)
@app.get("/try")
def try_redirect():
    return RedirectResponse("/screen", 303)


@app.get("/screen", response_class=HTMLResponse)
def screen_page(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return templates.TemplateResponse(request, "screen.html", ctx(request, user, db, active="screen", results=None,
                                                                    token="", details=trylab.parse_details({})))


@app.post("/screen", response_class=HTMLResponse)
async def screen_run(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Analyse newly uploaded photos (by eye), or re-run held ones (token) with changed patient details."""
    import base64
    form = await request.form()
    uploads = [(eye, f) for eye, key in (("OD", "files_OD"), ("OS", "files_OS"), ("unknown", "files"))
               for f in form.getlist(key) if hasattr(f, "read") and getattr(f, "filename", "")]
    if len(uploads) > MAX_IMAGES:
        wf.error(413, "too_many_images", f"Up to {MAX_IMAGES} photos per screening.")
    if uploads:
        blobs = []
        for i, (eye, uf) in enumerate(uploads):
            data = await uf.read()
            _validate_upload(data)
            blobs.append((i + 1, uf.filename or f"photo {i + 1}", data, eye))
        token = trylab.store(db, user, blobs)
    else:
        token = form.get("token", "")
        if not token:
            wf.error(422, "no_photos", "Add at least one retinal photo to screen.")
        blobs = trylab.load(db, user, token)
        if not blobs:
            wf.error(410, "photos_expired", "Those photos are no longer held (kept for 1 hour). Upload them again.")
    details = trylab.parse_details(form)
    svc = get_service()
    res = svc.analyze([{"id": i, "laterality": eye, "bytes": d} for i, _, d, eye in blobs], trylab.model_inputs(details))
    uri = lambda b, mt: f"data:{mt};base64," + base64.b64encode(b).decode()
    results = []
    for i, name, data, eye in blobs:
        r = dict(res["images"].get(str(i)) or res["images"].get(i) or {})
        q = r.get("quality", "unsupported")
        attn = None
        if res.get("source") == "live" and q in ("assessable", "uncertain", "unassessable"):
            png = svc.explain(data, "quality_poor" if q == "unassessable" else "dr_referable")
            if png:
                attn = uri(thumbnail(png, 900), "image/jpeg")
        verdict = ("Unable to assess" if q in ("unassessable", "unsupported") else
                   "Referable DR signal" if r.get("dr_positive") else
                   "Uncertain quality" if q == "uncertain" else "No model finding")
        results.append({**r, "name": name, "eye": eye, "quality": q, "src": uri(thumbnail(data, 900), "image/jpeg"),
                        "attn": attn, "verdict": verdict})
    overall = res.get("overall")
    if all(eye == "unknown" for *_, eye in blobs):      # no eye labels: summarise from the photos themselves
        per = [r for r in results if r.get("quality") in ("assessable", "uncertain")]
        overall = ("Referable DR signal" if any(r.get("dr_positive") for r in per) else
                   "No model finding" if per else "Unable to assess")
    labels = {**SYSTEMIC_LABELS, **COMPOSITE_LABELS}
    snapshot = oc.patient_snapshot(trylab.as_patient(details), {**res, "overall": overall}, labels, svc.metrics)
    evidence = trylab.evidence_for(overall, any(r.get("edema_flag") for r in results), snapshot)
    fnd = tx.findings({**res, "overall": overall}, snapshot)
    meds = details["medications"]
    wf.audit(db, user.tenant_id, None, user.id, "screen", f"{len(blobs)} photo(s), held 1 hour")
    db.commit()
    eyes_known = any(eye != "unknown" for *_, eye in blobs)
    return templates.TemplateResponse(request, "screen.html", ctx(
        request, user, db, active="screen", results=results, res=res, overall=overall, eyes_known=eyes_known,
        thr=res["thresholds"], version=res.get("model_version"), source=res.get("source"),
        latency=res.get("latency_ms", 0), token=token, details=details, systemic=res.get("systemic", {}),
        snapshot=snapshot, evidence=evidence, patient_ref=form.get("patient_ref", ""),
        options=tx.options_for(fnd, meds), sponsored=tx.sponsored_for(fnd),
        current_alerts=tx.merge_alerts(tx.check_interactions(meds, (), [f["key"] for f in fnd])),
        n_inputs=sum(v is not None and v != "unknown" for k, v in details.items() if k not in ("conditions", "medications"))))


@app.post("/screen/save")
async def screen_save(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Turn a screening into a case (patient, photos by eye, analysis) and continue to review and consultation."""
    require(user, "operator", "referring")
    form = await request.form()
    blobs = trylab.load(db, user, form.get("token", ""))
    if not blobs:
        wf.error(410, "photos_expired", "Those photos are no longer held (kept for 1 hour). Screen them again.")
    d = trylab.parse_details(form)
    owner = user if user.role == "referring" else db.scalar(
        select(User).where(User.tenant_id == user.tenant_id, User.role == "referring", User.active.is_(True)))
    ref = (form.get("patient_ref") or "").strip()[:40] or f"RL-P{secrets.randbelow(9000) + 1000}"
    conds = {c: {"value": v, "source": f"entered by {user.name}" if v != "unknown" else "not recorded",
                 "date": now().date().isoformat() if v != "unknown" else "", "verification": "reported"}
             for c, v in d["conditions"].items()}
    patient = Patient(tenant_id=user.tenant_id, ref=ref, age=d["age"], sex=d["sex"], dm_time=d["dm_time"],
                      insulin=d["insulin"], oral_treatment=d["oral_treatment"], conditions=conds,
                      medications=d["medications"])
    db.add(patient)
    db.flush()
    case = Case(tenant_id=user.tenant_id, patient_id=patient.id, owner_id=owner.id, created_by=user.id,
                encounter_date=datetime.now(TZ).date().isoformat(), device="Portable smartphone fundus camera",
                symptoms=form.get("symptoms", ""))
    db.add(case)
    db.flush()
    wf.audit(db, user.tenant_id, case.id, user.id, "case_created", f"patient {ref} (from Screen)", 1)
    for i, name, data, eye in blobs:
        img = _validate_upload(data)
        sha, path = _store(case, data, img)
        db.add(Image(tenant_id=case.tenant_id, case_id=case.id, sha256=sha, path=path, laterality=eye,
                     view="macula-centred", source=case.device, width=img.size[0], height=img.size[1],
                     uploaded_by=user.id, data=None if path else data))
    case.status = "Images ready"
    wf.bump_version(db, case, user, f"{len(blobs)} image(s) from Screen")
    db.flush()
    run_analysis(db, case, user)
    db.commit()
    if user.role == "referring":
        return RedirectResponse(f"/cases/{case.id}/consult?msg=Saved+as+{ref}.+Now+choose+a+specialist+and+send.", 303)
    return RedirectResponse(f"/cases/{case.id}?msg=Saved+as+{ref}.+{owner.name.replace(' ', '+')}+has+been+asked+to+review.", 303)


# ------------------------------------------------------------------ patients (case list + whole-body panel)
@app.get("/patients", response_class=HTMLResponse)
def patients_page(request: Request, q: str = "", user: User = Depends(current_user), db: Session = Depends(get_db)):
    svc = get_service()
    labels = {**SYSTEMIC_LABELS, **COMPOSITE_LABELS}
    cases = visible_cases(db, user)
    if q:
        cases = [c for c in cases if q.lower() in c.patient.ref.lower()]
    latest = {}
    if cases:
        for r in db.scalars(select(ModelRun).where(ModelRun.case_id.in_([c.id for c in cases]), ModelRun.status == "completed")
                            .order_by(ModelRun.id)):
            latest[r.case_id] = r.result or {}
    open_refs = Counter(r.case_id for r in db.scalars(select(Referral).where(
        Referral.tenant_id == user.tenant_id, Referral.stage.not_in(list(wf.TERMINAL)))))
    rows = [{"case": c, "res": latest.get(c.id), "refs": open_refs.get(c.id, 0),
             "snap": oc.patient_snapshot(c.patient, latest.get(c.id), labels, svc.metrics)} for c in cases]
    panel = oc.panel_stats(db, cases, labels, svc.metrics)
    return templates.TemplateResponse(request, "patients.html", ctx(request, user, db, active="patients", rows=rows,
                                                                      p=panel, q=q))


# ------------------------------------------------------------------ oculomics (whole-body view)
def visible_cases(db: Session, user: User):
    q = select(Case).where(Case.tenant_id == user.tenant_id)
    if user.role == "referring":
        q = q.where(Case.owner_id == user.id)
    elif user.role == "specialist":
        q = q.where(Case.id.in_(select(Referral.case_id).where(Referral.recipient_id == user.id)))
    elif user.role == "admin":
        return []                       # administrators do not get clinical access
    return db.scalars(q.order_by(Case.id.desc())).all()


@app.get("/oculomics", response_class=HTMLResponse)
def oculomics_page(request: Request, view: str = "panel", user: User = Depends(current_user),
                   db: Session = Depends(get_db)):
    svc = get_service()
    labels = {**SYSTEMIC_LABELS, **COMPOSITE_LABELS}
    view = "science" if view == "science" else "panel"
    panel = oc.panel_stats(db, visible_cases(db, user), labels, svc.metrics) if view == "panel" else None
    return templates.TemplateResponse(request, "oculomics.html", ctx(
        request, user, db, active="oculomics", view=view, p=panel, tiers=oc.evidence_map(svc.metrics)))


# ------------------------------------------------------------------ evidence (FR23 / FR28)
@app.get("/evidence", response_class=HTMLResponse)
def evidence_page(request: Request, q: str = "", user: User = Depends(current_user), db: Session = Depends(get_db)):
    brief = evidence_brief(q) if q else None
    return templates.TemplateResponse(request, "evidence.html", ctx(request, user, db, q=q, brief=brief, active="evidence"))


# ------------------------------------------------------------------ notifications / preferences (FR27)
@app.get("/notifications", response_class=HTMLResponse)
def notifications(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    ns = db.scalars(select(Notification).where(Notification.user_id == user.id).order_by(Notification.id.desc())).all()
    page = templates.TemplateResponse(request, "notifications.html", ctx(request, user, db, ns=ns, active=""))
    for n in ns:
        n.read = True
    db.commit()
    return page


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return templates.TemplateResponse(request, "settings.html", ctx(request, user, db, active="settings"))


@app.post("/settings")
def save_settings(mode: str = Form("immediate"), quiet_start: str = Form(""), quiet_end: str = Form(""),
                  user: User = Depends(current_user), db: Session = Depends(get_db)):
    user.prefs = {"mode": mode if mode in ("immediate", "digest") else "immediate",
                  "quiet_start": int(quiet_start) if quiet_start else None,
                  "quiet_end": int(quiet_end) if quiet_end else None}
    db.commit()
    return RedirectResponse("/settings", 303)


# ------------------------------------------------------------------ analytics (FR38, FR46)
@app.get("/analytics")
def analytics_redirect(request: Request):
    return RedirectResponse("/performance" + (f"?{request.url.query}" if request.url.query else ""), 303)


@app.get("/performance", response_class=HTMLResponse)
def analytics(request: Request, view: str = "model", user: User = Depends(current_user), db: Session = Depends(get_db)):
    refs = db.scalars(select(Referral).where(Referral.tenant_id == user.tenant_id)).all()
    reached = Counter()
    order = {s: i for i, s in enumerate(wf.FUNNEL)}
    for r in refs:
        events = db.scalars(select(AuditEvent).where(AuditEvent.case_id == r.case_id, AuditEvent.action == "referral_stage",
                                                     AuditEvent.detail.like(f"#{r.id} %"))).all()
        stages = {"Sent"} | {e.detail.split("-> ")[-1].split(".")[0].strip() for e in events}
        top = max((order[s] for s in stages if s in order), default=0)
        for s in wf.FUNNEL[:top + 1]:
            reached[s] += 1
    lat = [(r.acknowledged_at - r.sent_at).total_seconds() / 3600 for r in refs if r.acknowledged_at]
    closed = sum(r.stage == "Closed" for r in refs)
    alt = Counter(r.stage for r in refs if r.stage in wf.ALTERNATIVE_DISPOSITIONS)
    open_tasks = db.scalars(select(Task).where(Task.tenant_id == user.tenant_id, Task.status == "open")).all()
    runs = db.scalars(select(ModelRun).where(ModelRun.tenant_id == user.tenant_id)).all()
    svc = get_service()
    return templates.TemplateResponse(request, "analytics.html", ctx(
        request, user, db, view=view, active="performance", funnel=[(s, reached[s]) for s in wf.FUNNEL], n_sent=len(refs),
        ack_median=(sorted(lat)[len(lat) // 2] if lat else None), n_ack=len(lat), closed=closed, alt=alt,
        pending=len(refs) - closed - sum(alt.values()), overdue=sum(wf.is_overdue(t.due_at) for t in open_tasks),
        n_open=len(open_tasks), runs=runs, metrics=svc.metrics, calib=svc.calib, model_version=svc.version))


# ------------------------------------------------------------------ JSON API (subset of spec §13)
@app.post("/api/vision/register")
async def register_vision(request: Request, db: Session = Depends(get_db)):
    """Called by the vision worker (scripts/run_vision_tunnel.sh) to announce its current tunnel URL."""
    key = os.environ.get("INSIGHTRX_VISION_KEY", "")
    if not key or not secrets.compare_digest(request.headers.get("X-InsightRx-Key", ""), key):
        wf.error(401, "bad_key", "Not authorised.")
    url = (await request.json()).get("url", "")
    if not url.startswith("https://"):
        wf.error(422, "bad_url", "HTTPS URL required.")
    s = db.get(AppSetting, "vision_url") or AppSetting(key="vision_url", value="")
    s.value = url
    db.merge(s)
    db.commit()
    svc = get_service()
    if hasattr(svc, "reset"):
        svc.reset()
    return {"registered": url, "vision": svc.mode}


@app.get("/api/health")
def health():
    svc = get_service()
    return {"status": "ok", "vision": svc.mode, "model_version": svc.version}


@app.get("/api/cases/{case_id}/timeline")
def api_timeline(case_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    case = case_for(db, user, case_id)
    ev = db.scalars(select(AuditEvent).where(AuditEvent.case_id == case.id).order_by(AuditEvent.id)).all()
    return [{"at": e.created_at.isoformat(), "actor": e.actor_id, "action": e.action, "detail": e.detail,
             "version": e.version} for e in ev]


@app.get("/api/cases/{case_id}/result")
def api_result(case_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    case = case_for(db, user, case_id)
    run = latest_run(db, case)
    return {"case_version": case.version, "run": run.result if run else None,
            "stale": bool(run and run.case_version != case.version)}


# ------------------------------------------------------------------ therapeutics layer (routes_therapy.py)
from .routes_therapy import router as therapy_router  # noqa: E402
from .seed import backfill_medications, ensure_desk  # noqa: E402

app.include_router(therapy_router)
with SessionLocal() as _db:                     # additive demo upgrades for workspaces seeded before this layer
    if _db.query(Tenant).count():
        backfill_medications(_db)
        ensure_desk(_db)
