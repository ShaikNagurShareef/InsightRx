"""Insight Rx Copilot routes (Backboard.io memory + retrieval): ask about a case, ask in general, and see or edit what
the copilot remembers about you."""
import logging
import re
from urllib.parse import quote

from markupsafe import Markup, escape

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from . import copilot
from . import personalize
from .db import get_db
from .llm import RestrictedPayload
from .main import case_for, ctx, current_user, templates
from .models import User

router = APIRouter()
log = logging.getLogger("insightrx.copilot")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CITE = re.compile(r"\[(\d{1,2}(?:\s*[,–-]\s*\d{1,2})*)\]")
_ITALIC = re.compile(r"(?<![*\w])\*(?!\s)([^*]+?)(?<!\s)\*(?![*\w])")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def cp_md(text):
    """Minimal, safe rendering of an answer: escape everything, then bold, bullet lists and [n] citations."""
    out, items = [], []

    def flush():
        if items:
            out.append("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
            items.clear()
    for line in str(text or "").splitlines():
        html = _BOLD.sub(r"<strong>\1</strong>", str(escape(line.strip())))
        html = _ITALIC.sub(r"<em>\1</em>", html)
        html = _CITE.sub(lambda m: "".join(f'<sup class="cp-cite">{n}</sup>' for n in re.findall(r"\d+", m.group(1))), html)
        if _BULLET.match(line):
            items.append(_BULLET.sub("", html, count=1))
        elif html:
            flush()
            out.append(f"<p>{html.lstrip('#').strip()}</p>")
        else:
            flush()
    flush()
    return Markup("".join(out))


def question_of(form):
    """A suggestion button and the text box both submit 'question'; use the first one that is filled in."""
    return next((q.strip() for q in form.getlist("question") if q and q.strip()), "")


def engine_label():
    from .llm import GEMINI_MODEL, gemini_key
    return (f"{GEMINI_MODEL} · memory by Backboard" if gemini_key()
            else f"{copilot.PROVIDER} · {copilot.MODEL} via Backboard")


def allow(user, scope="general"):
    """403 for roles outside the copilot's RBAC; 404 when the copilot is not configured on this deployment."""
    if not copilot.enabled():
        raise HTTPException(404, "Not found")
    if not copilot.can_use(user, scope):
        raise HTTPException(403, "The copilot is not available to your role.")


templates.env.filters["cp_md"] = cp_md
templates.env.globals["can_copilot"] = copilot.can_use
templates.env.globals["copilot_engine"] = engine_label


def _fail(url, e):
    if isinstance(e, RestrictedPayload):
        msg = "That question contains identifying details (a reference, date or file name). Rephrase without them."
    elif isinstance(e, ValueError):
        msg = "Type a question first."
    elif isinstance(e, copilot.CopilotUnavailable):
        msg = str(e)
    else:
        log.exception("copilot call failed")
        msg = "The copilot could not answer just now. Try again in a moment."
    return RedirectResponse(f"{url}{'&' if '?' in url else '?'}msg={quote(msg)}", 303)


@router.post("/cases/{case_id}/copilot")
async def ask_about_case(case_id: int, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    allow(user, "case")
    case = case_for(db, user, case_id)
    f = await request.form()
    back = f"/cases/{case.id}/therapy"
    from .routes_therapy import _fresh_result, _snapshot
    res = _fresh_result(db, case)
    b = personalize.bundle(res, _snapshot(case.patient, res), case.patient.medications or [], with_trials=False)
    try:
        brief = copilot.case_brief(b, res, case.patient.age, case.patient.sex)
        from starlette.concurrency import run_in_threadpool
        await run_in_threadpool(copilot.ask, db, user, question_of(f), f"case:{case.id}", brief, case.id)
    except Exception as e:                       # noqa: BLE001 - show a safe message, keep the page usable
        return _fail(back, e)
    return RedirectResponse(back + "#copilot", 303)


@router.get("/copilot", response_class=HTMLResponse)
def copilot_page(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    allow(user)
    mems, error = [], None
    try:
        mems = copilot.memories(db, user)
    except Exception as e:                       # noqa: BLE001
        log.exception("copilot memories failed")
        error = f"Could not load memories ({type(e).__name__})."
    return templates.TemplateResponse(request, "copilot.html", ctx(
        request, user, db, active="copilot", enabled=copilot.enabled(), memories=mems, error=error,
        transcript=copilot.transcript(db, user, "general"), engine=engine_label(),
        recent_cases=[r for r in copilot.recent_scopes(db, user) if r.scope.startswith("case:")]))


@router.post("/copilot/ask")
async def ask_general(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    allow(user)
    f = await request.form()
    try:
        from starlette.concurrency import run_in_threadpool
        await run_in_threadpool(copilot.ask, db, user, question_of(f), "general")
    except Exception as e:                       # noqa: BLE001
        return _fail("/copilot", e)
    return RedirectResponse("/copilot#cp-end", 303)


@router.post("/copilot/remember")
async def remember(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    allow(user)
    f = await request.form()
    try:
        from starlette.concurrency import run_in_threadpool
        await run_in_threadpool(copilot.remember, db, user, f.get("text"))
    except Exception as e:                       # noqa: BLE001
        return _fail("/copilot", e)
    return RedirectResponse("/copilot?msg=Saved.+The+copilot+will+use+this+from+now+on.", 303)


@router.post("/copilot/memories/{memory_id}/delete")
def forget(memory_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    allow(user)
    try:
        copilot.forget(db, user, memory_id)
    except Exception as e:                       # noqa: BLE001
        return _fail("/copilot", e)
    return RedirectResponse("/copilot?msg=Forgotten.", 303)
