"""Insight Rx Copilot routes (Backboard.io memory + retrieval): ask about a case, ask in general, and see or edit what
the copilot remembers about you."""
import logging
from urllib.parse import quote

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from . import copilot
from . import personalize
from .db import get_db
from .llm import RestrictedPayload
from .main import case_for, ctx, current_user, require, templates
from .models import User

router = APIRouter()
log = logging.getLogger("insightrx.copilot")
CLINICIANS = ("referring", "specialist")


def _fail(url, e):
    if isinstance(e, RestrictedPayload):
        msg = "That question contains identifying details (a reference, date or file name). Rephrase without them."
    elif isinstance(e, copilot.CopilotUnavailable):
        msg = str(e)
    else:
        log.exception("copilot call failed")
        msg = "The copilot could not answer just now. Try again in a moment."
    return RedirectResponse(f"{url}{'&' if '?' in url else '?'}msg={quote(msg)}", 303)


@router.post("/cases/{case_id}/copilot")
async def ask_about_case(case_id: int, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    require(user, *CLINICIANS)
    case = case_for(db, user, case_id)
    f = await request.form()
    back = f"/cases/{case.id}/therapy"
    from .routes_therapy import _fresh_result, _snapshot
    res = _fresh_result(db, case)
    b = personalize.bundle(res, _snapshot(case.patient, res), case.patient.medications or [], with_trials=False)
    try:
        brief = copilot.case_brief(b, res, case.patient.age, case.patient.sex)
        from starlette.concurrency import run_in_threadpool
        await run_in_threadpool(copilot.ask, db, user, f.get("question"), f"case:{case.id}", brief, case.id)
    except Exception as e:                       # noqa: BLE001 - show a safe message, keep the page usable
        return _fail(back, e)
    return RedirectResponse(back + "#copilot", 303)


@router.get("/copilot", response_class=HTMLResponse)
def copilot_page(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    mems, error = [], None
    if copilot.enabled() and user.role in CLINICIANS:
        try:
            mems = copilot.memories(db, user)
        except Exception as e:                   # noqa: BLE001
            log.exception("copilot memories failed")
            error = f"Could not load memories ({type(e).__name__})."
    return templates.TemplateResponse(request, "copilot.html", ctx(
        request, user, db, active="copilot", enabled=copilot.enabled(), memories=mems, error=error,
        transcript=copilot.transcript(db, user, "general"), recent=copilot.recent_scopes(db, user),
        provider=copilot.PROVIDER, model=copilot.MODEL))


@router.post("/copilot/ask")
async def ask_general(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    require(user, *CLINICIANS)
    f = await request.form()
    try:
        from starlette.concurrency import run_in_threadpool
        await run_in_threadpool(copilot.ask, db, user, f.get("question"), "general")
    except Exception as e:                       # noqa: BLE001
        return _fail("/copilot", e)
    return RedirectResponse("/copilot#chat", 303)


@router.post("/copilot/remember")
async def remember(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    require(user, *CLINICIANS)
    f = await request.form()
    try:
        from starlette.concurrency import run_in_threadpool
        await run_in_threadpool(copilot.remember, db, user, f.get("text"))
    except Exception as e:                       # noqa: BLE001
        return _fail("/copilot", e)
    return RedirectResponse("/copilot?msg=Saved.+The+copilot+will+use+this+from+now+on.", 303)


@router.post("/copilot/memories/{memory_id}/delete")
def forget(memory_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    require(user, *CLINICIANS)
    try:
        copilot.forget(db, user, memory_id)
    except Exception as e:                       # noqa: BLE001
        return _fail("/copilot", e)
    return RedirectResponse("/copilot?msg=Forgotten.", 303)
