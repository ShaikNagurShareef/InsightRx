"""Patient explainer routes: plain-language result (Gemini) in English or Portuguese, read aloud (ElevenLabs)."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from sqlalchemy.orm import Session

from . import explainer
from . import workflow as wf
from .db import get_db
from .main import case_for, ctx, current_user, require, templates
from .models import User
from .routes_therapy import _fresh_result, _snapshot

router = APIRouter()
log = logging.getLogger("insightrx.explainer")
CLINICIANS = ("referring", "specialist")


def _facts(db, case):
    res = _fresh_result(db, case)
    return explainer.facts(res, _snapshot(case.patient, res), wf.current_review(db, case))


@router.get("/cases/{case_id}/explain", response_class=HTMLResponse)
def explain_page(case_id: int, request: Request, lang: str = "en", user: User = Depends(current_user),
                 db: Session = Depends(get_db)):
    require(user, *CLINICIANS)
    case = case_for(db, user, case_id)
    lang = lang if lang in explainer.LANGS else "en"
    f = _facts(db, case)
    return templates.TemplateResponse(request, "explainer.html", ctx(
        request, user, db, case=case, lang=lang, langs=explainer.LANGS, facts=f,
        summary=explainer.summary(f, lang) if f else None, voice=explainer.voice_enabled(),
        gemini=explainer.gemini_enabled()))


@router.get("/cases/{case_id}/explain.mp3")
def explain_audio(case_id: int, lang: str = "en", user: User = Depends(current_user), db: Session = Depends(get_db)):
    require(user, *CLINICIANS)
    case = case_for(db, user, case_id)
    f = _facts(db, case)
    if not f:
        raise HTTPException(404, "No completed analysis for this case.")
    try:
        mp3 = explainer.speech(explainer.summary(f, lang)["text"])
    except explainer.ExplainerUnavailable as e:
        log.warning("patient audio unavailable: %s", e)
        raise HTTPException(503, str(e)) from e
    return Response(mp3, media_type="audio/mpeg", headers={"Cache-Control": "private, max-age=600"})
