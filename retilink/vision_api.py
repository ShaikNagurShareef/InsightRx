"""
RetiLink vision worker: the GPU models behind a small authenticated HTTP API.

The web app (e.g. on Vercel) sends the images of one case and receives the analysis; nothing else leaves this
machine. Run with scripts/run_vision_tunnel.sh (uvicorn + a Cloudflare quick tunnel + registration with the app).

  RETILINK_VISION_KEY=<shared secret> uvicorn retilink.vision_api:app --host 127.0.0.1 --port 8100
"""
import json
import os
import secrets
import shutil
import tempfile

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response

from .app.vision import get_service

KEY = os.environ.get("RETILINK_VISION_KEY", "")
MAX_BYTES, MAX_FILES = 10 * 1024 * 1024, 8
app = FastAPI(title="RetiLink vision worker", docs_url=None, redoc_url=None, openapi_url=None)


def check(key):
    if not KEY or not secrets.compare_digest(key or "", KEY):
        raise HTTPException(401, "bad key")


@app.get("/health")
def health(x_retilink_key: str = Header(None)):
    check(x_retilink_key)
    svc = get_service()
    return {"mode": svc.mode, "version": svc.version, "metrics": svc.metrics, "calib": svc.calib}


@app.post("/analyze")
async def analyze(files: list[UploadFile] = File(...), meta: str = Form(...), patient: str = Form("{}"),
                  x_retilink_key: str = Header(None)):
    check(x_retilink_key)
    meta, patient = json.loads(meta), json.loads(patient)
    if len(files) != len(meta) or len(files) > MAX_FILES:
        raise HTTPException(422, "files/meta mismatch")
    tmp = tempfile.mkdtemp(prefix="retilink_")
    try:
        images = []
        for f, m in zip(files, meta):
            data = await f.read()
            if len(data) > MAX_BYTES:
                raise HTTPException(413, "image too large")
            p = os.path.join(tmp, f"{int(m['id'])}.img")
            with open(p, "wb") as fh:
                fh.write(data)
            images.append({"id": int(m["id"]), "path": p, "laterality": str(m.get("laterality", "unknown"))})
        patient = {k: (float(v) if v is not None else float("nan")) for k, v in patient.items()}
        return get_service().analyze(images, patient)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/explain")
async def explain(file: UploadFile = File(...), head: str = Form(...), x_retilink_key: str = Header(None)):
    check(x_retilink_key)
    tmp = tempfile.mkdtemp(prefix="retilink_")
    try:
        p = os.path.join(tmp, "x.img")
        with open(p, "wb") as fh:
            fh.write(await file.read())
        png = get_service().explain(p, head)
        if png is None:
            raise HTTPException(404, "no explanation")
        return Response(png, media_type="image/png")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
