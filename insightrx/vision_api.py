"""
Insight Rx vision worker: the models behind a small authenticated HTTP API.

Runs anywhere the models load: a Hugging Face Space (CPU, ONNX bundle; see deploy/hf_space), a GPU machine, or a laptop.
The web app sends the images of one case and receives the analysis; nothing else leaves the worker.

  POST /analyze          synchronous (fast hardware)
  POST /jobs             queue an analysis, returns {job_id} at once (slow hardware, e.g. free CPU Spaces)
  GET  /jobs/{job_id}    queued | running | done (+ result) | failed; queued jobs report their position
One background thread runs jobs in order so a small CPU is never oversubscribed.

  INSIGHTRX_VISION_KEY=<shared secret> uvicorn insightrx.vision_api:app --host 0.0.0.0 --port 7860
"""
import json
import os
import queue
import secrets
import shutil
import tempfile
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response

from .app.vision import get_service, service_ready

KEY = os.environ.get("INSIGHTRX_VISION_KEY", "")
MAX_BYTES, MAX_FILES = 10 * 1024 * 1024, 8
app = FastAPI(title="Insight Rx vision worker", docs_url=None, redoc_url=None, openapi_url=None)
JOB_TTL_S, MAX_JOBS = 2 * 3600, 500
_jobs: dict = {}
_queue: "queue.Queue[str]" = queue.Queue()
_jobs_lock = threading.Lock()
_timings: list = []                                    # recent seconds per image, for queue estimates


def _worker():
    while True:
        jid = _queue.get()
        job = _jobs.get(jid)
        if not job:
            continue
        job.update(status="running", started=time.time())
        try:
            job["result"] = get_service().analyze(job["images"], job["patient"])
            job["status"] = "done"
            _timings.append((time.time() - job["started"]) / max(1, len(job["images"])))
            del _timings[:-20]
        except Exception as e:                           # noqa: BLE001 - report, keep the worker alive
            job.update(status="failed", error=type(e).__name__)
        finally:
            job["finished"] = time.time()
            shutil.rmtree(job.pop("tmp"), ignore_errors=True)


def _prune():
    now = time.time()
    with _jobs_lock:
        for jid in [j for j, v in _jobs.items() if v.get("finished") and now - v["finished"] > JOB_TTL_S]:
            _jobs.pop(jid, None)
        while len(_jobs) > MAX_JOBS:
            _jobs.pop(next(iter(_jobs)))


threading.Thread(target=_worker, name="jobs", daemon=True).start()
if os.environ.get("INSIGHTRX_WARMUP") == "1":           # load the models at startup, not on the first request
    threading.Thread(target=get_service, name="model-warmup", daemon=True).start()


def check(key):
    if not KEY or not secrets.compare_digest(key or "", KEY):
        raise HTTPException(401, "bad key")


@app.get("/health")
def health(x_insightrx_key: str = Header(None)):
    check(x_insightrx_key)
    if not service_ready():                              # warming up: answer at once instead of blocking
        threading.Thread(target=get_service, daemon=True).start()
        return {"mode": "loading"}
    svc = get_service()
    return {"mode": svc.mode, "version": svc.version, "metrics": svc.metrics, "calib": svc.calib,
            "backend": svc.backend, "async": True}


async def _receive(files, meta, patient):
    """Validate and spill uploaded images to a temp dir -> (tmp, images, patient)."""
    meta, patient = json.loads(meta), json.loads(patient)
    if len(files) != len(meta) or len(files) > MAX_FILES:
        raise HTTPException(422, "files/meta mismatch")
    tmp = tempfile.mkdtemp(prefix="insightrx_")
    images = []
    try:
        for f, m in zip(files, meta):
            data = await f.read()
            if len(data) > MAX_BYTES:
                raise HTTPException(413, "image too large")
            p = os.path.join(tmp, f"{int(m['id'])}.img")
            with open(p, "wb") as fh:
                fh.write(data)
            images.append({"id": int(m["id"]), "path": p, "laterality": str(m.get("laterality", "unknown"))})
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return tmp, images, {k: (float(v) if v is not None else float("nan")) for k, v in patient.items()}


@app.post("/analyze")
async def analyze(files: list[UploadFile] = File(...), meta: str = Form(...), patient: str = Form("{}"),
                  x_insightrx_key: str = Header(None)):
    check(x_insightrx_key)
    tmp, images, patient = await _receive(files, meta, patient)
    try:
        return get_service().analyze(images, patient)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/jobs", status_code=202)
async def submit(files: list[UploadFile] = File(...), meta: str = Form(...), patient: str = Form("{}"),
                 x_insightrx_key: str = Header(None)):
    check(x_insightrx_key)
    _prune()
    tmp, images, patient = await _receive(files, meta, patient)
    jid = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[jid] = {"status": "queued", "images": images, "patient": patient, "tmp": tmp, "created": time.time()}
    _queue.put(jid)
    return {"job_id": jid, **_status(jid)}


def _status(jid):
    job = _jobs.get(jid)
    if not job:
        raise HTTPException(404, "unknown or expired job")
    out = {"status": job["status"]}
    per_img = (sum(_timings) / len(_timings)) if _timings else None
    if job["status"] == "queued":
        ahead = [j for j, v in list(_jobs.items()) if v["status"] in ("queued", "running") and v["created"] < job["created"]]
        out["position"] = len(ahead)
        if per_img:
            out["eta_s"] = int(per_img * (sum(len(_jobs[j]["images"]) for j in ahead if j in _jobs) + len(job["images"])))
    elif job["status"] == "running":
        out["elapsed_s"] = int(time.time() - job["started"])
        if per_img:
            out["eta_s"] = max(5, int(per_img * len(job["images"]) - out["elapsed_s"]))
    elif job["status"] == "done":
        out["result"] = job["result"]
    else:
        out["error"] = job.get("error")
    return out


@app.get("/jobs/{job_id}")
def job_status(job_id: str, x_insightrx_key: str = Header(None)):
    check(x_insightrx_key)
    return _status(job_id)


@app.post("/explain")
async def explain(file: UploadFile = File(...), head: str = Form(...), x_insightrx_key: str = Header(None)):
    check(x_insightrx_key)
    tmp = tempfile.mkdtemp(prefix="insightrx_")
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
