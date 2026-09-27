"""
Client for a remote Insight Rx vision worker (insightrx/vision_api.py), used when the web app runs without the models,
e.g. on Vercel. The worker is a Hugging Face Space (INSIGHTRX_VISION_URL=https://<user>-<space>.hf.space; private Spaces
also need INSIGHTRX_VISION_HF_TOKEN) or any host running the worker. If the worker is unreachable the app keeps
working in SIMULATED mode and says so.

INSIGHTRX_VISION_ASYNC=1 (slow CPU hosts): analyses are queued with submit() and collected with job(), so no web request
has to wait minutes for the models.
"""
import json
import os
import time

import httpx

KEY = os.environ.get("INSIGHTRX_VISION_KEY", "")
HF_TOKEN = os.environ.get("INSIGHTRX_VISION_HF_TOKEN", "")


def headers():
    h = {"X-InsightRx-Key": KEY}
    if HF_TOKEN:                                           # private Hugging Face Space
        h["Authorization"] = f"Bearer {HF_TOKEN}"
    return h


class RemoteVisionService:
    backend = "remote"

    def __init__(self):
        self._health, self._checked = None, 0.0
        self._fallback = None
        self.loading = False

    @property
    def is_async(self):
        return os.environ.get("INSIGHTRX_VISION_ASYNC") == "1"

    # -------------------------------------------------------------- worker discovery / health
    def url(self):
        if os.environ.get("INSIGHTRX_VISION_URL"):
            return os.environ["INSIGHTRX_VISION_URL"].rstrip("/")
        from .db import SessionLocal
        from .models import AppSetting
        with SessionLocal() as db:
            s = db.get(AppSetting, "vision_url")
            return s.value.rstrip("/") if s else None

    def reset(self):
        self._health, self._checked = None, 0.0

    def health(self):
        ttl = 30 if self._health else 5          # re-check quickly after a failure (e.g. worker URL just re-registered)
        if time.time() - self._checked < ttl:
            return self._health
        self._checked, self._health, self.loading = time.time(), None, False
        u = self.url()
        if u and KEY:
            try:
                r = httpx.get(f"{u}/health", headers=headers(), timeout=8)
                mode = r.json().get("mode") if r.status_code == 200 else None
                if mode == "live":
                    self._health = r.json()
                self.loading = mode == "loading" or r.status_code == 503      # warming up / Space waking
            except (httpx.HTTPError, ValueError):
                pass
        return self._health

    def fallback(self):
        if self._fallback is None:
            from .vision import VisionService
            self._fallback = VisionService(model_dir="/nonexistent")     # SIMULATED
        return self._fallback

    @property
    def mode(self):
        return "live" if self.health() else "simulated"

    @property
    def version(self):
        h = self.health()
        return h["version"] if h else self.fallback().version

    @property
    def metrics(self):
        h = self.health()
        return h["metrics"] if h else {}

    @property
    def calib(self):
        h = self.health()
        return h["calib"] if h else None

    # -------------------------------------------------------------- calls
    def analyze(self, images, patient):
        """images: [{id, laterality, bytes or path}] ; patient: metadata dict (NaN for unknown)."""
        h = self.health()
        if h:
            try:
                files = [("files", (f"{im['id']}.jpg", im.get("bytes") or open(im["path"], "rb").read(),
                                    "application/octet-stream")) for im in images]
                meta = json.dumps([{"id": im["id"], "laterality": im["laterality"]} for im in images])
                pat = json.dumps({k: (None if v != v else v) for k, v in patient.items()})
                r = httpx.post(f"{self.url()}/analyze", files=files, data={"meta": meta, "patient": pat},
                               headers=headers(), timeout=55)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError:
                self.reset()
        return self.fallback().analyze(images, patient)

    # -------------------------------------------------------------- queued analyses (slow hosts)
    def submit(self, images, patient):
        """Queue an analysis -> job id, or None if the worker cannot take it (caller falls back or retries)."""
        if not (self.health() or self.loading):
            return None
        try:
            files = [("files", (f"{im['id']}.jpg", im.get("bytes") or open(im["path"], "rb").read(),
                                "application/octet-stream")) for im in images]
            meta = json.dumps([{"id": im["id"], "laterality": im["laterality"]} for im in images])
            pat = json.dumps({k: (None if v != v else v) for k, v in patient.items()})
            r = httpx.post(f"{self.url()}/jobs", files=files, data={"meta": meta, "patient": pat},
                           headers=headers(), timeout=45)
            r.raise_for_status()
            return r.json()["job_id"]
        except (httpx.HTTPError, KeyError, ValueError):
            self.reset()
            return None

    def job(self, job_id):
        """{status: queued|running|done|failed|lost, ...}; 'lost' when the worker no longer knows the job
        (restarted): the caller resubmits."""
        try:
            r = httpx.get(f"{self.url()}/jobs/{job_id}", headers=headers(), timeout=15)
            if r.status_code == 404:
                return {"status": "lost"}
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError):
            return {"status": "unreachable"}

    def explain(self, data, head):
        """data: image bytes, or a file path."""
        if isinstance(data, str):
            with open(data, "rb") as fh:
                data = fh.read()
        if not self.health():
            return None
        try:
            r = httpx.post(f"{self.url()}/explain", files={"file": ("x.jpg", data, "application/octet-stream")},
                           data={"head": head}, headers=headers(), timeout=30)
            return r.content if r.status_code == 200 else None
        except httpx.HTTPError:
            return None
