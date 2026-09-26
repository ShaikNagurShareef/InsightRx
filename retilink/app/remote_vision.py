"""
Client for a remote RetiLink vision worker (retilink/vision_api.py), used when the web app runs without the models,
e.g. on Vercel. The worker URL comes from RETILINK_VISION_URL or from the URL the worker registered itself with
(/api/vision/register). If the worker is unreachable the app keeps working in SIMULATED mode and says so.
"""
import json
import os
import time

import httpx

KEY = os.environ.get("RETILINK_VISION_KEY", "")


class RemoteVisionService:
    def __init__(self):
        self._health, self._checked = None, 0.0
        self._fallback = None

    # -------------------------------------------------------------- worker discovery / health
    def url(self):
        if os.environ.get("RETILINK_VISION_URL"):
            return os.environ["RETILINK_VISION_URL"].rstrip("/")
        from .db import SessionLocal
        from .models import AppSetting
        with SessionLocal() as db:
            s = db.get(AppSetting, "vision_url")
            return s.value.rstrip("/") if s else None

    def reset(self):
        self._health, self._checked = None, 0.0

    def health(self):
        if time.time() - self._checked < 30:
            return self._health
        self._checked, self._health = time.time(), None
        u = self.url()
        if u and KEY:
            try:
                r = httpx.get(f"{u}/health", headers={"X-RetiLink-Key": KEY}, timeout=5)
                if r.status_code == 200 and r.json().get("mode") == "live":
                    self._health = r.json()
            except httpx.HTTPError:
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
                               headers={"X-RetiLink-Key": KEY}, timeout=55)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError:
                self.reset()
        return self.fallback().analyze(images, patient)

    def explain(self, data, head):
        """data: image bytes, or a file path."""
        if isinstance(data, str):
            with open(data, "rb") as fh:
                data = fh.read()
        if not self.health():
            return None
        try:
            r = httpx.post(f"{self.url()}/explain", files={"file": ("x.jpg", data, "application/octet-stream")},
                           data={"head": head}, headers={"X-RetiLink-Key": KEY}, timeout=30)
            return r.content if r.status_code == 200 else None
        except httpx.HTTPError:
            return None
