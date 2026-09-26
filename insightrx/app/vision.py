"""
Vision worker: suitability -> quality gate -> DR / ICDR / edema -> eye & patient aggregation -> systemic heads.
Uses the same calibration, thresholds and aggregation rule that insightrx.ml.evaluate measured on the test split.
If no trained checkpoint is available the service runs in SIMULATED mode and says so on every output.
"""
import glob
import hashlib
import json
import os
import threading
import time

import numpy as np
from PIL import Image as PILImage

REPO_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "weights")
MODEL_DIR = os.environ.get("INSIGHTRX_MODEL_DIR", REPO_WEIGHTS)

SYSTEMIC_LABELS = {
    "systemic_hypertension": "Hypertension", "nephropathy": "Diabetic kidney involvement",
    "vascular_disease": "Diabetes-related vascular disease", "acute_myocardial_infarction": "Prior myocardial infarction",
    "neuropathy": "Diabetic neuropathy", "diabetic_foot": "Diabetic foot", "obesity": "Obesity",
}
# research composites (evaluated only as association signals; no recorded-history row of their own)
COMPOSITE_LABELS = {"cardiovascular": "Cardiovascular composite (vascular disease or prior MI)",
                    "dm_complications": "Diabetes complications composite (kidney, nerve or foot)"}
META_LABELS = {"age": "Age", "sex": "Sex", "dm_time": "Diabetes duration", "insulin": "Insulin use",
               "oraltreatment_dm": "Oral diabetes treatment", "retinal image": "Retinal images"}
RELIABILITY_TEXT = {
    "research": "Passes the release gate: cross-validated AUROC {auc:.2f} (95% CI {lo:.2f} to {hi:.2f})",
    "exploratory": "Exploratory: cross-validated AUROC {auc:.2f} (95% CI {lo:.2f} to {hi:.2f}), below the release gate",
    "near_chance": "Near chance: cross-validated AUROC {auc:.2f} (95% CI {lo:.2f} to {hi:.2f} includes 0.5)",
}
SIGNAL = {"research": "Research signal", "exploratory": "Exploratory signal", "near_chance": "Near-chance score"}
NO_SIGNAL = {"research": "No research signal", "exploratory": "No exploratory signal", "near_chance": "Near-chance score"}


def reliability(h):
    if h["enabled"]:
        return "research"
    return "exploratory" if h["auroc_ci95"][0] > 0.5 else "near_chance"


ICDR_NAMES = ["No apparent DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _to_rgb(img):
    """Flatten palette/alpha/greyscale images onto black (fundus background) as RGB."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = PILImage.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def suitability(path):
    """Cheap fundus-compatibility check (FR06). Not claimed to catch every out-of-distribution input."""
    img = PILImage.open(path).convert("RGB")
    w, h = img.size
    reasons = []
    if min(w, h) < 400:
        reasons.append(f"resolution {w}x{h} below 400 px")
    a = np.asarray(img.resize((128, 128))).astype(float)
    corners = np.concatenate([a[:12, :12].reshape(-1, 3), a[:12, -12:].reshape(-1, 3),
                              a[-12:, :12].reshape(-1, 3), a[-12:, -12:].reshape(-1, 3)])
    centre = a[32:96, 32:96].reshape(-1, 3)
    if corners.mean() > 60:
        reasons.append("no dark border around a circular field of view")
    if not (centre[:, 0].mean() > centre[:, 2].mean() * 1.3):
        reasons.append("colour profile unlike a fundus photograph")
    return {"suitable": not reasons, "reasons": reasons}


class VisionService:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.lock = threading.Lock()
        self.models = []
        self.systemic = None
        self.calib = None
        self.metrics = {}
        self.mode = "simulated"
        self.version = "simulated-v0"
        self._load()

    # -------------------------------------------------------------- loading
    def _load(self):
        cal = os.path.join(self.model_dir, "calibration.json")
        ckpts = sorted(glob.glob(os.path.join(self.model_dir, "ckpt", "image_seed*.pth")))
        if not (os.path.exists(cal) and ckpts):
            return
        import torch
        from insightrx.ml.model import RetiModel
        self.calib = json.load(open(cal))
        self.device = os.environ.get("INSIGHTRX_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        keep = [c for c in ckpts if int(c.split("seed")[-1].split(".")[0]) in self.calib["seeds"]]
        h = hashlib.sha256()
        for c in keep:
            ck = torch.load(c, map_location="cpu", weights_only=False)
            m = RetiModel(heads=ck["heads"], grad_checkpointing=False)
            m.load_state_dict(ck["state_dict"], strict=False)
            m = m.to(self.device).eval()
            if self.device.startswith("cuda"):
                m = m.to(torch.bfloat16)          # runs under bf16 autocast anyway; halves GPU memory
            self.models.append(m)
            self.heads, self.size = ck["heads"], ck["image_size"]
            h.update(f"{c}:{ck['epoch']}:{ck['val_score']}".encode())
        sp = os.path.join(self.model_dir, "ckpt", "systemic_cv.joblib")
        self.encoders = {}
        if os.path.exists(sp):
            import joblib
            from insightrx.ml.frozen import FrozenEncoder
            self.systemic = joblib.load(sp)
            for bb in {h["backbone"] for h in self.systemic["heads"].values() if h["backbone"]}:
                enc = FrozenEncoder(bb).to(self.device)
                self.encoders[bb] = enc.half() if self.device.startswith("cuda") else enc   # features were cached under fp16
        for name in ("image_metrics.json", "systemic_metrics.json", "systemic_cv.json"):
            p = os.path.join(self.model_dir, "metrics", name)
            if os.path.exists(p):
                self.metrics[name] = json.load(open(p))
        self.mode = "live"
        self.version = f"insightrx-dinov2L-lora-ens{len(self.models)}-{h.hexdigest()[:10]}"

    # -------------------------------------------------------------- per-image inference
    def _infer(self, paths):
        import torch
        from insightrx.ml.data import eval_transform, open_fundus
        from insightrx.ml.model import split_logits
        tf = eval_transform(self.size)
        x = torch.stack([tf(open_fundus(p, self.size)) for p in paths]).to(self.device)
        logits, embs = [], []
        # bf16 autocast only on GPU: on CPUs without AMX it is far slower than fp32
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            for m in self.models:
                o, g = m(x, return_embedding=True)
                of, gf = m(torch.flip(x, dims=[3]), return_embedding=True)
                z = torch.cat([o[n].float() for n, *_ in self.heads], 1)
                zf = torch.cat([of[n].float() for n, *_ in self.heads], 1)
                logits.append(((z + zf) / 2).cpu().numpy())
                embs.append(((g.float() + gf.float()) / 2).cpu().numpy())
        return split_logits(np.mean(logits, 0)), np.concatenate(embs, 1)

    def _simulated(self, paths):
        out = []
        for p in paths:
            r = np.random.RandomState(int(hashlib.sha256(open(p, "rb").read()).hexdigest()[:8], 16))
            out.append({"p_quality_poor": float(r.beta(1, 8)), "p_dr": float(r.beta(1.2, 4)), "p_edema": float(r.beta(1, 9)),
                        "icdr_probs": list(np.round(r.dirichlet([6, 1, 1.5, .4, .4]), 3))})
        return out

    # -------------------------------------------------------------- case-level analysis
    def analyze(self, images, patient):
        """images: list of dicts {id, laterality, path or bytes}; patient: metadata dict. Returns a result dict."""
        import tempfile
        tmp = None
        if any(not im.get("path") for im in images):              # byte-only images (database store)
            tmp = tempfile.mkdtemp(prefix="insightrx_")
            images = [dict(im, path=im.get("path") or self._spill(tmp, im)) for im in images]
        try:
            return self._analyze(images, patient)
        finally:
            if tmp:
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def _spill(tmp, im):
        p = os.path.join(tmp, f"{im['id']}.img")
        with open(p, "wb") as fh:
            fh.write(im["bytes"])
        return p

    def _analyze(self, images, patient):
        t0 = time.time()
        suit = {im["id"]: suitability(im["path"]) for im in images}
        ok_imgs = [im for im in images if suit[im["id"]]["suitable"]]
        per = {}
        th = (self.calib or {}).get("thresholds", {"quality_poor": 0.5, "dr_referable_patient": 0.5, "edema": 0.5})
        tq, tdr, ted = th["quality_poor"], th["dr_referable_patient"], th["edema"]
        tunc = th.get("quality_uncertain", tq * 0.5)
        emb = None
        if ok_imgs:
            with self.lock:
                if self.mode == "live":
                    from insightrx.ml.model import ordinal_probs_np
                    z, emb = self._infer([im["path"] for im in ok_imgs])
                    T = self.calib["temperatures"]
                    icdr = ordinal_probs_np(z["icdr"])
                    rows = [{"p_quality_poor": float(sigmoid(z["quality_poor"][i] / T["quality_poor"])),
                             "p_dr": float(sigmoid(z["dr_referable"][i] / T["dr_referable"])),
                             "p_edema": float(sigmoid(z["edema"][i] / T["edema"])),
                             "icdr_probs": [round(float(v), 3) for v in icdr[i]]}
                            for i in range(len(ok_imgs))]
                else:
                    rows = self._simulated([im["path"] for im in ok_imgs])
            for im, r in zip(ok_imgs, rows):
                q = "unassessable" if r["p_quality_poor"] >= tq else ("uncertain" if r["p_quality_poor"] >= tunc else "assessable")
                r.update(quality=q, dr_positive=bool(r["p_dr"] >= tdr) if q != "unassessable" else None,
                         icdr_grade=int(np.argmax(r["icdr_probs"])), edema_flag=bool(r["p_edema"] >= ted))
                per[im["id"]] = r
        for im in images:
            if im["id"] not in per:
                per[im["id"]] = {"quality": "unsupported", "reasons": suit[im["id"]]["reasons"]}

        # ---- eye aggregation (max over assessable views), discrepancy flag
        eyes = {}
        for eye in ("OD", "OS"):
            views = [(im, per[im["id"]]) for im in images if im["laterality"] == eye]
            usable = [(im, r) for im, r in views if r.get("quality") in ("assessable", "uncertain")]
            e = {"n_images": len(views), "n_assessable": sum(r["quality"] == "assessable" for _, r in views),
                 "n_uncertain": sum(r["quality"] == "uncertain" for _, r in views)}
            if not views:
                e["status"] = "Not provided"
            elif not usable:
                e["status"] = "Unable to assess"
            else:
                s = max(r["p_dr"] for _, r in usable)
                pos = [r["dr_positive"] for _, r in usable]
                e.update(score=round(s, 4), max_icdr=max(r["icdr_grade"] for _, r in usable),
                         edema_signal=any(r["edema_flag"] for _, r in usable),
                         discrepant=len(set(pos)) > 1,
                         status="Referable DR signal" if s >= tdr else
                         ("No model finding" if e["n_uncertain"] == 0 else "Uncertain quality"))
            eyes[eye] = e
        unknown_lat = [im["id"] for im in images if im["laterality"] not in ("OD", "OS")]

        statuses = [eyes[e]["status"] for e in ("OD", "OS")]
        if "Referable DR signal" in statuses:
            overall = "Referable DR signal"
        elif all(s == "No model finding" for s in statuses) and not unknown_lat:
            overall = "No model finding"
        elif all(s in ("Unable to assess", "Not provided") for s in statuses):
            overall = "Unable to assess"
        else:
            overall = "Assessment incomplete"
        complete = all(s in ("Referable DR signal", "No model finding") for s in statuses) and not unknown_lat
        scores = [eyes[e]["score"] for e in eyes if "score" in eyes[e]]

        result = {
            "source": self.mode, "model_version": self.version,
            "threshold_version": (self.calib or {}).get("threshold_version", "simulated"),
            "thresholds": {"quality_poor": tq, "dr_referable": tdr, "edema": ted},
            "overall": overall, "complete": complete,
            "patient_score": round(max(scores), 4) if scores else None,
            "score_type": "calibrated probability of the dataset label (ICDR >= 2), image-level temperature scaling"
            if self.mode == "live" else "SIMULATED score (no trained model loaded)",
            "eyes": eyes, "images": per, "unknown_laterality": unknown_lat,
            "discrepant": any(eyes[e].get("discrepant") for e in eyes),
            "systemic": self._systemic(images, per, emb, ok_imgs, patient),
            "limitations": [
                "Research prototype trained on mBRSET (Phelcom Eyer portable camera, dilated, people with diabetes, Brazil).",
                "Output estimates a dataset label for clinician review; it is not a diagnosis or referral rule.",
                "Other cameras, undilated images or other populations have not been validated.",
            ] + (["One or both eyes lack an assessable image: no overall reassurance can be given."] if not complete else [])
              + (["Views of the same eye disagree; review required."] if any(eyes[e].get("discrepant") for e in eyes) else []),
        }
        result["latency_ms"] = int((time.time() - t0) * 1000)
        return result

    def _systemic(self, images, per, emb, ok_imgs, patient):
        """Per-head association signals from frozen-encoder models (5-fold CV validated), with explanations."""
        labels = {**SYSTEMIC_LABELS, **COMPOSITE_LABELS}
        if self.mode != "live" or self.systemic is None:
            return {t: {"label": l, "status": "Not evaluated", "reason": "no validated systemic model loaded"}
                    for t, l in labels.items()}
        usable = [im for im in ok_imgs if per[im["id"]]["quality"] != "unassessable"]
        meta = np.array([patient.get(f, np.nan) for f in self.systemic["metadata_features"]], dtype=float)
        emb_cache = {}
        out = {}
        for t, label in labels.items():
            h = self.systemic["heads"].get(t)
            if not h:
                out[t] = {"label": label, "status": "Not evaluated", "reason": "no model trained"}
                continue
            info = {"label": label, "variant": h["variant"], "cv_auroc": round(h["oof_auroc"], 3),
                    "cv_ci95": [round(v, 3) for v in h["auroc_ci95"]], "n_pos": h["n_pos"], "n": h["n"],
                    "retinal_added_value": {k: (round(v, 3) if isinstance(v, float) else v)
                                            for k, v in h["retinal_added_value"].items() if k != "ci95"}
                                           | {"ci95": [round(v, 3) for v in h["retinal_added_value"]["ci95"]]},
                    "global_importance": {META_LABELS.get(k, k): round(v, 4) for k, v in
                                          sorted(h["permutation_importance"].items(), key=lambda kv: -kv[1])}}
            # every head is scored; reliability travels with the output so weak heads cannot pass as validated
            rel = reliability(h)
            info |= {"reliability": rel, "reliability_text": RELIABILITY_TEXT[rel].format(
                auc=h["oof_auroc"], lo=h["auroc_ci95"][0], hi=h["auroc_ci95"][1])}
            bb = h["backbone"]
            if bb and not usable:
                out[t] = info | {"status": "Abstained", "reason": "no assessable image"}
                continue
            if bb and bb not in emb_cache:
                emb_cache[bb] = self.encoders[bb].encode([im["path"] for im in usable], self.device)
            e = emb_cache[bb].mean(0) if bb else None
            p, contrib = self._systemic_score(h, e, meta)
            per_image = None
            if bb:
                per_image = {str(im["id"]): round(self._systemic_score(h, emb_cache[bb][i], meta, explain=False)[0], 3)
                             for i, im in enumerate(usable)}
            out[t] = info | {
                "status": SIGNAL[rel] if p >= h["threshold"] else NO_SIGNAL[rel],
                "score": round(p, 3), "threshold": round(h["threshold"], 3),
                "score_type": "Platt-calibrated association score for the self-reported mBRSET label",
                "inputs": {"metadata": "age/sex/diabetes history only", "image": "retinal images only",
                           "fused": "retinal images + age/sex/diabetes history"}[h["variant"].split(":")[0]],
                "backbone": bb, "contributions": contrib, "per_image": per_image}
        return out

    def _systemic_score(self, h, e, meta, explain=True):
        """Calibrated score and occlusion contributions (score change when a group is set to the cohort baseline)."""
        from insightrx.ml.systemic_cv import logit

        def score(e_, m_):
            v = h["variant"]
            X = m_[None] if v == "metadata" else e_[None] if v.startswith("image:") else np.concatenate([e_, m_])[None]
            raw = h["model"].predict_proba(X)[:, 1]
            return float(h["platt"].predict_proba(logit(raw))[0, 1])

        p = score(e, meta)
        if not explain:
            return p, None
        contrib = {}
        if e is not None:
            contrib["Retinal images"] = round(p - score(h["baseline_image"], meta), 3)
        if not h["variant"].startswith("image:"):
            for j, f in enumerate(self.systemic["metadata_features"]):
                m2 = meta.copy()
                m2[j] = h["baseline_meta"][j]
                d = p - score(e, m2)
                contrib[META_LABELS.get(f, f) + ("" if np.isfinite(meta[j]) else " (unknown)")] = round(d, 3)
        return p, dict(sorted(contrib.items(), key=lambda kv: -abs(kv[1])))

    # -------------------------------------------------------------- explanation maps
    def explain(self, path, head):
        """PNG overlay of model attention for one image. head: 'dr_referable' | 'quality_poor' | 'edema' | systemic target."""
        import torch
        from PIL import Image as P
        from insightrx.ml.data import eval_transform, open_fundus
        from insightrx.ml.explain import overlay_png, patch_relevance, systemic_image_direction
        if self.mode != "live":
            return None
        if isinstance(path, (bytes, bytearray)):                  # database-stored image
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as fh:
                fh.write(path)
            try:
                return self.explain(fh.name, head)
            finally:
                os.unlink(fh.name)
        with self.lock:
            if head in ("dr_referable", "quality_poor", "edema"):
                x = eval_transform(self.size)(open_fundus(path, self.size))[None].to(self.device)
                cams = []
                for m in self.models:
                    xm = x.to(next(m.parameters()).dtype)
                    c = patch_relevance(m, list(m.backbone.encoder.layer[-5:-1]), xm, lambda o: o[head][:, 0].float())[0]
                    cams.append(c / (c.max() + 1e-8))
                cam = np.mean(cams, 0)
            else:
                h = (self.systemic or {}).get("heads", {}).get(head)
                if not h or not h["backbone"]:
                    return None
                enc = self.encoders[h["backbone"]]
                v = torch.tensor(systemic_image_direction(h, len(h["baseline_image"])), dtype=torch.float32,
                                 device=self.device)
                x = enc.transform()(open_fundus(path, 224))[None].to(self.device)
                blocks = list(enc.vit.encoder.layer[-5:-1] if enc.family == "dinov2" else enc.vit.blocks[-5:-1])
                cam = patch_relevance(enc, blocks, x.to(next(enc.parameters()).dtype), lambda o: o.float() @ v)[0]
            return overlay_png(P.open(path), cam)


_service = None


def get_service():
    """Local models by default; INSIGHTRX_VISION=remote uses a vision worker over HTTP (e.g. from Vercel).
    If the models cannot be loaded (missing weights, GPU out of memory) the app keeps running in SIMULATED mode."""
    global _service
    if _service is None:
        if os.environ.get("INSIGHTRX_VISION") == "remote":
            from .remote_vision import RemoteVisionService
            _service = RemoteVisionService()
        else:
            try:
                _service = VisionService()
            except Exception as e:                      # noqa: BLE001 - degrade, never take the workspace down
                import logging
                logging.getLogger("insightrx").exception("vision models failed to load: %s", e)
                _service = VisionService(model_dir="/nonexistent")
    return _service
