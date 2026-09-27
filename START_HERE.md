# Insight Rx: run it on any computer (no GPU, no cluster)

> **Insight Rx is a new HCP engagement channel triggered by a clinical signal: one no-needle eye photo tells the clinician what to treat, which protein and drug to target, and who to engage next (a specialist, a trial or the manufacturer).**

HackGT 13 · Impiricus challenge · Team **Coding Claws**: Nagur Shareef Shaik, Sahith Reddy Thummala, Pranav Nagothu and Geethanjali Nagaboina.

This package contains the full application, the trained models and sample retinal photos. The models run **on the CPU** through ONNX Runtime, give the same results as the GPU version, and need no PyTorch, GPU or cluster.

## What you need

- Python **3.10, 3.11 or 3.12** ([python.org](https://www.python.org/downloads/); on Windows, tick "Add to PATH" and keep the `py` launcher).
- About 14 GB of free disk space (6.2 GB zip, 6.3 GB unzipped, plus about 1 GB of dependencies) and **12 GB of RAM minimum** (16 GB recommended). The models use about 7 GB of RAM once loaded.
- Internet access **once**, to install Python packages. The app then runs offline; live trial and NPI lookups fall back to bundled snapshots.

## Run it

**Linux / macOS**
```bash
bash setup.sh        # installs dependencies, writes .env, builds a demo workspace from test_data/ with the real models
bash run.sh          # then open http://127.0.0.1:8000
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
powershell -ExecutionPolicy Bypass -File run.ps1      # then open http://127.0.0.1:8000
```

Sign in as **Dr. Alex Morgan** (referring clinician), then:
1. **Patients:** six test patients (`TD-01` … `TD-06`), already screened by the models on your CPU.
2. **Screen:** upload photos from `test_data/mbrset/patients/<scenario>/` (right eye `OD_*.jpg`, left eye `OS_*.jpg`), add age and medicines, and press **Screen**. Two photos take about 30 s on an 8–12 core CPU. The first page after `run.sh` may take 20–40 s while the models load in the background.
3. **Therapy and trials** (on a patient): therapy options, interaction alerts, personalized protein targets, the 3D AlphaFold viewer and PDF reports.

Other accounts show the other roles: **Dr. Priya Nair** (specialist), **Taylor Brooks** (coordinator) and **Morgan Lee** (manufacturer medical-information desk).

## What's inside

| Path | What |
|---|---|
| `insightrx/` | Application (FastAPI) and ML code |
| `models/insightrx-onnx-v1/` | **CPU-ready models** (ONNX, 5.7 GB): 2 retina seeds with LoRA merged, 3 systemic encoders, scikit-learn heads, calibration, manifest with checksums, and a parity report against PyTorch |
| `weights/` | Original PyTorch checkpoints (for GPU use, retraining or attention maps) |
| `test_data/` | Sample retinal photos: 6 mBRSET test patients with `patient.json`, single mBRSET photos, and BRSET photos from other cameras |
| `docs/` | User guide, architecture, Impiricus fit, portable-inference notes, screenshots, sample PDF reports, and the demo video |
| `scripts/` | `seed_local.py` (demo workspace), `predict.py` (score a folder of photos), `export_models.py`, training and demo tools |
| `tests/` | Test suite: `.venv/bin/python -m pytest tests` |

## Useful commands

```bash
# score any folder of fundus photos to a CSV (CPU)
.venv/bin/python scripts/predict.py test_data/brset --out brset_predictions.csv

# run the tests
.venv/bin/python -m pytest -q tests
```

To enable optional features, edit `.env` and restart:
- `GEMINI_API_KEY`: evidence briefs written by Gemini.
- `BACKBOARD_API_KEY`: Insight Rx Copilot with persistent memory.

## Notes

- **Speed.** Every screening runs the full validated ensemble: 2 seeds, each on the photo and its mirror image. On an 8–12 core CPU that is about 10–15 s per photo. Faster shortcuts changed decisions in testing, so they are not used. With an NVIDIA GPU, run `pip install onnxruntime-gpu` in the venv; the app picks it up automatically.
- **Attention maps** ("Where the model looked") need the PyTorch weights and a PyTorch install; everything else is identical on CPU.
- **Data use.** The sample photos come from mBRSET and BRSET (PhysioNet, credentialed access). The RETFound encoders are under their authors' non-commercial terms. Keep this package private and use it for evaluation and demonstration only.
- This is a research prototype: the outputs support clinician review and are not diagnoses.
