---
title: Insight Rx Vision
emoji: 👁️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
startup_duration_timeout: 1h
pinned: false
---

# Insight Rx vision worker

Authenticated inference API for [Insight Rx](https://github.com/ShaikNagurShareef/InsightRx) (HackGT 13, Impiricus challenge, team Coding Claws).
It runs the validated retinal ensemble (DINOv2-L + LoRA, 2 seeds with flip TTA) and the systemic models on CPU with ONNX Runtime, using weights from a private model repo.

`GET /health`, `POST /analyze`, `POST /jobs` + `GET /jobs/{id}` (queued analyses). Every call needs the `X-InsightRx-Key` header. Research prototype; outputs support clinician review and are not diagnoses.
