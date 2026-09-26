# RetiLink

**HackGT 13 · Impiricus challenge: "Invent the next way we engage HCPs."**

A portable retinal photo taken in primary care often leads nowhere. The result sits in a chart, the referral is a fax, and nobody knows whether the retina specialist ever saw the patient. RetiLink makes that photo the start of an accountable exchange between two clinicians:

1. **Screen.** An operator uploads fundus photos. A local vision model checks suitability and image quality, records coverage for each eye, and estimates a referable-DR signal plus research systemic-association signals.
2. **Review.** The referring HCP accepts or overrides the model output and signs their own interpretation. The model output and the clinician's interpretation are always shown as separate things.
3. **Engage.** RetiLink builds a consultation package in which every statement is traced to case data. It includes an evidence brief drawn from curated guidelines. The HCP signs it and sends it to a specialist from an approved directory.
4. **Close the loop.** The specialist explicitly acknowledges the request, asks for more information, or declines it. They then return a signed response. A coordinator schedules the visit and tracks access barriers. The handoff only closes when the referrer acknowledges the response.

This is a new engagement channel: **HCP-to-HCP engagement triggered by a clinical signal**, not marketing and not SMS. It has an action inbox, due-time tracking, notification preferences, and analytics on acknowledgement latency and completion.

> Research prototype. Every patient is synthetic and the workflow analytics are **simulated**. Model outputs estimate mBRSET dataset labels for clinician review; they are not diagnoses.

## Models (mBRSET only)

| Component | What it is |
|---|---|
| Image model | DINOv2-L at 392 px, LoRA (r16) on q/v, last 4 blocks unfrozen, one multi-task head set: **unusable-image**, **referable DR (ICDR ≥ 2)**, **ICDR 0–4 (CORAL ordinal)**, **macular edema**, plus 7 auxiliary systemic heads. Uses horizontal-flip TTA and a 2-seed ensemble. |
| Calibration | Per-head temperature scaling on validation. Thresholds are frozen on validation: quality gate at unusable recall ≥ 95%, patient DR threshold at sensitivity ≥ 90%. |
| Cascade | Quality gate → maximum over assessable views per eye → patient positive if either eye is positive. If either eye has no assessable image, the result is **incomplete**, never reassuring. |
| Systemic heads | 7 reported conditions + 2 composites, **5-fold patient cross-validation over all 1,291 patients**. Features: frozen RETFound (224/448 px) or DINOv2-L embeddings, which never saw the labels, pooled over quality-ok images; metadata = age, sex, diabetes duration, insulin, oral treatment. Variants: metadata-only, image-only and fused (logistic regression); the best out-of-fold variant is chosen. A head is enabled only if CV AUROC ≥ 0.65 **and** its patient-bootstrap 95% CI lower bound is ≥ 0.55; otherwise it shows **Not evaluated**. |
| Split | Frozen patient-grouped 70/10/20 split (OculoMoE protocol P1: 904/129/258 patients). The test set is untouched until evaluation. |

### Results (untouched test split: 258 patients / 1,032 images, exp 1)

| Endpoint | Unit | AUROC (95% CI) | Sens / Spec at frozen threshold |
|---|---|---|---|
| **Referable DR (ICDR ≥ 2)** after the quality gate | patient (n=247, 60 positive) | **0.980** (0.959–0.996) | **81.7% / 98.4%** · PPV 0.94 · NPV 0.94 |
| Referable DR | image (n=981) | 0.983 | 83.6% / 97.8% |
| Macular edema (research) | image (n=986) | 0.983 | 76.4% / 98.3% |
| ICDR 0–4 | image | quadratic κ **0.874**, accuracy 88.6% | – |
| Unusable image (quality gate) | image (53 unusable) | 0.917 | recall 84.9%, falsely rejects 17.9% of usable images |

#### Systemic associations (5-fold patient CV, out-of-fold)

| Target | Pos / n | Metadata only | Best image-only | Chosen (95% CI) | Retinal added value (95% CI) | Status |
|---|---|---|---|---|---|---|
| Hypertension | 914 / 1280 | 0.668 | 0.631 | **0.668** (0.636–0.700), metadata | −0.002 (−0.03, +0.03) | **enabled** |
| Nephropathy | 46 / 1271 | 0.545 | 0.602 | 0.602 (0.51–0.69), RETFound 448 image | +0.057 (−0.05, +0.18) | not evaluated |
| Vascular disease | 217 / 1272 | 0.564 | 0.585 | 0.587 (0.54–0.63), RETFound fused | +0.022 (−0.03, +0.07) | not evaluated |
| Neuropathy | 55 / 1272 | 0.452 | 0.589 | 0.592 (0.52–0.66), DINOv2 fused | **+0.140 (+0.03, +0.25)** | not evaluated |
| Prior MI | 98 / 1270 | 0.616 | 0.587 | 0.616 (0.56–0.67), metadata | −0.011 | not evaluated |
| Diabetic foot | 173 / 1264 | 0.590 | 0.592 | 0.615 (0.57–0.66), RETFound fused | +0.025 | not evaluated |
| Obesity | 102 / 1272 | 0.526 | 0.504 | 0.526, metadata | −0.008 | not evaluated |
| Cardiovascular composite | 275 / 1270 | 0.590 | 0.577 | 0.590, metadata | −0.004 | not evaluated |
| Diabetes complications composite | 250 / 1263 | 0.552 | 0.542 | 0.552, metadata | −0.006 | not evaluated |

What this shows:

- In mBRSET, retinal images add a statistically supported signal only for **neuropathy**, and that head is still too weak to release.
- Hypertension is enabled, but on age, sex and diabetes history alone; the workspace says so.
- The chosen variant is selected among 7 on the same folds, so it is mildly optimistic.
- The first run used a gradient-boosting metadata baseline. It overfit rare targets (nephropathy 0.35, below chance) and was replaced with regularised logistic regression.
- The fine-tuned-model systemic heads evaluated on the P1 split are kept for reference in `weights/metrics/systemic_metrics.json`.

**Missed targets and threshold history.** G2 (sensitivity ≥ 90%) and G3 (unusable recall ≥ 95%) are **not met**.

- **v1 threshold:** set at 90% patient sensitivity on only 28 validation positives. It gave 73.3% / 99.5% on test.
- **v2 threshold (in use):** set at 90% image sensitivity on the gated validation images. This rule was adopted **after** looking at the v1 test result. The v1 metrics are kept in `metrics/image_metrics_v1_patient_threshold.json`.
- **What the model could do:** the ROC curve reaches about 92% sensitivity at 96% specificity. A larger validation set, or cross-fitted thresholds, is the next step; the threshold should not be re-tuned on the test set.

Full metrics: `metrics/image_metrics.json` and `metrics/systemic_metrics.json` under the output directory. The app shows them under **Analytics → Model performance**.

### Outputs and checkpoints

The released weights used by the app are in [`weights/`](weights/), stored with Git LFS: two image checkpoints (199 MB each), `systemic_cv.joblib`, `calibration.json` and aggregate metrics only. No per-patient outputs or mBRSET images are included, and the repository must stay **private** (see Data use below). Training writes everything to `/data/users3/nshaik3/Projects/Oculomics/RetiLink/<RETILINK_EXP_NO>/`:

```
splits.json  label_audit.json  calibration.json
ckpt/image_seed42.pth  ckpt/image_seed43.pth  ckpt/*_history.json  ckpt/systemic_cv.joblib
features/  metrics/
```

### Train

```bash
# SLURM (2 GPUs, one seed per GPU)
ssh nshaik3@arctrdlogin001.rs.gsu.edu
cd /home/users/nshaik3/Desktop/Oculomics/RetiLink && sbatch scripts/JobSubmit.sh
RETILINK_SMOKE=1 sbatch --export=ALL scripts/JobSubmit.sh            # 5-minute end-to-end check

# or one local GPU, seeds run one after the other
GPU=0 RETILINK_EXP_NO=1 bash scripts/train_local.sh
```

## Explainability

| What | How | Where in the app |
|---|---|---|
| DR / image-quality attention maps | Gradient × activation on patch tokens from the last 4 ViT blocks of both fine-tuned models, averaged. Masked to the fundus field of view, top 15% shown. Labelled as model attention, not lesion segmentation. | Retinal assessment → *Show model attention* under each image |
| Why this result | For each eye: which image drove the maximum, its score vs the frozen threshold and the margin, and ICDR grade probabilities. Also added to the consultation package as traced facts. | Retinal assessment side panel; consultation package |
| Systemic: this patient | Occlusion contributions: how the calibrated score changes when the retinal images or each metadata field are set to the cohort baseline. Also per-image scores and an attention map for image-based heads. | Systemic health → *Why / how reliable?* |
| Systemic: cohort level | Grouped permutation importance on held-out folds; CV AUROC with CI; paired "retinal added value" | same, plus Analytics → Model performance |
| Calibration | Reliability curves (test split for DR; out-of-fold for systemic heads) | Analytics → Model performance |

## App

FastAPI with server-rendered Jinja/Tailwind pages and SQLAlchemy (SQLite by default; set `DATABASE_URL` to use Postgres). Vision inference runs on the local GPU.

```bash
python scripts/seed_demo.py --reset          # synthetic orgs, users, 5 scenario cases, simulated history
bash scripts/run_app.sh                       # uses ./weights (override with RETILINK_MODEL_DIR)
# open http://127.0.0.1:8000  -> pick a role
```

Optional: put `GEMINI_API_KEY=...` in `.env` to have evidence briefs written by Gemini. Only the generic question and the curated public passages are sent; a guard blocks case, patient and image content. Without a key, or if the call fails, RetiLink falls back to a deterministic template.

### Deploy (Hugging Face Space)

```bash
python scripts/deploy_space.py               # private Docker Space <hf-user>/RetiLink
```

The Space is built from `deploy/space/Dockerfile`, using the exact library versions the models were trained with and CUDA 12.6 torch. Choose **T4 GPU** hardware in the Space settings: analysis then takes a few seconds, whereas free CPU needs about a minute per image. The Space starts with a synthetic workspace and no images, so upload fundus photographs you are authorised to use. Add `GEMINI_API_KEY` as a Space secret to enable LLM evidence briefs. The SQLite store resets whenever the Space restarts.

### Demo script (3 minutes)

| Time | Account | What to show |
|---|---|---|
| 0:00 | – | The problem: retinal screening results rarely reach the specialist or come back. |
| 0:20 | Sam Rivera (operator) | Case RL-P0104. Upload, then run analysis. Only one eye is present, so the result shows **Assessment incomplete**. |
| 0:55 | Dr. Alex Morgan (referring) | Case RL-P0102. **Referable DR signal**, its limitations, and the Systemic health tab (recorded, model and HCP columns kept separate). Sign the review. |
| 1:25 | Dr. Alex Morgan | Consultation tab. Enter a question and pick Dr. Priya Nair, preview the traced package with its evidence brief, then sign and send. |
| 1:55 | Dr. Priya Nair (specialist) | Inbox. Accept, then sign a response. |
| 2:20 | Taylor Brooks (coordinator) | Schedule the visit and add a transport barrier; the accountable owner changes. |
| 2:40 | Any | Analytics: workflow funnel (SIMULATED), then Model performance (real test metrics and limitations). |

### Tests

```bash
/data/users3/nshaik3/Projects/Oculomics/RetiLink/venv/bin/python -m pytest -q tests
```

The tests cover:

- **Case intake:** analysis is blocked when identity is missing; non-image uploads are rejected; duplicate images are detected.
- **Roles and access:** role checks; cross-tenant and unassigned access is denied.
- **Consultations:** a double-submitted referral creates only one referral; illegal state transitions are rejected; a signature is invalidated when the case changes; the recipient must be granted access.
- **Safety:** unusable or one-eye cases never produce a reassuring result; the Gemini payload guard works.
- **Full handoff:** a complete two-HCP exchange from send to close.

## Layout

```
retilink/ml/    config, data, model, train_image, evaluate, frozen, systemic_cv, explain (train_systemic: P1-split reference)
retilink/app/   main (routes), models, workflow (state machine / audit / tasks), vision, llm, evidence, templates/
scripts/        JobSubmit.sh, train_local.sh, seed_demo.py, run_app.sh, deploy_space.py
deploy/space/   Dockerfile + pinned requirements for the Hugging Face Space
weights/        released checkpoints (Git LFS) + calibration + aggregate metrics
tests/          test_workflow.py
```

## Limitations

- The model was trained and tested on a single dataset: mBRSET (Phelcom Eyer portable camera, dilated eyes, people with diabetes in Brazil). It has not been validated on other cameras or populations.
- Systemic labels are self-reported history. Systemic outputs are associations for review, not diagnoses or risk predictions.
- Due dates, historical referrals and engagement analytics are simulated. No EHR integration, external messaging or real clinical data.
- Dataset: Nakayama LF et al., *mBRSET*, Scientific Data 2025; PhysioNet (credentialed access). mBRSET images are not redistributed and are never sent to external APIs.

## Data use

The weights in `weights/` were trained on credentialed PhysioNet data. Keep this repository and the Space **private** until redistribution of the trained models has been cleared under the mBRSET data use agreement.
