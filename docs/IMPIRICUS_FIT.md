# Insight Rx for Impiricus: judging criteria and commercial fit

> **Insight Rx is a new HCP engagement channel triggered by a clinical signal: one no-needle eye photo tells the clinician what to treat, which protein and drug to target, and who to engage next (a specialist, a trial or the manufacturer).**

**Challenge:** *Invent the next way we engage HCPs.* Build an HCP engagement tool Impiricus does not offer today, either as a net-new channel or as new value on its platform. SMS and existing Impiricus features are off-limits.

**Our answer:** a **clinical-signal-triggered engagement channel**. A no-needle retinal photo taken in primary care becomes a personalised, target-level therapy picture. That moment then opens compliant engagement between:
- the clinician and a specialist;
- the clinician and a manufacturer's medical-information team;
- the patient and a recruiting trial.

The engagement happens when the therapy decision is made, inside the clinician's workflow. It is not a message pushed to a phone.

Team **Coding Claws** (HackGT 13): Nagur Shareef Shaik, Sahith Reddy Thummala, Pranav Nagothu and Geethanjali Nagaboina.
Demo video: [docs/demo/InsightRx_demo.mp4](demo/InsightRx_demo.mp4) · Live: https://insightrx-hcp.vercel.app

## Rules check

| Rule | How Insight Rx complies |
|---|---|
| A new HCP engagement tool | The engagement is triggered by a clinical finding at the point of care: signed consults, NPI Registry referrals, med-info requests and trial matches. |
| A net-new channel or new value on the platform | It is a new **channel** (in-workflow, signal-triggered) and new **data**: eye-detected demand per protein target, plus qualified med-info and trial-eligible counts. |
| No SMS | There is no SMS anywhere. Notifications are in-app, with digest and quiet hours. |
| No existing Impiricus features | Retinal AI, the personalised target engine, AlphaFold structure insights, the CMS quality and billing layer and the NPI Registry handoff are all new capabilities. |

## Scorecard: each judging criterion and the evidence

| Criterion | What we built | Evidence |
|---|---|---|
| **Impact on the HCP** | One photo gives a referable-DR verdict with attention maps, a whole-body view, guideline therapy options, eye-specific drug-safety alerts, ranked targets, trials and a one-click consult or referral letter. It saves chart review and guesswork at the moment of decision, and closes the diabetic eye-exam quality gap the clinician is measured on. | [Screen result](screenshots/03_screen_result.png) · [alerts](screenshots/04_screen_treatment.png) · [consult](screenshots/14_consult_specialist.png) · [CMS gap](screenshots/19_cms_quality_billing.png) |
| **Originality** | Screening tools stop at "refer". Insight Rx carries the retinal phenotype **to the protein target, the drug and the physician for each patient**. It measures AlphaFold confidence at the exact drug-contact residues from real PDB complexes, and ties drug safety to what the retina shows. | [Personalised targets](screenshots/05_screen_personal_targets.png) · [VEGF-A AlphaFold](screenshots/07_target_vegfa_alphafold.png) · [dossier](reports/target_dossier_VEGFA.pdf) |
| **Technical execution** | A DINOv2-L + LoRA ensemble trained on real Brazilian portable-camera data reaches **AUROC 0.980** on held-out patients. It adds calibrated thresholds, a quality gate and explainability, plus systemic models with a release gate. The system is FastAPI on Vercel with Neon Postgres, a live GPU worker over a Cloudflare tunnel, a strict CSP, tenant and role isolation, an audit trail anchored on Solana, 84 tests and a reproducible demo pipeline. Partner services add copilot memory (Backboard), a bilingual patient explainer (Gemini) read aloud (ElevenLabs), and finding trends on Tiger Data. | [Architecture](ARCHITECTURE.md) · [Model performance](screenshots/20_model_performance.png) · README results |
| **Commercial fit** | Three paying customers, a compliance design built for pharma, a land-and-expand path through primary care, and a direct layer onto Impiricus's HCP network. Details below. | This document · [ROI slide](demo/slides/slide_roi.jpg) · [portfolio report](reports/portfolio_report.pdf) |

## Commercial fit in detail

### Who pays, and for what

| Customer | Buys | Why they pay |
|---|---|---|
| **Life-science brands and medical affairs** | Qualified **medical-information engagements** triggered by a clinical finding. Also **eye-detected demand by protein target** (portfolio matrix) and target dossiers. | They reach the HCP exactly when a relevant finding is on screen, with a de-identified context, inside a firewall. This is a higher-intent moment than any broadcast channel. |
| **Trial sponsors and CROs** | **Trial-eligible patient signals** by finding (DR, macular edema, kidney, heart, nerve), matched to recruiting trials near the clinic. | Recruitment is a well-known bottleneck, and a screening in primary care surfaces candidates before they reach a specialist. |
| **Health systems and primary-care networks** | A per-screen subscription. | Each screen can close an eye-exam quality gap (CMS131 / HEDIS EED) and bill **CPT 92228 (about $30.39 at the 2026 Medicare national rate)**, so the tool pays for itself. |

### Illustrative unit economics per 10,000 patients with diabetes screened (sourced)

| Metric | Value | Source |
|---|---|---|
| Open eye-exam gaps that can close | **3,520** | Healthy People 2030 D-04: 64.8% had a yearly exam |
| Physician-read screening billing | **about $304K** (10,000 × $30.39) | CMS PFS 2026 RVU file: 0.91 RVU × $33.4009 |
| People with retinopathy found | **2,600**, including **500** vision-threatening | Lundeen et al., JAMA Ophthalmology 2023 (CDC): 26% and 5% |
| Life-science signals | Counted per finding and per target, not priced | Insight Rx portfolio report |

These figures are illustrative. Actual payment depends on payer mix, locality and eligibility.

### Why it fits Impiricus specifically

1. **It layers onto an existing HCP network.** Every screening creates an engagement opportunity tied to a clinician, a finding and a therapy class, which is exactly the unit of value an HCP-engagement company monetises. It uses a new channel (the clinical workflow), not SMS.
2. **It is compliance-first by design.**
   - The medical-information desk sees only an age band and finding labels; it cannot open cases.
   - Sponsored content is labelled and never changes ranking, scores or referrals, and a test proves it.
   - Every action is audited.
3. **It creates a proprietary data asset:** anonymised, clinic-level **demand per protein target**, detected non-invasively. This is market intelligence no messaging platform has.
4. **It has a defensible moat:** validated retinal AI, plus a personalised target engine built on AlphaFold, the PDB, UniProt and ChEMBL, plus a CMS billing and quality layer that drives adoption without a sales push.
5. **It is cheap to scale.** It runs on serverless infrastructure and a single GPU worker, and the whole stack runs on free tiers today.

### Go-to-market

1. **Land:** primary-care and diabetes clinics, on the quality-gap and billing story.
2. **Expand:** switch on med-info and trial-matching for life-science partners through Impiricus.
3. **Scale:** add EHR integration (FHIR) and more targets and conditions, pursue an FDA pathway to unlock autonomous CPT 92229, and run prospective validation.

### Honest risks and mitigations

| Risk | Mitigation |
|---|---|
| Not FDA-cleared | Positioned as decision support with a physician read (CPT 92228); the 92229 pathway is on the roadmap. |
| Domain shift beyond mBRSET cameras | A quality gate and exploratory labels; external validation is planned. The test on BRSET samples reached AUROC 0.97. |
| Pharma influence concerns | A firewall, labelling and audit, and ranking that is invariant to sponsorship. |
| Systemic signals are modest | They are shown with evidence tiers (research, exploratory or near chance) and never presented as diagnoses. |
