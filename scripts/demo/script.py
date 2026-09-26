"""
Insight Rx demo storyboard: one source of truth for narration (spoken form), captions (written form), badges and
which slide or app action each scene shows. narrate.py, record.py and compose.py all read SCENES.

Each sentence is either a string (spoken == caption) or a (caption, spoken) pair when pronunciation differs.
Figures and their sources: see ROI in this file and the slide footers (slides.html).
"""

VOICE = "af_heart"          # Kokoro-82M voice
SPEED = 1.08                # slightly brisk: about 150 words per minute
GAP_S = 0.28                # silence between sentences
SCENE_PAD_S = 0.5           # breathing room after each scene's narration

# Sourced figures (fetched 2026-09-26), used by the ROI slide and narration
ROI = {
    "per": 10_000,
    "exam_rate": 0.648,       # Healthy People 2030 D-04: adults with diabetes with a yearly eye exam (2019)
    "dr_prev": 0.26,          # Lundeen et al., JAMA Ophthalmol 2023 (CDC): DR in 26% of people with diabetes
    "vtdr_prev": 0.05,        # same study: vision-threatening DR in 5%
    "cpt_92228_rvu": 0.91,    # CMS PFS 2026 RVU file (RVU26D), non-facility total RVUs
    "conversion_factor": 33.4009,  # CMS PFS 2026 conversion factor (non-QPP), from the same file
}


def roi_numbers(r=ROI):
    per = r["per"]
    rate = r["cpt_92228_rvu"] * r["conversion_factor"]
    return {"gaps": round(per * (1 - r["exam_rate"])), "dr": round(per * r["dr_prev"]),
            "vtdr": round(per * r["vtdr_prev"]), "rate_92228": round(rate, 2), "billing": round(per * rate)}


SCENES = [
    {"id": "title", "slide": "title", "badges": ["Team Coding Claws · HackGT 13 · Impiricus challenge"],
     "say": [("We are team Coding Claws at HackGT 13: Nagur Shareef Shaik, Sahith Reddy Thummala, Pranav Nagothu and Geethanjali Nagaboina.",
              "We are team Coding Claws, at Hack G T thirteen: Nagur Shareef Shaik, Sahith Reddy Thummala, Pranav Nagothu, and Geethanjali Nagaboina."),
             ("This is Insight Rx: from one eye photo to a personalised, target-level therapy plan.",
              "This is Insight R X. From one eye photo, to a personalised, target-level therapy plan.")]},

    {"id": "motivation", "slide": "motivation", "badges": [],
     "say": [("I'm Nagur Shareef, and I research oculomics at Georgia State.",
              "I'm Nagur Shareef, and I research oculomics at Georgia State."),
             "What still amazes me is that the retina is the one place in the body where we can look straight at blood vessels and nerves, with no needle and no scan.",
             ("Yet in most clinics that photo ends as a line in a chart. I built Insight Rx so it starts a treatment plan instead.",
              "Yet in most clinics, that photo ends as a line in a chart. I built Insight R X so it starts a treatment plan instead.")]},

    {"id": "clinical", "slide": "clinical", "badges": ["CDC · JAMA Ophthalmology 2023 · Healthy People 2030"],
     "say": ["Diabetic retinopathy is the leading cause of blindness in working-age adults.",
             "One in four people with diabetes already has it, yet about a third skip their yearly eye exam.",
             "And the retina's tiny vessels mirror the heart, the kidneys and the nerves."]},

    {"id": "screen", "app": "screen", "badges": ["Real patients · mBRSET, Itabuna, Brazil", "DINOv2-L + LoRA ensemble · AUROC 0.98"],
     "say": [("Everything here runs on real patients: retinal photos taken with a handheld smartphone camera in Itabuna, Brazil, from the mBRSET dataset.",
              "Everything here runs on real patients: retinal photos taken with a handheld smartphone camera in Itabuna, Brazil, from the mobile Brazilian retinal dataset."),
             "A primary-care clinician adds both eyes, the patient's age and current medicines.",
             ("A quality gate checks every photo, then our fine-tuned DINOv2 model finds referable diabetic retinopathy in both eyes, and shows where it looked.",
              "A quality gate checks every photo. Then our fine-tuned Dino V 2 model finds referable diabetic retinopathy in both eyes, and shows where it looked."),
             ("On held-out patients it reaches an AUROC of 0.98.",
              "On held-out patients, it reaches an area under the curve of zero point nine eight.")]},

    {"id": "therapy", "app": "therapy", "badges": ["Guidelines: ADA · AAO · KDIGO · FDA labels"],
     "say": ["Insight Rx turns that finding into guideline therapy options for this patient,",
             "and it catches two eye-specific safety risks in the current medicines: semaglutide with retinopathy, and pioglitazone with macular edema."]},

    {"id": "targets", "app": "targets", "badges": ["Personalised target ranking"],
     "say": ["Now the part no screening tool does: personalised target discovery.",
             "The patient's phenotype ranks the protein targets behind their disease, by evidence, guideline support, and how well we understand the drug-binding site."]},

    {"id": "vegfa", "app": "vegfa", "badges": ["AlphaFold DB (DeepMind · EMBL-EBI)", "RCSB Protein Data Bank · ChEMBL · 3Dmol.js"],
     "say": [("The top target for this patient is VEGF-A, the protein that makes retinal vessels leak.",
              "The top target for this patient is V E G F A, the protein that makes retinal vessels leak."),
             "Here is its real structure from the Protein Data Bank, bound by the antibody lineage behind ranibizumab.",
             "Switch to DeepMind's AlphaFold prediction, coloured by confidence.",
             ("We compute the 21 residues that touch the drug. AlphaFold's confidence there is 95 out of 100: reliable for structure-based design, while floppy regions are flagged.",
              "We compute the twenty-one residues that touch the drug. AlphaFold's confidence there is ninety-five out of a hundred: reliable for structure-based design, while floppy regions are flagged."),
             ("Approved and pipeline drugs on the target come from ChEMBL.",
              "Approved and pipeline drugs on the target come from Kemble.")]},

    {"id": "sglt2", "app": "sglt2", "badges": ["Drug-contact residues at 4.5 Å · UniProt"],
     "say": [("The same analysis runs for every target, like empagliflozin sitting in the SGLT2 pocket, mapped to UniProt numbering.",
              "The same analysis runs for every target, like empagliflozin sitting in the S G L T 2 pocket, mapped to Uni Prot numbering.")]},

    {"id": "portfolio", "slide": "reports", "badges": ["Portfolio matrix · PDF reports"],
     "say": ["Across a clinic, eye-detected demand per target meets how crowded each drug space is, showing a pharma partner exactly where to engage.",
             "And one click produces a personalised therapy report and a discovery dossier."]},

    {"id": "workflow", "app": "workflow", "badges": ["ClinicalTrials.gov · CMS NPI Registry · med-info firewall"],
     "say": [("Recruiting trials are matched by finding, real specialists come from the CMS NPI Registry, and manufacturers answer questions through a de-identified, firewalled medical-information channel.",
              "Recruiting trials are matched by finding. Real specialists come from the C M S N P I Registry. And manufacturers answer questions through a de-identified, firewalled medical information channel.")]},

    {"id": "roi", "slide": "roi", "badges": ["Sources: CMS PFS 2026 · CDC · Healthy People 2030"],
     "say": ["The return is concrete. For every ten thousand patients with diabetes:",
             "about thirty-five hundred open eye-exam gaps can close, the ones Medicare and HEDIS quality scores track;",
             ("each physician-read screening bills CPT 92228, about $30 at the 2026 Medicare rate, around $300,000 per ten thousand reads;",
              "each physician-read screening bills C P T nine two two two eight, about thirty dollars at the twenty twenty-six Medicare rate: around three hundred thousand dollars per ten thousand reads;"),
             "and about twenty-six hundred people with retinopathy, five hundred of them vision-threatening, are found and routed to care."]},

    {"id": "social", "slide": "social", "badges": [],
     "say": ["Socially, this is a low-cost, no-needle screen that fits in a clinic bag, built on data from a Brazilian diabetes program, so it can reach rural and underserved patients first.",
             "It is a research prototype that still needs prospective validation."]},

    {"id": "acquire", "slide": "acquire", "badges": [],
     "say": ["Why should Impiricus buy it?",
             "It creates a new engagement moment: a clinical signal, exactly when therapy is decided.",
             "It opens life-science revenue: targeted medical information, trial recruitment, and demand by protein target, inside a compliance firewall.",
             "Its moat is validated retinal AI plus a personalised target engine built on AlphaFold.",
             "And clinics adopt it on their own, because it closes quality gaps and it bills."]},

    {"id": "stack", "slide": "stack", "badges": ["Vercel · Neon · Cloudflare · Google Gemini · EMBL-EBI"],
     "say": [("Under the hood: FastAPI, Neon and Vercel, GPU models over a Cloudflare tunnel, and open data from EMBL-EBI.",
              "Under the hood: Fast A P I, Neon and Vercel, G P U models over a Cloudflare tunnel, and open data from E M B L, E B I."),
             ("Insight Rx. One eye photo. The right therapy, and the right physician.",
              "Insight R X. One eye photo. The right therapy, and the right physician.")]},
]


def sentences(scene):
    """[(caption, spoken)] for a scene."""
    return [(s, s) if isinstance(s, str) else s for s in scene["say"]]
