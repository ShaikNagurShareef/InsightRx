"""
Insight Rx demo storyboard: one source of truth for narration (spoken form), captions (written form), badges and
which slide or app action each scene shows. narrate.py, record.py and compose.py all read SCENES.

Each sentence is either a string (spoken == caption) or a (caption, spoken) pair when pronunciation differs.
Figures and their sources: see ROI in this file and the slide footers (slides.html).
"""

ENGINE = "vibevoice"        # Microsoft VibeVoice-1.5B (MIT): long-form, human-sounding narration read scene by scene
VV_MODEL = "microsoft/VibeVoice-1.5B"
VV_VOICE = "/data/users3/nshaik3/Projects/Oculomics/RetiLink/tools/VibeVoice/demo/voices/in-Samuel_man.wav"
VV_CFG = 1.3                # classifier-free guidance for VibeVoice
EXAGGERATION = 0.55         # Chatterbox expressiveness (0.5 neutral; higher = more animated)
CFG_WEIGHT = 0.45           # Chatterbox pacing/adherence (lower = slower, more deliberate delivery)
VOICE = "af_heart"          # Kokoro fallback voice
SPEED = 1.08                # Kokoro fallback speed
GAP_S = 0.0 if ENGINE == "vibevoice" else 0.32   # VibeVoice clips keep their own natural pauses
SCENE_PAD_S = 0.7           # breathing room after each scene's narration

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
     "say": [("Hi! We're team Coding Claws at HackGT 13: Nagur Shareef Shaik, Sahith Reddy Thummala, Pranav Nagothu and Geethanjali Nagaboina.",
              "Hi! We're team Coding Claws, at HackGT thirteen. I'm Nuh-goor Sha-reef Shaik, with Saa-hith Reddy Thoo-mala, Pra-nuv Naa-go-thoo, and Geeth-aanjali Naa-ga-boyna."),
             ("This is Insight Rx, a new HCP engagement channel triggered by a clinical signal. One no-needle eye photo tells the clinician what to treat, which protein and drug to target, and who to engage next.",
              "And this is Insight R-X. It's a new way to engage H-C-Ps, triggered by a clinical signal. One simple, no-needle eye photo tells the clinician what to treat, which protein and drug to target, and who to talk to next.")]},

    {"id": "motivation", "slide": "motivation", "badges": [],
     "say": [("I research oculomics at Georgia State.",
              "So, a little about why. I research oculomics at Georgia State."),
             ("What still amazes me is that the retina is the one place in the body where we can look straight at blood vessels and nerves, with no needle and no scan.",
              "And what still amazes me is that the retina is the one place in the body where you can look straight at blood vessels and nerves. No needle. No scan."),
             ("Yet in most clinics that photo ends as a line in a chart. We built Insight Rx so it starts a treatment plan instead.",
              "But in most clinics, that photo just ends up as a line in a chart. We built Insight R-X so it starts a treatment plan instead.")]},

    {"id": "clinical", "slide": "clinical", "badges": ["CDC · JAMA Ophthalmology 2023 · Healthy People 2030"],
     "say": [("Diabetic retinopathy is the leading cause of blindness in working-age adults.",
              "Here's the problem. Diabetic retinopathy is the leading cause of blindness in working-age adults."),
             ("One in four people with diabetes already has it, yet about a third skip their yearly eye exam.",
              "One in four people with diabetes already has it, and yet about a third of them skip their yearly eye exam."),
             ("And the retina's tiny vessels mirror the heart, the kidneys and the nerves.",
              "And those tiny vessels in the retina? They mirror what's happening in the heart, the kidneys, and the nerves.")]},

    {"id": "screen", "app": "screen", "badges": ["Real patients · mBRSET, Itabuna, Brazil", "DINOv2-L + LoRA ensemble · AUROC 0.98"],
     "say": [("Everything you'll see runs on real patients: retinal photos taken with a handheld smartphone camera in Itabuna, Brazil, from the mBRSET dataset.",
              "Everything you're about to see runs on real patients: retinal photos taken with a handheld smartphone camera in Itabuna, Brazil, from the mobile Brazilian retinal dataset."),
             ("A primary-care clinician adds both eyes, the patient's age and current medicines.",
              "A primary-care clinician adds both eyes, the patient's age, and their current medicines."),
             ("A quality gate checks every photo, then our fine-tuned DINOv2 model finds referable diabetic retinopathy in both eyes, and shows where it looked.",
              "A quality gate checks every photo first. Then our fine-tuned Dino-V-two model finds referable diabetic retinopathy in both eyes, and shows you exactly where it looked."),
             ("On held-out patients it reaches an AUROC of 0.98.",
              "On patients it has never seen, it reaches an area under the curve of point nine eight.")]},

    {"id": "therapy", "app": "therapy", "badges": ["Guidelines: ADA · AAO · KDIGO · FDA labels"],
     "say": [("Insight Rx turns that finding into guideline therapy options for this patient,",
              "From there, Insight R-X turns that finding into guideline therapy options for this patient."),
             ("and it catches two eye-specific safety risks in the current medicines: semaglutide with retinopathy, and pioglitazone with macular edema.",
              "And it catches two eye-specific safety risks in the medicines they're already on: semaglutide with retinopathy, and pioglitazone with macular edema.")]},

    {"id": "targets", "app": "targets", "badges": ["Personalised target ranking"],
     "say": [("Now the part no screening tool does: personalised target discovery.",
              "Now here's the part no screening tool does. Personalized target discovery."),
             ("The patient's phenotype ranks the protein targets behind their disease, by evidence strength, guideline support, and how well we understand the drug-binding site.",
              "This patient's phenotype ranks the protein targets behind their disease, by the strength of the evidence, guideline support, and how well we actually understand the drug-binding site."),
             ("Every rank explains itself.",
              "And every rank explains itself.")]},

    {"id": "vegfa", "app": "vegfa", "badges": ["AlphaFold DB (DeepMind · EMBL-EBI)", "RCSB Protein Data Bank · ChEMBL · 3Dmol.js"],
     "say": [("The top target for this patient is VEGF-A, the protein that makes retinal vessels leak.",
              "For this patient, the top target is V-E-G-F A, the protein that makes retinal vessels leak."),
             ("Here's its real structure from the Protein Data Bank, bound by the antibody lineage behind ranibizumab.",
              "Here's its real structure from the Protein Data Bank, bound by the antibody family behind ranibizumab."),
             ("Switch to DeepMind's AlphaFold prediction, coloured by confidence.",
              "Now let's switch to DeepMind's AlphaFold prediction, colored by confidence."),
             ("We compute the 21 residues that touch the drug. AlphaFold's confidence there is 95 out of 100: reliable for structure-based design, while floppy regions are flagged.",
              "We compute the twenty-one residues that actually touch the drug. And AlphaFold's confidence right there is ninety-five out of a hundred, reliable enough for structure-based design, while the floppy regions get flagged."),
             ("Approved and pipeline drugs on the target come from ChEMBL.",
              "The approved and pipeline drugs on this target come straight from the Kem-B-L database.")]},

    {"id": "sglt2", "app": "sglt2", "badges": ["Drug-contact residues at 4.5 Å · UniProt"],
     "say": [("The same analysis runs for every target, like empagliflozin sitting in the SGLT2 pocket, mapped to UniProt numbering.",
              "The same analysis runs for every target. Here's empagliflozin, sitting right in the S-G-L-T-two pocket.")]},

    {"id": "portfolio", "slide": "reports", "badges": ["Portfolio matrix · PDF reports"],
     "say": [("Across a clinic, eye-detected demand per target meets how crowded each drug space is, showing a pharma partner exactly where to engage.",
              "Zoom out to a whole clinic, and you can see demand for each target, detected from the eye, against how crowded that drug space is. That shows a pharma partner exactly where to engage."),
             ("And one click produces a personalised therapy report and a discovery dossier.",
              "And with one click, you get a personalized therapy report and a discovery dossier.")]},

    {"id": "workflow", "app": "workflow", "badges": ["ClinicalTrials.gov · CMS NPI Registry · med-info firewall"],
     "say": [("Recruiting trials are matched by finding, real specialists come from the CMS NPI Registry, and manufacturers answer questions through a de-identified, firewalled medical-information channel.",
              "Recruiting trials are matched by finding. Real specialists come from the C-M-S N-P-I registry. And manufacturers answer questions through a de-identified, firewalled medical information channel.")]},

    {"id": "novelty", "slide": "novelty", "badges": ["What makes Insight Rx different"],
     "say": [("So what makes this new?", "So, what makes this new?"),
             ("Retinal screening tools stop at 'refer'. Insight Rx carries one photo all the way to the protein target, the drug and the physician, for each patient.",
              "Retinal screening tools stop at refer. Insight R-X carries one photo all the way to the protein target, the drug, and the physician, for each individual patient."),
             ("It checks drug safety against what the eye shows, measures AlphaFold confidence at the exact drug-binding site, and turns the finding into a compliant engagement for clinicians and manufacturers.",
              "It checks drug safety against what the eye actually shows, measures AlphaFold confidence at the exact drug-binding site, and turns that finding into a compliant engagement for clinicians and manufacturers.")]},

    {"id": "roi", "slide": "roi", "badges": ["Sources: CMS PFS 2026 · CDC · Healthy People 2030"],
     "say": [("The return is concrete. For every ten thousand patients with diabetes:",
              "And the return is concrete. For every ten thousand patients with diabetes,"),
             ("about 3,500 open eye-exam gaps can close, the ones Medicare and HEDIS quality scores track;",
              "about thirty-five hundred open eye-exam gaps can close. Those are the ones Medicare and HEDIS quality scores track."),
             ("each physician-read screening bills CPT 92228, about $30 at the 2026 Medicare rate, around $300,000 per ten thousand reads;",
              "Each physician-read screening bills as C-P-T ninety-two, two twenty-eight. That's about thirty dollars at the twenty twenty-six Medicare rate, or around three hundred thousand dollars per ten thousand reads."),
             ("and about 2,600 people with retinopathy, 500 of them vision-threatening, are found and routed to care.",
              "And about twenty-six hundred people with retinopathy, five hundred of them vision-threatening, are found and routed to care.")]},

    {"id": "social", "slide": "social", "badges": [],
     "say": [("Socially, it's a low-cost, no-needle screen that fits in a clinic bag, built on data from a Brazilian diabetes program, so it can reach rural and underserved patients first.",
              "Socially, this is a low-cost, no-needle screen that fits in a clinic bag. It's built on data from a Brazilian diabetes program, so it can reach rural and underserved patients first."),
             ("It's a research prototype that still needs prospective validation, and every line of code is open.",
              "To be clear, it's a research prototype that still needs prospective validation. And every line of code is open.")]},

    {"id": "acquire", "slide": "acquire", "badges": ["Commercial fit"],
     "say": [("Why should Impiricus buy it?", "So why should Impiricus buy it?"),
             ("It's a new channel on Impiricus's HCP network: a clinical signal, not an SMS, at the moment therapy is decided.",
              "Because it's a brand-new channel on Impiricus's H-C-P network. A clinical signal, not a text message, right at the moment therapy is decided."),
             ("Manufacturers pay per qualified medical-information engagement and trial referral, inside a compliance firewall. Clinics pay per screen, offset by billable reads.",
              "Manufacturers pay per qualified medical-information engagement and trial referral, all inside a compliance firewall. And clinics pay per screen, which billable reads help cover."),
             ("And the moat is validated retinal AI plus a personalised target engine built on AlphaFold.",
              "The moat? Validated retinal AI, plus a personalized target engine built on AlphaFold.")]},

    {"id": "stack", "slide": "stack", "badges": ["Vercel · Neon · Cloudflare · Google Gemini · EMBL-EBI"],
     "say": [("Under the hood: FastAPI and Neon Postgres on Vercel, GPU models over a Cloudflare tunnel, Gemini behind a restricted-data guard, and open data from EMBL-EBI.",
              "Under the hood, it's Fast-A-P-I and Neon Postgres on Vercel, G-P-U models over a Cloudflare tunnel, Gemini behind a restricted-data guard, and open data from the European Bioinformatics Institute."),
             ("Insight Rx: HCP engagement, triggered by a clinical signal. One eye photo. The right therapy, and the right physician.",
              "Insight R-X. H-C-P engagement, triggered by a clinical signal. One eye photo. The right therapy, and the right physician. Thanks for watching!")]},
]

# Spoken-form spellings (for the voice) mapped back to how they're written (for captions)
CAPTION_FORMS = [
    ("I'm Nuh-goor Sha-reef Shaik, with Saa-hith Reddy Thoo-mala, Pra-nuv Naa-go-thoo, and Geeth-aanjali Naa-ga-boyna",
     "I'm Nagur Shareef Shaik, with Sahith Reddy Thummala, Pranav Nagothu and Geethanjali Nagaboina"),
    ("Insight R-X", "Insight Rx"), ("H-C-Ps", "HCPs"), ("H-C-P", "HCP"), ("Dino-V-two", "DINOv2"), ("V-E-G-F A", "VEGF-A"),
    ("the Kem-B-L database", "ChEMBL"), ("S-G-L-T-two", "SGLT2"), ("C-M-S N-P-I registry", "CMS NPI Registry"),
    ("C-P-T ninety-two, two twenty-eight", "CPT 92228"), ("Fast-A-P-I", "FastAPI"), ("G-P-U", "GPU"),
    ("the European Bioinformatics Institute", "EMBL-EBI"), ("HackGT thirteen", "HackGT 13"),
    ("the mobile Brazilian retinal dataset", "the mBRSET dataset"),
    ("an area under the curve of point nine eight", "an AUROC of 0.98"),
    ("thirty-five hundred", "3,500"), ("twenty-six hundred", "2,600"), ("twenty-one", "21"),
    ("ninety-five out of a hundred", "95 out of 100"), ("thirty dollars", "$30"), ("twenty twenty-six", "2026"),
    ("three hundred thousand dollars", "$300,000"), ("five hundred", "500"), ("ten thousand", "10,000"),
]


def to_caption(spoken):
    for said, written in CAPTION_FORMS:
        spoken = spoken.replace(said, written)
    return spoken


def sentences(scene):
    """[(caption, spoken)] for a scene. Captions are derived from the spoken line so they always match the voice."""
    return [(to_caption(s if isinstance(s, str) else s[1]), s if isinstance(s, str) else s[1]) for s in scene["say"]]
