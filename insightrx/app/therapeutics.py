"""
Therapeutics layer: from an eye-derived signal to targets, therapy options, interactions and trials.

The retina does not discover drugs; it phenotypes the patient. That phenotype (findings) selects guideline therapy
classes, the molecular targets behind them (AlphaFold / PDB structures, ChEMBL drugs) and recruiting trials.
Ranking is guideline-driven only. Sponsored content lives in SPONSORED, is attached after ranking and never reorders
anything (tests/test_therapeutics.py asserts this).
"""
import json
import os
import re
from functools import lru_cache

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
STRUCT_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "public", "static", "structures")
SEVERITY_RANK = {"major": 0, "moderate": 1, "minor": 2}
STATE_STRENGTH = {"signal": "Research signal", "recorded": "Known condition", "exploratory": "Exploratory signal"}
FINDING_ORDER = ["dr", "edema", "kidneys", "heart", "nerves", "metabolism"]
MAX_MEDS = 15
SALT = re.compile(r" (potassium|hydrochloride|maleate|mesylate|sodium|acetate|besylate|arginine|cilexetil|medoxomil|"
                  r"kamedoxomil|propanediol|pidolate|gamma|pivalate)$")
HIDE = {"NR3C2": {"felodipine", "nimodipine", "desoxycorticosterone", "fludrocortisone", "drospirenone", "canrenone"},
        "AGTR1": {"angiotensin ii", "saralasin"}}
PHASE = {4: "Approved", 3: "Phase 3", 2: "Phase 2", 1: "Phase 1"}
CLASS_COVERS = {"acei": {"acei", "arb"}}                  # the 'ACE inhibitor or ARB' option is met by either

SPONSORED = [   # fictional sponsor; shown only after guideline-ranked options, always labelled
    {"sponsor": "Demo Pharma (fictional)", "finding": "kidneys", "title": "Kidney protection in diabetes: CME module",
     "text": "A 20-minute accredited module on albuminuria testing and kidney-protective therapy in type 2 diabetes."},
    {"sponsor": "Demo Pharma (fictional)", "finding": "edema", "title": "Retina referral pathway toolkit",
     "text": "Printable patient leaflet and referral checklist for diabetic macular edema."},
]


@lru_cache(maxsize=None)
def _load(name):
    return json.load(open(os.path.join(DATA, name)))


def catalogue():
    return _load("therapeutics.json")


def interactions_db():
    return _load("interactions.json")


def cms():
    return _load("cms.json")


def common_meds():
    return interactions_db()["common"]


def normalize_meds(values) -> list:
    """Validated generic names: lowercase letters, spaces and hyphens only, deduplicated, bounded."""
    out = []
    for v in values or []:
        for part in str(v).split(","):
            name = part.strip().lower()
            if re.fullmatch(r"[a-z][a-z \-]{1,39}", name) and name not in out:      # over-long names are rejected
                out.append(name)
    return out[:MAX_MEDS]


# ------------------------------------------------------------------ findings from a screening result
def findings(result, snapshot) -> list:
    """Ordered findings: validated retinal ones first, then organ signals / known conditions, then diabetes itself."""
    res = result or {}
    per = res.get("images") or {}
    found = {}
    if res.get("overall") == "Referable DR signal":
        found["dr"] = "Validated retinal model"
    if any((r or {}).get("edema_flag") for r in per.values()):
        found["edema"] = "Retinal research signal"
    for o in snapshot or []:
        if o["key"] in ("heart", "kidneys", "nerves") and o["state"] in STATE_STRENGTH:
            found[o["key"]] = STATE_STRENGTH[o["state"]]
    found["metabolism"] = "Applies to every screened patient"
    labels = catalogue()["findings"]
    return [{"key": k, "label": labels[k]["label"], "strength": found[k], "organ": labels[k]["organ"]}
            for k in FINDING_ORDER if k in found]


# ------------------------------------------------------------------ interactions
def med_class(name):
    return interactions_db()["drug_classes"].get(name)


def check_interactions(meds, candidates=(), finding_keys=()) -> list:
    """Alerts among current meds, between candidates and current meds, and drug-disease alerts for the findings."""
    db = interactions_db()
    current = {m: med_class(m) for m in meds}
    cand = {d: med_class(d) for d in candidates}
    everything = {**current, **cand}
    alerts, seen = [], set()
    for rule in db["pairs"]:
        a = [d for d, c in everything.items() if c in rule["a"]]
        b = [d for d, c in everything.items() if c in rule["b"]]
        for x in a:
            for y in b:
                if x == y or (x not in current and y not in current):   # at least one side is already prescribed
                    continue
                key = (rule["title"], frozenset((x, y)))
                if key not in seen:
                    seen.add(key)
                    alerts.append({**rule, "drugs": [x, y], "kind": "drug-drug",
                                   "on_current": x in current and y in current})
    for rule in db["disease"]:
        if rule["finding"] not in finding_keys:
            continue
        for d, c in everything.items():
            if c in rule["class"]:
                alerts.append({**rule, "drugs": [d], "kind": "drug-disease", "on_current": d in current})
    return sorted(alerts, key=lambda a: (SEVERITY_RANK[a["severity"]], not a["on_current"]))


def merge_alerts(alerts):
    """One alert per rule, listing every drug it involves (current medicines first)."""
    by = {}
    for a in alerts:
        m = by.setdefault(a["title"], {**a, "drugs": []})
        m["on_current"] = m["on_current"] or a["on_current"]
        m["drugs"] += [d for d in a["drugs"] if d not in m["drugs"]]
    return list(by.values())


# ------------------------------------------------------------------ therapy options
def options_for(fnd, meds) -> list:
    """Guideline therapy classes for the findings, in finding order then guideline tier. No sponsorship input."""
    cat = catalogue()
    tiers = cat["tiers"]
    keys = [f["key"] for f in fnd]
    picked = {}
    for k in keys:
        for c in sorted((c for c in cat["classes"] if k in c["findings"]), key=lambda c: tiers[c["tier"]]["rank"]):
            if c["key"] not in picked:
                picked[c["key"]] = {**c, "for": [k]}
            elif k not in picked[c["key"]]["for"]:
                picked[c["key"]]["for"].append(k)
    out = []
    for c in picked.values():
        covers = CLASS_COVERS.get(c.get("med_class"), {c.get("med_class")})
        on = [m for m in meds if (med_class(m) is not None and med_class(m) in covers) or m in c["drugs"]]
        alerts = merge_alerts([a for a in check_interactions(meds, c["drugs"], keys) if set(a["drugs"]) & set(c["drugs"])])
        out.append({**c, "tier_label": tiers[c["tier"]]["label"], "already_on": on, "alerts": alerts,
                    "for_labels": [cat["findings"][k]["label"] for k in c["for"]]})
    return out


def sponsored_for(fnd) -> list:
    keys = {f["key"] for f in fnd}
    return [s for s in SPONSORED if s["finding"] in keys]


# ------------------------------------------------------------------ targets
def targets():
    return _load("targets.json")["targets"]


def _drug_name(n):
    prev = None
    while prev != n:
        prev, n = n, SALT.sub("", n)
    return n


@lru_cache(maxsize=None)
def plddt(uniprot):
    """Mean AlphaFold confidence and the share of residues above 70, from the model's B-factors."""
    path = os.path.join(STRUCT_DIR, f"AF-{uniprot}.pdb")
    if not os.path.exists(path):
        return None
    vals = [float(ln[60:66]) for ln in open(path) if ln.startswith("ATOM") and ln[12:16].strip() == "CA"]
    if not vals:
        return None
    return {"mean": sum(vals) / len(vals), "confident": sum(v >= 70 for v in vals) / len(vals), "residues": len(vals)}


def target(gene):
    t = next((t for t in targets() if t["gene"] == gene), None)
    if not t:
        return None
    raw = (_load("chembl_drugs.json").get(gene) or {})
    drugs, seen = [], set()
    for d in t.get("extra_drugs", []) + raw.get("drugs", []):          # curated additions carry their provenance
        name = _drug_name(d["name"])
        if name in seen or name in HIDE.get(gene, set()):
            continue
        seen.add(name)
        drugs.append({**d, "name": name, "phase": PHASE.get(int(d["max_phase"]), "Early or unknown phase")})
    drugs.sort(key=lambda d: (-d["max_phase"], d["name"]))
    cx = t.get("complex")
    return {**t, "drugs": drugs, "chembl_id": raw.get("target_chembl_id"), "plddt": plddt(t["uniprot"]),
            "has_model": os.path.exists(os.path.join(STRUCT_DIR, f"AF-{t['uniprot']}.pdb")),
            "has_complex": bool(cx and os.path.exists(os.path.join(STRUCT_DIR, f"{cx['pdb']}.pdb")))}


def targets_for(fnd):
    keys = [f["key"] for f in fnd]
    return [target(t["gene"]) for t in targets() if set(t["findings"]) & set(keys)]


# ------------------------------------------------------------------ trials
def _years(s):
    m = re.match(r"(\d+)\s*Year", s or "")
    return int(m.group(1)) if m else None


def trial_fit(trial, age, sex):
    """Obvious pre-checks only (age, sex). Everything else is for the clinician and the site."""
    lo, hi = _years(trial.get("min_age")), _years(trial.get("max_age"))
    if age is not None and ((lo and age < lo) or (hi and age > hi)):
        return "outside", "Age outside the trial range"
    tsex = (trial.get("sex") or "ALL").upper()
    if sex and tsex in ("MALE", "FEMALE") and tsex != sex.upper():
        return "outside", "Trial enrols one sex only"
    return ("possible", "Age and sex fit; confirm full eligibility") if age is not None else ("check", "Add age to pre-check")


def trial_topics(fnd):
    from .external import TRIAL_QUERIES
    return [f for f in fnd if f["key"] in TRIAL_QUERIES]


def trial_matches(fnd, age=None, sex=None, zip_code=None, per_topic=4):
    from concurrent.futures import ThreadPoolExecutor

    from . import external
    topics = trial_topics(fnd)
    with ThreadPoolExecutor(max_workers=max(1, len(topics))) as pool:        # topics fetched in parallel
        fetched = list(pool.map(lambda f: external.trials(f["key"], zip_code), topics))
    out = []
    for f, got in zip(topics, fetched):
        items = []
        for t in got["items"]:
            fit, why = trial_fit(t, age, sex)
            items.append({**t, "fit": fit, "fit_text": why})
        items.sort(key=lambda t: {"possible": 0, "check": 1, "outside": 2}[t["fit"]])
        out.append({"finding": f, "source": got["source"], "trials": items[:per_topic]})
    return out


# ------------------------------------------------------------------ med-info context (de-identified)
def age_band(age):
    if age is None:
        return "adult"
    a = int(age)
    return "adult under 40" if a < 40 else f"adult in their {a // 10 * 10}s" if a < 90 else "adult 90 or older"


def deidentified_context(age, fnd) -> str:
    """What a manufacturer's medical-information desk may see: an age band and the finding labels. Nothing else."""
    labels = [f["label"].lower() for f in fnd if f["key"] != "metabolism"]
    return f"{age_band(age).capitalize()} with diabetes" + (f"; screening findings: {', '.join(labels)}." if labels else ".")


# ------------------------------------------------------------------ CMS care gap
def care_gap(case_status, signed, specialist_answered):
    """Where this encounter stands against the diabetic eye exam measures (CMS131 / HEDIS EED)."""
    if specialist_answered:
        return "closed", "Specialist read recorded: the diabetic eye exam gap can be closed for this year."
    if signed:
        return "pending", "Interpreted by the referring clinician. Record the eye care professional's read to close the gap."
    if case_status in ("Unable to assess",):
        return "open", "No gradable photos: the gap stays open. Recapture or refer for a dilated exam."
    return "open", "Screening captured; the gap closes once an eye care professional reads it."
