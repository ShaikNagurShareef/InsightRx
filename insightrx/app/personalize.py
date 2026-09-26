"""
Personalised therapeutics: one patient's eye-derived phenotype -> ranked targets, therapy options, interactions, trials.

Target priority index (0-100), transparent by design:
    priority = 100 x phenotype link x (0.5 x actionability + 0.3 x structural tractability + 0.2 x clinical fit)
  phenotype link        strongest finding linked to the target (validated retina 1.0, research 0.85, known 0.75,
                        exploratory 0.55, diabetes baseline 0.35)
  actionability         strongest guideline class acting on the target (first-line or specialist-administered 1.0,
                        add-on 0.85, trial evidence 0.7, investigational in the US 0.4); with no guideline class,
                        ChEMBL stage (approved anywhere 0.5, clinical-stage 0.4, none 0.2)
  structural tractability  AlphaFold confidence at the drug-contact site (or whole-model confidence) / 100
  clinical fit          1.0 if no interaction alert blocks the class, 0.5 if a major alert applies, 0.8 if moderate
It ranks where to look for this patient; it is not a treatment recommendation.
"""
import json
import os
from functools import lru_cache

from . import therapeutics as tx

TIER_WEIGHT = {"first_line": 1.0, "specialist": 1.0, "add_on": 0.85, "trial": 0.7, "investigational": 0.4}
STRENGTH_WEIGHT = {"Validated retinal model": 1.0, "Research signal": 0.85, "Retinal research signal": 0.85,
                   "Known condition": 0.75, "Exploratory signal": 0.55, "Applies to every screened patient": 0.35}


@lru_cache(maxsize=None)
def structure_insights():
    path = os.path.join(tx.DATA, "structure_insights.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def structural_confidence(gene):
    si = structure_insights().get(gene) or {}
    pocket = si.get("pocket") or {}
    if pocket.get("mean_plddt"):
        return pocket["mean_plddt"] / 100, f"drug-contact site pLDDT {pocket['mean_plddt']:.0f} (PDB {pocket['pdb']})"
    vals = si.get("plddt") or []
    if vals:
        conf = sum(v >= 70 for v in vals) / len(vals)
        return conf, f"{conf * 100:.0f}% of residues confidently predicted (no drug complex in this set)"
    return 0.5, "no structure analysis"


def _actionability(t, classes):
    """Guideline strength of the best class acting on the target; ChEMBL stage only when no class exists."""
    approved = sum(d["phase"] == "Approved" for d in t["drugs"])
    pipeline = len(t["drugs"]) - approved
    if classes:
        return max(TIER_WEIGHT[c["tier"]] for c in classes), approved, pipeline
    return (0.5 if approved else 0.4 if pipeline else 0.2), approved, pipeline


def target_priorities(fnd, meds, options=None):
    """Targets this patient's findings point to, ranked by the priority index, each with its rationale."""
    options = options if options is not None else tx.options_for(fnd, meds)
    strength = {f["key"]: (STRENGTH_WEIGHT.get(f["strength"], 0.5), f) for f in fnd}
    out = []
    for t in tx.targets_for(fnd):
        linked = [strength[k] for k in t["findings"] if k in strength]
        link, best = max(linked, key=lambda x: x[0])
        classes = [o for o in options if t["gene"] in o["targets"]]
        act, approved, pipeline = _actionability(t, classes)
        struct, struct_text = structural_confidence(t["gene"])
        sev = {a["severity"] for o in classes for a in o["alerts"]}
        fit = 0.5 if "major" in sev else 0.8 if "moderate" in sev else 1.0
        engaged = sorted({m for o in classes for m in o["already_on"]} |
                         {m for m in meds if m in {d["name"] for d in t["drugs"]}})
        score = round(100 * link * (0.5 * act + 0.3 * struct + 0.2 * fit))
        out.append({"gene": t["gene"], "name": t["name"], "organ": t["organ"], "priority": score,
                    "link": link, "finding": best["label"], "strength": best["strength"],
                    "findings": [strength[k][1]["label"] for k in t["findings"] if k in strength],
                    "approved": approved, "pipeline": pipeline, "struct": struct, "struct_text": struct_text,
                    "fit": fit, "classes": [c["label"] for c in classes], "engaged": engaged,
                    "top_drugs": [d["name"] for d in t["drugs"] if d["phase"] == "Approved"][:4],
                    "tier": max(classes, key=lambda c: TIER_WEIGHT[c["tier"]])["tier_label"] if classes else "No guideline class",
                    "why": _why(t, best, classes, approved, pipeline, struct_text, engaged, fit)})
    return sorted(out, key=lambda r: -r["priority"])


def _why(t, best, classes, approved, pipeline, struct_text, engaged, fit):
    bits = [f"{best['label']} ({best['strength'].lower()})"]
    if classes:
        top = max(classes, key=lambda c: TIER_WEIGHT[c["tier"]])
        bits.append(f"{top['label']}: {top['tier_label'].lower()}")
    else:
        bits.append("no guideline therapy class")
    bits.append(f"{approved} approved drug{'s' if approved != 1 else ''}" if approved else
                f"clinical-stage only ({pipeline} in development)" if pipeline else "no drugs recorded")
    bits.append(struct_text)
    if engaged:
        bits.append("already engaged by " + ", ".join(engaged))
    if fit < 1:
        bits.append("interaction alert on this class")
    return "; ".join(bits)


def bundle(res, snapshot, meds, age=None, sex=None, with_trials=True, per_topic=2):
    """Everything the therapy views and reports need for one patient."""
    fnd = tx.findings(res, snapshot)
    options = tx.options_for(fnd, meds)
    keys = [f["key"] for f in fnd]
    return {"fnd": fnd, "options": options, "meds": meds,
            "current_alerts": tx.merge_alerts(tx.check_interactions(meds, (), keys)),
            "sponsored": tx.sponsored_for(fnd), "priorities": target_priorities(fnd, meds, options),
            "trials": tx.trial_matches(fnd, age, sex, per_topic=per_topic) if with_trials else []}
