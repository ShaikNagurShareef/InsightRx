"""
PDF reports (reportlab): a personalised therapy and target report for one patient, a target dossier for drug
discovery, and a portfolio view of all targets. Every dynamic string is XML-escaped before it reaches a Paragraph.
"""
import io
import os
import re
import textwrap
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import BaseDocTemplate, Frame, Image, PageTemplate, Paragraph, Spacer, Table, TableStyle

from . import report_charts as ch
from . import therapeutics as tx
from .personalize import structure_insights

W, H = LETTER
MARGIN = 0.7 * inch
BODY_W = W - 2 * MARGIN
DISCLAIMER = ("Insight Rx research prototype. Decision support for clinician review: not a diagnosis, prescription or "
              "treatment recommendation. Verify every drug decision against the full label and current guidelines.")

_base = dict(fontName="Helvetica", fontSize=9, leading=12.5, textColor=ch.INK, alignment=TA_LEFT)
S = {
    "title": ParagraphStyle("title", **{**_base, "fontName": "Helvetica-Bold", "fontSize": 19, "leading": 23}),
    "sub": ParagraphStyle("sub", **{**_base, "fontSize": 10, "leading": 14, "textColor": ch.INK3}),
    "h2": ParagraphStyle("h2", **{**_base, "fontName": "Helvetica-Bold", "fontSize": 12.5, "leading": 16,
                                  "spaceBefore": 12, "spaceAfter": 5, "textColor": ch.MOL}),
    "h3": ParagraphStyle("h3", **{**_base, "fontName": "Helvetica-Bold", "fontSize": 10, "leading": 13, "spaceBefore": 6,
                                  "spaceAfter": 2}),
    "body": ParagraphStyle("body", **_base),
    "small": ParagraphStyle("small", **{**_base, "fontSize": 8, "leading": 10.5}),
    "muted": ParagraphStyle("muted", **{**_base, "fontSize": 7.5, "leading": 10, "textColor": ch.INK3}),
    "cell": ParagraphStyle("cell", **{**_base, "fontSize": 7.8, "leading": 10}),
    "cellb": ParagraphStyle("cellb", **{**_base, "fontName": "Helvetica-Bold", "fontSize": 7.8, "leading": 10}),
    "bullet": ParagraphStyle("bullet", **{**_base, "leftIndent": 10, "bulletIndent": 0, "spaceAfter": 2}),
}
SEV = {"major": ch.SIGNAL, "moderate": ch.CAUTION, "minor": ch.INK3}


def esc(v):
    return escape("" if v is None else str(v))


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, widths, header=True, zebra=True, style_extra=()):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("LEFTPADDING", (0, 0), (-1, -1), 4),
          ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("LINEBELOW", (0, 0), (-1, -1), 0.3, ch.LINE)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f5")), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ch.INK3)]
    if zebra:
        st += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafbfc")])]
    t.setStyle(TableStyle(st + list(style_extra)))
    return t


class M(str):
    """Trusted Paragraph markup built from already-escaped parts. Plain strings are always escaped."""


def cells(row, bold_first=False):
    return [c if hasattr(c, "wrapOn") else P(c if isinstance(c, M) else esc(c), "cellb" if bold_first and i == 0 else "cell")
            for i, c in enumerate(row)]


def callout(text, color=ch.MOL):
    t = Table([[P(text, "small")]], colWidths=[BODY_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.Color(color.red, color.green, color.blue, alpha=0.08)),
                           ("LINEBEFORE", (0, 0), (0, -1), 2.5, color), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def _render(title, story, kind):
    buf = io.BytesIO()
    stamp = datetime.now().strftime("%b %d, %Y %H:%M")

    def page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(ch.FUNDUS)
        canvas.circle(MARGIN + 5, H - 0.45 * inch, 5, fill=1, stroke=0)
        canvas.setFillColor(ch.INK)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(MARGIN + 14, H - 0.48 * inch, "Insight Rx")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(ch.INK3)
        canvas.drawRightString(W - MARGIN, H - 0.48 * inch, f"{kind} · {title} · generated {stamp}")
        canvas.setStrokeColor(ch.LINE)
        canvas.line(MARGIN, H - 0.56 * inch, W - MARGIN, H - 0.56 * inch)
        canvas.setFont("Helvetica", 6.8)
        for k, line in enumerate(textwrap.wrap(DISCLAIMER, 140)[:2]):
            canvas.drawString(MARGIN, (0.42 - 0.12 * k) * inch, line)
        canvas.drawRightString(W - MARGIN, 0.42 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(buf, pagesize=LETTER, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=0.8 * inch,
                          bottomMargin=0.7 * inch, title=f"Insight Rx {kind}: {title}", author="Insight Rx")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(MARGIN, 0.7 * inch, BODY_W, H - 1.5 * inch, id="f")],
                                       onPage=page)])
    doc.build(story)
    return buf.getvalue()


# ================================================================== shared sections
def therapy_section(b):
    story = [P("Therapy options to consider", "h2"),
             P("Guideline classes for this patient's findings, ranked by guideline strength only (sponsorship never "
               "enters the ranking). Current medicines: " + (esc(", ".join(b["meds"])) or "none recorded, so "
                                                              "interactions could not be checked") + ".", "small")]
    rows = [cells(["Therapy class", "Why (findings)", "Evidence tier", "Example drugs", "Already on", "Source"])]
    for o in b["options"]:
        rows.append(cells([o["label"], ", ".join(o["for_labels"]), o["tier_label"], ", ".join(o["drugs"]) or "-",
                           ", ".join(o["already_on"]) or "-", o["source"]], bold_first=True))
    story.append(table(rows, [1.25 * inch, 1.2 * inch, 0.85 * inch, 1.3 * inch, 0.75 * inch, 1.45 * inch]))
    alerts = {a["title"]: a for o in b["options"] for a in o["alerts"]}
    for a in b["current_alerts"]:
        alerts.setdefault(a["title"], a)
    if alerts:
        story += [P("Interaction and drug-disease alerts", "h3")]
        rows = [cells(["Severity", "Alert", "Involves", "Detail and source"])]
        for a in sorted(alerts.values(), key=lambda a: tx.SEVERITY_RANK[a["severity"]]):
            sev = P(f'<font color="{SEV[a["severity"]].hexval()}"><b>{esc(a["severity"].capitalize())}</b></font>', "cell")
            rows.append([sev] + cells([a["title"], " + ".join(a["drugs"]) + (" (already prescribed)" if a["on_current"] else " (if started)"),
                                       M(f'{esc(a["detail"])} <font color="#66788a">{esc(a["source"])}</font>')]))
        story.append(table(rows, [0.65 * inch, 1.35 * inch, 1.3 * inch, 3.5 * inch]))
    return story


def priority_section(b, top_n=8):
    pr = b["priorities"][:top_n]
    story = [P("Personalised target priorities", "h2"),
             P("Where this patient's phenotype points at the molecular level. Priority = 100 x phenotype link x "
               "(0.5 guideline actionability + 0.3 structural tractability at the drug site + 0.2 clinical fit). "
               "Actionability comes from US guideline strength, so drugs approved only abroad do not lift a target. "
               "It ranks where to look; it is not a treatment recommendation.", "small"), Spacer(1, 4)]
    if not pr:
        return story + [P("No targets linked to these findings.", "small")]
    top = pr[0]
    story.append(callout(f"<b>Top target for this patient: {esc(top['gene'])}</b> ({esc(top['name'])}), priority "
                         f"{top['priority']}. {esc(top['why'])}."))
    story.append(Spacer(1, 6))
    rows = [cells(["Target", "Priority", "Linked findings", "Drugs", "Structure", "Rationale"])]
    for r in pr:
        rows.append([P(f"<b>{esc(r['gene'])}</b><br/>{esc(r['name'])}", "cell"),
                     Table([[ch.priority_bar(r["priority"], 44), P(f"<b>{r['priority']}</b>", "cell")]],
                           colWidths=[48, 22], style=[("LEFTPADDING", (0, 0), (-1, -1), 0)]),
                     P(esc("; ".join(r["findings"])), "cell"),
                     P(f"{r['approved']} approved<br/>{r['pipeline']} pipeline", "cell"),
                     P(esc(r["struct_text"]), "cell"),
                     P(esc(("Engaged by " + ", ".join(r["engaged"]) + ". ") if r["engaged"] else "") +
                       esc(("Classes: " + ", ".join(r["classes"])) if r["classes"] else "No guideline class yet"), "cell")])
    story.append(table(rows, [1.25 * inch, 0.85 * inch, 1.25 * inch, 0.75 * inch, 1.3 * inch, 1.4 * inch]))
    return story


def trials_section(groups, fit=True):
    story = [P("Recruiting clinical trials", "h2"),
             P("ClinicalTrials.gov, searched by finding near the clinic; no patient data is sent. The site confirms "
               "full eligibility.", "small")]
    rows = [cells(["Finding", "Trial", "Phase / sponsor", "Pre-check" if fit else "Sites"])]
    for g in groups:
        for t in g["trials"]:
            rows.append([P(esc(g["finding"]["label"]), "cell"),
                         P(f'<b>{esc(t["title"])}</b><br/><font color="#1d5fa8">{esc(t["nct"])}</font> {esc(t["site"])}', "cell"),
                         P(esc(f'{t["phase"].replace("PHASE", "Phase ").replace("_", " ")} · {t["sponsor"]}'), "cell"),
                         P(esc(t.get("fit_text", "") if fit else f'{t["sites"]} site(s)'), "cell")])
    if len(rows) == 1:
        return story + [P("No recruiting trials found.", "small")]
    story.append(table(rows, [1.2 * inch, 3.3 * inch, 1.3 * inch, 1.0 * inch]))
    return story


# ================================================================== patient report
def patient_report(p):
    """p: ref, subtitle, meta (list of (k, v)), res, eyes (list of rows), systemic (dict), images [(jpeg, caption)],
    organs (snapshot), bundle, cms, gap, gap_text, clinician."""
    b, res = p["bundle"], p["res"] or {}
    story = [P(f"Personalised therapy and target report", "sub"), P(esc(p["ref"]), "title"), P(esc(p["subtitle"]), "sub"),
             Spacer(1, 8)]
    story.append(table([cells([k, v]) for k, v in p["meta"]], [1.4 * inch, BODY_W - 1.4 * inch], header=False,
                       zebra=False))
    major = [a for a in b["current_alerts"] if a["severity"] == "major"]
    n_trials = sum(len(g["trials"]) for g in b["trials"])
    top = b["priorities"][0] if b["priorities"] else None
    summary = [f"<b>Retina:</b> {esc(res.get('overall') or 'no current analysis')}.",
               "<b>Findings:</b> " + esc(", ".join(f["label"] for f in b["fnd"] if f["key"] != "metabolism") or "none beyond diabetes") + ".",
               f"<b>Therapy options:</b> {len(b['options'])} guideline classes; "
               f"<b>{len(major)}</b> major alert(s) on current medicines.",
               (f"<b>Top target:</b> {esc(top['gene'])} (priority {top['priority']}): {esc(top['why'])}." if top else
                "<b>Top target:</b> none."),
               f"<b>Trials:</b> {n_trials} recruiting trial(s) matched by finding.",
               f"<b>CMS diabetic eye exam gap:</b> {esc(p['gap'])}. {esc(p['gap_text'])}"]
    story += [P("Summary", "h2")] + bullets(summary)

    story += [P("Retina", "h2")]
    if p["eyes"]:
        story.append(table([cells(["Eye", "Result", "Referable DR score", "Highest ICDR grade", "Macular edema signal"])] +
                           [cells(r, bold_first=True) for r in p["eyes"]],
                           [0.8 * inch, 1.8 * inch, 1.3 * inch, 1.3 * inch, 1.6 * inch]))
    if p["images"]:
        story.append(Spacer(1, 6))
        imgs = [[Image(io.BytesIO(j), width=1.55 * inch, height=1.55 * inch), P(esc(c), "muted")] for j, c in p["images"][:4]]
        grid = Table([[i[0] for i in imgs], [i[1] for i in imgs]], colWidths=[1.7 * inch] * len(imgs))
        grid.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(grid)

    if p["systemic"]:
        story += [P("Whole-body signals", "h2"),
                  P("Research association models from the retina plus patient details; history recorded by the clinician "
                    "takes precedence. Red bar: score at or above the frozen threshold (black tick).", "small")]
        rows = [cells(["Condition", "Output", "Score vs threshold", "Reliability"])]
        for s in p["systemic"].values():
            g = ch.gauge(s["score"], s["threshold"], 110) if "score" in s else P("-", "cell")
            rows.append(cells([s["label"], s.get("status", ""), g,
                               s.get("reliability_text") or s.get("reason", "")], bold_first=True))
        story.append(table(rows, [1.5 * inch, 1.2 * inch, 1.35 * inch, 2.75 * inch]))

    story += therapy_section(b)
    story += priority_section(b)
    story += trials_section(b["trials"])
    story += [P("CMS quality and billing", "h2")]
    rows = [cells(["Measure / code", "What it is", "Fit for this encounter"])]
    rows += [cells([m["id"], m["name"], m["how"]], True) for m in p["cms"]["measures"]]
    rows += [cells([f'CPT {c["code"]}', c["name"], c["fit"]], True) for c in p["cms"]["codes"]]
    story.append(table(rows, [1.5 * inch, 2.4 * inch, 2.9 * inch]))
    story += [P("Methods, sources and limits", "h2")] + bullets([
        "Retinal models: DINOv2-L multi-task ensemble trained on mBRSET (portable camera, dilated); other cameras are a domain shift.",
        "Therapy classes and interactions are curated from ADA, AAO, KDIGO guidelines and FDA labels, cited per row; not a complete interaction checker.",
        "Targets: UniProt; structures: AlphaFold DB (CC-BY 4.0) and RCSB PDB; drugs: ChEMBL mechanisms; trials: ClinicalTrials.gov.",
        f"Prepared for {esc(p['clinician'])}. Patient identity in this workspace is synthetic."])
    return _render(p["ref"], story, "Patient report")


# ================================================================== target dossier
def clean_function(text):
    """UniProt function text without evidence tags, cut at the last full sentence."""
    text = re.sub(r"\s*\((?:PubMed|By similarity|Probable|Ref\.)[^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[: text.rfind(".") + 1] if "." in text else text


def target_insights(t, si, demand, panel_n, n_trials):
    """Rule-based discovery insights, each traceable to a number on the page."""
    out = []
    pocket = (si or {}).get("pocket")
    plddt = (si or {}).get("plddt") or []
    mean = sum(plddt) / len(plddt) if plddt else None
    if pocket and pocket.get("mean_plddt"):
        m = pocket["mean_plddt"]
        verdict = ("high: the AlphaFold model reproduces the site well enough for structure-based screening and docking"
                   if m >= 90 else "moderate: use the AlphaFold model with care and validate against the experimental structure"
                   if m >= 70 else "low: rely on the experimental structure for design")
        out.append(f"<b>Druggable site mapped.</b> {len(pocket['residues'])} residues lie within {pocket['cutoff_A']} A of the "
                   f"{esc(pocket['drug'])} in PDB {esc(pocket['pdb'])}; AlphaFold confidence there is {m:.0f} ({verdict}).")
    elif mean is not None:
        out.append(f"<b>No drug-bound structure in this set.</b> The AlphaFold model has mean pLDDT {mean:.0f}; pocket "
                   f"detection on the confident regions is the next step.")
    if (si or {}).get("disordered"):
        ranges = ", ".join(f"{a}-{b}" for a, b in si["disordered"][:5])
        out.append(f"<b>Flexible regions</b> (pLDDT below 50) at residues {ranges}: likely disordered or unstructured; "
                   "exclude them from pocket searches and expect them to be absent in crystal structures.")
    approved = [d for d in t["drugs"] if d["phase"] == "Approved"]
    pipeline = [d for d in t["drugs"] if d["phase"] != "Approved"]
    mods = {}
    for d in t["drugs"]:
        mods[d.get("type") or "unknown"] = mods.get(d.get("type") or "unknown", 0) + 1
    mix = ", ".join(f"{k} {v}" for k, v in sorted(mods.items(), key=lambda x: -x[1]))
    if len(approved) >= 5:
        out.append(f"<b>Validated and crowded mechanism.</b> {len(approved)} approved drugs ({mix}); a new entrant needs "
                   "differentiation in selectivity, dosing interval, delivery or a defined patient subgroup.")
    elif approved:
        out.append(f"<b>Validated mechanism with room.</b> {len(approved)} approved and {len(pipeline)} clinical-stage drugs ({mix}).")
    elif pipeline:
        out.append(f"<b>Clinical-stage only.</b> {len(pipeline)} drugs in development and none approved: an unmet-need target "
                   "with higher risk and higher reward.")
    else:
        out.append("<b>No drugs recorded in ChEMBL</b> against this exact protein: early discovery territory.")
    if panel_n:
        out.append(f"<b>Patient demand signal.</b> {demand} of {panel_n} screened patients ({demand / panel_n * 100:.0f}%) "
                   "carry a finding linked to this target: a recruitment and market signal detected non-invasively.")
    out.append(f"<b>Clinical activity.</b> {n_trials} recruiting trial(s) near the clinic for the linked findings.")
    return out


def target_report(t, demand, panel_n, trials, classes, patient_priority=None):
    si = structure_insights().get(t["gene"]) or {}
    pocket = (si.get("pocket") or {}).get("residues") or []
    n_trials = sum(len(g["trials"]) for g in trials)
    story = [P("Target dossier", "sub"), P(f"{esc(t['gene'])}: {esc(t['name'])}", "title"),
             P(f"UniProt {esc(t['uniprot'])} · {si.get('length', '?')} residues · ChEMBL {esc(t.get('chembl_id') or '-')} · "
               f"linked findings: {esc(', '.join(tx.catalogue()['findings'][k]['label'] for k in t['findings']))}", "sub"),
             Spacer(1, 6)]
    if patient_priority:
        story.append(callout(f"<b>For patient {esc(patient_priority['ref'])}:</b> priority {patient_priority['priority']}. "
                             f"{esc(patient_priority['why'])}.", ch.FUNDUS))
    story += [P("Key insights", "h2")] + bullets(target_insights(t, si, demand, panel_n, n_trials))

    story += [P("Structure", "h2")]
    snap = ch.structure_snapshot(os.path.join(tx.STRUCT_DIR, f"AF-{t['uniprot']}.pdb"), pocket, 2.6 * inch)
    stats = [["Mean pLDDT", f"{t['plddt']['mean']:.1f}" if t.get("plddt") else "-"],
             ["Confident residues (pLDDT 70+)", f"{t['plddt']['confident'] * 100:.0f}%" if t.get("plddt") else "-"],
             ["Disordered regions", ", ".join(f"{a}-{b}" for a, b in si.get("disordered", [])) or "none"],
             ["Drug-contact residues", str(len(pocket)) + (f" (PDB {si['pocket']['pdb']})" if pocket else "")],
             ["Contact-site pLDDT", f"{si['pocket']['mean_plddt']:.1f}" if pocket and si["pocket"].get("mean_plddt") else "-"]]
    side = [table([cells(r, True) for r in stats], [1.75 * inch, 1.6 * inch], header=False),
            Spacer(1, 6), P("AlphaFold model projected on its two principal axes, coloured by confidence; orange dots mark "
                            "residues touching the drug in the experimental complex.", "muted")]
    story.append(Table([[snap, side]], colWidths=[2.8 * inch, BODY_W - 2.8 * inch],
                       style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [Spacer(1, 6), ch.plddt_legend(), ch.plddt_track(si.get("plddt") or [], si.get("features") or [], pocket, BODY_W)]
    if pocket:
        story.append(P("<b>Drug-contact residues (UniProt numbering):</b> " + esc(", ".join(si["pocket"].get("aa", []))),
                       "small"))
    if si.get("function"):
        story += [P("Function (UniProt)", "h3"), P(esc(clean_function(si["function"])), "small")]
    story += [P("Why the eye points here", "h3"), P(esc(t["role"]) + " " + esc(t["eye_link"]), "small")]

    story += [P("Drug landscape", "h2")]
    rows = [cells(["Drug", "Stage", "Action", "Modality", "ChEMBL"])]
    rows += [cells([d["name"], d["phase"], d["action"] or "-", d["type"] or "-", d["chembl_id"]], True) for d in t["drugs"]]
    story.append(Table([[ch.drug_landscape(t["drugs"], 2.9 * inch),
                         table(rows, [1.0 * inch, 0.62 * inch, 0.62 * inch, 0.76 * inch, 1.1 * inch]) if t["drugs"]
                         else P("No mechanism records.", "small")]],
                       colWidths=[3.0 * inch, BODY_W - 3.0 * inch],
                       style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    notes = sorted({d["via"] for d in t["drugs"] if d.get("via")})
    if notes:
        story.append(P("Note: " + esc("; ".join(notes)) + ".", "muted"))
    if classes:
        story += [P("In the Insight Rx therapy guide", "h3")]
        story += bullets([f"<b>{esc(c['label'])}</b> ({esc(tx.catalogue()['tiers'][c['tier']]['label'])}): "
                          f"{esc(c['summary'])} <font color='#66788a'>{esc(c['source'])}</font>" for c in classes])
    story += trials_section(trials, fit=False)
    story += [P("Sources", "h2"), P("UniProt; AlphaFold Protein Structure Database (DeepMind/EMBL-EBI, CC-BY 4.0); RCSB PDB; "
                                   "ChEMBL mechanisms (EMBL-EBI); ClinicalTrials.gov. Contacts computed at 4.5 A between "
                                   "heavy atoms of the drug and the target, mapped to UniProt numbering by sequence alignment.",
                                   "small")]
    return _render(t["gene"], story, "Target dossier")


# ================================================================== portfolio
def portfolio_rows(counts, panel_n):
    rows = []
    for g in tx.targets():
        t = tx.target(g["gene"])
        si = structure_insights().get(g["gene"]) or {}
        demand = max([counts.get(k, 0) for k in t["findings"]] + [0])
        approved = sum(d["phase"] == "Approved" for d in t["drugs"])
        pipeline = len(t["drugs"]) - approved
        pk = (si.get("pocket") or {}).get("mean_plddt")
        struct = (pk or (t["plddt"]["mean"] if t.get("plddt") else 50)) / 100
        share = demand / panel_n if panel_n else 0
        novelty = 1.2 if not approved and pipeline else 1.0 if approved < 5 else 0.7
        rows.append({"gene": g["gene"], "name": t["name"], "organ": t["organ"], "demand": demand, "demand_share": share,
                     "approved": approved, "pipeline": pipeline, "struct": struct, "pocket": pk,
                     "opportunity": round(100 * share * (0.5 + 0.5 * struct) * novelty),
                     "label": ("Unmet need" if not approved and pipeline else "Early discovery" if not approved else
                               "Crowded" if approved >= 5 else "Validated, room to differentiate")})
    return sorted(rows, key=lambda r: -r["opportunity"])


def portfolio_report(rows, panel_n):
    story = [P("Portfolio view", "sub"), P("Eye-derived demand across disease targets", "title"),
             P(f"{panel_n} screened patients · {len(rows)} targets · structures from AlphaFold and the PDB, drugs from ChEMBL",
               "sub"), Spacer(1, 6)]
    if rows:
        top = rows[0]
        story.append(callout(f"<b>Highest opportunity: {esc(top['gene'])}</b> ({esc(top['name'])}). {top['demand']} of "
                             f"{panel_n} screened patients carry a linked finding; {top['approved']} approved and "
                             f"{top['pipeline']} pipeline drugs; label: {esc(top['label'])}."))
    story += [P("Opportunity matrix", "h2"),
              P("Demand is detected non-invasively from retinal screening; saturation is the number of approved drugs; "
                "bubble size is the clinical pipeline. Quadrants frame discussion and are not valuations.", "small"),
              ch.opportunity_matrix(rows, BODY_W, 270)]
    story += [P("Ranked targets", "h2"),
              P("Opportunity = 100 x demand share x (0.5 + 0.5 x structural confidence) x novelty (1.2 clinical-stage only, "
                "1.0 fewer than 5 approved, 0.7 crowded).", "small")]
    tr = [cells(["Target", "Organ", "Patients", "Approved", "Pipeline", "Structure", "Opportunity", "Label"])]
    for r in rows:
        tr.append(cells([M(f"<b>{esc(r['gene'])}</b><br/>{esc(r['name'])}"), r["organ"], r["demand"], r["approved"],
                         r["pipeline"], (f"pocket pLDDT {r['pocket']:.0f}" if r["pocket"] else f"model {r['struct'] * 100:.0f}"),
                         Table([[ch.priority_bar(min(100, r["opportunity"]), 40), P(str(r["opportunity"]), "cell")]],
                               colWidths=[44, 20], style=[("LEFTPADDING", (0, 0), (-1, -1), 0)]), r["label"]]))
    story.append(table(tr, [1.55 * inch, 0.7 * inch, 0.6 * inch, 0.6 * inch, 0.6 * inch, 0.95 * inch, 0.8 * inch, 1.0 * inch]))
    return _render("All targets", story, "Portfolio report")

