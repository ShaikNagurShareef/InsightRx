"""
Vector charts for the PDF reports (reportlab graphics): structure snapshot, AlphaFold confidence track, drug landscape,
opportunity matrix and score-vs-threshold gauges. Pure functions: data in, Drawing out.
"""
import os

import numpy as np
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors

INK = colors.HexColor("#16232e")
INK3 = colors.HexColor("#66788a")
LINE = colors.HexColor("#dce3e8")
MOL = colors.HexColor("#0f6b6b")
FUNDUS = colors.HexColor("#e0782b")
SIGNAL = colors.HexColor("#b3261e")
CLEAR = colors.HexColor("#1f7a4d")
CAUTION = colors.HexColor("#b07400")
RESEARCH = colors.HexColor("#5a4a9e")
PLDDT_BANDS = [(90, colors.HexColor("#0053d6"), "Very high (90+)"), (70, colors.HexColor("#65cbf3"), "Confident (70-90)"),
               (50, colors.HexColor("#ffdb13"), "Low (50-70)"), (0, colors.HexColor("#ff7d45"), "Very low (<50)")]
FEATURE_COLORS = {"Domain": MOL, "Transmembrane": colors.HexColor("#8a9aa9"), "Signal": RESEARCH,
                  "Topological domain": None, "Binding site": FUNDUS, "Active site": SIGNAL}


def plddt_color(v):
    return next(c for lo, c, _ in PLDDT_BANDS if v >= lo)


def _label(d, x, y, text, size=7, color=INK3, anchor="start", bold=False):
    d.add(String(x, y, text, fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size, fillColor=color,
                 textAnchor=anchor))


def plddt_legend(width=460):
    d = Drawing(width, 14)
    x = 0
    for _, c, name in PLDDT_BANDS:
        d.add(Rect(x, 3, 9, 9, fillColor=c, strokeColor=None))
        _label(d, x + 12, 4, name, 7.5)
        x += 105
    return d


# ------------------------------------------------------------------ structure snapshot (2D projection of the CA trace)
def ca_trace(pdb_path):
    xyz, b = [], []
    for ln in open(pdb_path):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            b.append(float(ln[60:66] or 0))
    return np.array(xyz), np.array(b)


def structure_snapshot(pdb_path, pocket=(), size=230):
    """CA trace projected on its two principal axes, coloured by pLDDT; drug-contact residues ringed in orange."""
    d = Drawing(size, size)
    d.add(Rect(0, 0, size, size, fillColor=colors.HexColor("#f5f8fa"), strokeColor=LINE, rx=6, ry=6))
    if not os.path.exists(pdb_path):
        _label(d, size / 2, size / 2, "Structure not available", 8, anchor="middle")
        return d
    xyz, b = ca_trace(pdb_path)
    if len(xyz) < 3:
        return d
    c = xyz - xyz.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    p = c @ vt[:2].T
    depth = c @ vt[2]
    pad = 14
    span = max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) or 1
    pts = (p - p.min(0)) / span * (size - 2 * pad) + pad
    order = np.argsort(depth)                                        # far segments first
    for i in order:
        if i + 1 < len(pts) and np.linalg.norm(xyz[i + 1] - xyz[i]) < 4.5:   # chain breaks are not drawn
            d.add(Line(pts[i, 0], pts[i, 1], pts[i + 1, 0], pts[i + 1, 1], strokeColor=plddt_color(b[i]),
                       strokeWidth=1.6 if depth[i] > 0 else 1.0))
    for r in pocket:
        if 0 < r <= len(pts):
            d.add(Circle(pts[r - 1, 0], pts[r - 1, 1], 2.6, fillColor=FUNDUS, strokeColor=colors.white, strokeWidth=0.6))
    return d


# ------------------------------------------------------------------ AlphaFold confidence track with features
def plddt_track(plddt, features, pocket=(), width=500):
    """Per-residue confidence strip, a domain/transmembrane track and drug-contact ticks, on a residue axis."""
    n = max(1, len(plddt))
    h = 134
    d = Drawing(width, h)
    left, right = 34, width - 6
    scale = (right - left) / n
    x = lambda r: left + (r - 1) * scale
    _label(d, 0, 96, "pLDDT", 7)
    for i, v in enumerate(plddt):                                   # confidence profile as coloured bars
        bh = max(1.0, v / 100 * 40)
        d.add(Rect(x(i + 1), 70, max(scale, 0.6), bh, fillColor=plddt_color(v), strokeColor=None))
    for v in (50, 70, 90):
        y = 70 + v / 100 * 40
        d.add(Line(left, y, right, y, strokeColor=LINE, strokeWidth=0.4, strokeDashArray=[2, 2]))
        _label(d, left - 3, y - 2, str(v), 6, anchor="end")
    _label(d, 0, 50, "Features", 7)
    d.add(Line(left, 52, right, 52, strokeColor=LINE, strokeWidth=2))
    for f in features:
        col = FEATURE_COLORS.get(f["type"])
        if col is None:
            continue
        if f["type"] in ("Binding site", "Active site"):
            d.add(Polygon([x(f["start"]), 44, x(f["start"]) - 2.5, 40, x(f["start"]) + 2.5, 40], fillColor=col, strokeColor=None))
            continue
        d.add(Rect(x(f["start"]), 47, max(1.5, (f["end"] - f["start"] + 1) * scale), 10, fillColor=col, strokeColor=None,
                   rx=2, ry=2))
        if f["type"] == "Domain" and (f["end"] - f["start"]) * scale > 40:
            _label(d, x(f["start"]) + 2, 49.5, f["desc"][:int((f["end"] - f["start"]) * scale / 4)], 6, colors.white)
    if pocket:
        _label(d, 0, 24, "Drug contact", 7)
        for r in pocket:
            d.add(Line(x(r), 20, x(r), 32, strokeColor=FUNDUS, strokeWidth=1.2))
    d.add(Line(left, 12, right, 12, strokeColor=INK3, strokeWidth=0.5))
    step = max(50, int(round(n / 8 / 50)) * 50)
    for r in list(range(1, n + 1, step)) + [n]:
        d.add(Line(x(r), 10, x(r), 12, strokeColor=INK3, strokeWidth=0.5))
        _label(d, x(r), 2, str(r), 6, anchor="middle")
    legend = [("Domain", MOL), ("Transmembrane", colors.HexColor("#8a9aa9")), ("Signal peptide", RESEARCH),
              ("Binding/active site", FUNDUS)]
    lx = left
    for name, c in legend:
        d.add(Rect(lx, 124, 7, 7, fillColor=c, strokeColor=None))
        _label(d, lx + 10, 125, name, 6.5)
        lx += 92
    return d


# ------------------------------------------------------------------ drug landscape
PHASES = ["Approved", "Phase 3", "Phase 2", "Phase 1", "Early or unknown phase"]
MODALITY = {"small molecule": ("Small molecule", MOL), "antibody": ("Antibody", FUNDUS), "protein": ("Protein", RESEARCH)}


def drug_landscape(drugs, width=300):
    """Stacked bars: drugs per development stage, split by modality."""
    h = 16 * len(PHASES) + 40
    d = Drawing(width, h)
    counts = {p: {} for p in PHASES}
    for dr in drugs:
        m = MODALITY.get(dr.get("type"), ("Other", INK3))
        counts[dr["phase"]][m] = counts[dr["phase"]].get(m, 0) + 1
    most = max([sum(v.values()) for v in counts.values()] + [1])
    bar_w = width - 150
    for i, p in enumerate(PHASES):
        y = h - 24 - i * 16
        _label(d, 0, y + 2, p, 7.5, INK)
        x = 100
        for (name, col), k in counts[p].items():
            w = k / most * bar_w
            d.add(Rect(x, y, w, 10, fillColor=col, strokeColor=colors.white, strokeWidth=0.5))
            x += w
        _label(d, x + 4, y + 2, str(sum(counts[p].values())), 7.5, INK, bold=True)
    for k, (name, col) in enumerate(list(MODALITY.values()) + [("Other", INK3)]):
        lx, ly = (k % 2) * 90, 12 - (k // 2) * 10
        d.add(Rect(lx, ly, 7, 7, fillColor=col, strokeColor=None))
        _label(d, lx + 10, ly + 0.5, name, 6.5)
    return d


# ------------------------------------------------------------------ portfolio opportunity matrix
def opportunity_matrix(rows, width=500, height=260):
    """x = share of screened patients with a linked finding (demand); y = approved drugs (saturation);
    bubble = pipeline drugs. Quadrant labels frame the discussion; they are heuristics, not valuations."""
    d = Drawing(width, height)
    l, b, r, t = 40, 28, width - 10, height - 12
    d.add(Rect(l, b, r - l, t - b, fillColor=colors.HexColor("#f7f9fb"), strokeColor=LINE))
    xmax = 1.0
    ymax = max([row["approved"] for row in rows] + [1]) + 1
    X = lambda v: l + v / xmax * (r - l)
    Y = lambda v: b + v / ymax * (t - b)
    mx, my = X(xmax / 2), Y(ymax / 2)
    d.add(Line(mx, b, mx, t, strokeColor=LINE, strokeDashArray=[3, 3]))
    d.add(Line(l, my, r, my, strokeColor=LINE, strokeDashArray=[3, 3]))
    for text, x, y, anchor in [("Unmet need: many patients, few approved drugs", r - 4, b + 4, "end"),
                               ("Crowded: many patients, many drugs", r - 4, t - 10, "end"),
                               ("Niche and early", l + 4, b + 4, "start"), ("Established, smaller demand", l + 4, t - 10, "start")]:
        _label(d, x, y, text, 6.5, INK3, anchor)
    for row in rows:
        cx, cy = X(row["demand_share"]), Y(row["approved"])
        rad = 4 + min(row["pipeline"], 10) * 0.9
        d.add(Circle(cx, cy, rad, fillColor=colors.Color(MOL.red, MOL.green, MOL.blue, alpha=0.25), strokeColor=MOL,
                     strokeWidth=0.8))
        _label(d, cx + rad + 2, cy - 2.5, row["gene"], 7, INK, bold=True)
    _label(d, (l + r) / 2, 6, "Share of screened patients with a linked finding (demand)", 7, INK3, "middle")
    for v in range(0, int(ymax) + 1, max(1, int(ymax // 5))):
        _label(d, l - 4, Y(v) - 2, str(v), 6.5, INK3, "end")
    _label(d, 2, t - 2, "Approved", 6.5, INK3)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        _label(d, X(v), b - 9, f"{v * 100:.0f}%", 6.5, INK3, "middle")
    return d


# ------------------------------------------------------------------ score vs threshold gauge
def gauge(score, threshold, width=120):
    d = Drawing(width, 14)
    d.add(Rect(0, 4, width, 6, fillColor=colors.HexColor("#eef2f4"), strokeColor=None, rx=3, ry=3))
    s = max(0.0, min(1.0, float(score)))
    fill = width * s
    if fill >= 1:                                        # rounded ends only when the bar is long enough to show them
        r = 3 if fill >= 6 else 0
        d.add(Rect(0, 4, fill, 6, fillColor=SIGNAL if score >= threshold else CLEAR, strokeColor=None, rx=r, ry=r))
    tx = width * max(0.0, min(1.0, float(threshold)))
    d.add(Line(tx, 1, tx, 13, strokeColor=INK, strokeWidth=1.2))
    return d


def priority_bar(value, width=70):
    d = Drawing(width, 10)
    d.add(Rect(0, 2, width, 6, fillColor=colors.HexColor("#eef2f4"), strokeColor=None, rx=3, ry=3))
    d.add(Rect(0, 2, width * max(0, min(100, value)) / 100, 6, fillColor=MOL, strokeColor=None, rx=3, ry=3))
    return d
