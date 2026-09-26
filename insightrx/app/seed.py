"""
Seed the Insight Rx demo workspace: two synthetic organizations, role accounts, synthetic patients, the five
acceptance scenarios (negative, positive, unusable, one eye, discrepant views) and, with images, a four-week caseload
of real test-split patients analysed by the real models (insightrx/app/demo_data.py).

Scenario images are taken from the mBRSET *test* split (never seen in training) and stay in the local research
environment. Patient identities and histories are synthetic and are NOT linked to the mBRSET record.

Used by scripts/seed_demo.py and, without images, automatically when the app starts on an empty database.
"""
import hashlib
import json
import os
import random
from datetime import timedelta

from . import workflow as wf
from .db import APP_ROOT, IMAGE_STORE
from .models import Case, Image, MedInfoRequest, Message, Patient, Referral, Review, Tenant, User, now

SPLIT = "/data/users3/nshaik3/Projects/Oculomics/OculoMoE/cache/mbrset/splits_protocol_P1.json"


def cond(**kw):
    out = {}
    for c in ["systemic_hypertension", "nephropathy", "vascular_disease", "acute_myocardial_infarction", "neuropathy",
              "diabetic_foot", "obesity"]:
        v = kw.get(c, "unknown")
        out[c] = {"value": v, "source": "synthetic intake record" if v != "unknown" else "not recorded",
                  "date": "2026-08-14" if v != "unknown" else "", "verification": "reported"}
    return out


def pick_patients():
    """Choose test-split patients that realise each scenario from the ground-truth labels."""
    from insightrx.ml.data import load_tables
    images, _ = load_tables()
    test = set(json.load(open(SPLIT))["splits"]["0"]["test"])
    t = images[images["patient_id"].isin(test)]
    g = t.groupby("patient_id")
    rng = random.Random(7)
    good = [p for p, d in g if (d["quality_poor"] == 0).all() and d["icdr"].notna().all()]
    neg = [p for p in good if (g.get_group(p)["icdr"] == 0).all()]
    pos = [p for p in good if (g.get_group(p)["icdr"] >= 2).all()]
    disc = [p for p in good if set(g.get_group(p).groupby("eye")["dr_referable"].max()) == {0.0, 1.0}]
    poor = sorted(g, key=lambda kv: -kv[1]["quality_poor"].sum())
    return {"negative": rng.choice(neg), "positive": rng.choice(pos), "unusable": poor[0][0],
            "one_eye": rng.choice([p for p in pos if p != pos[0]] or pos), "discrepant": rng.choice(disc)}, t


def add_image(db, case, path, eye, user):
    import io
    from PIL import Image as P
    data = open(path, "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    w, h = P.open(io.BytesIO(data)).size
    dst = ""
    if IMAGE_STORE == "fs":
        dst = os.path.join(APP_ROOT, "images", f"t{case.tenant_id}", f"{sha}.jpg")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            open(dst, "wb").write(data)
    db.add(Image(tenant_id=case.tenant_id, case_id=case.id, sha256=sha, path=dst, laterality=eye, view="macula-centred",
                 source=case.device, width=w, height=h, uploaded_by=user.id,
                 data=data if IMAGE_STORE == "db" else None))


SCENARIO_MEDS = {   # synthetic medication lists chosen to exercise the interaction checks
    "RL-P0101": ["metformin", "atorvastatin"],
    "RL-P0102": ["metformin", "insulin", "semaglutide", "lisinopril", "losartan"],
    "RL-P0103": ["metformin", "pioglitazone"],
    "RL-P0104": ["insulin", "pregabalin", "lisinopril"],
    "RL-P0105": ["metformin", "amlodipine", "hydrochlorothiazide"],
}


def demo_medications(p) -> list:
    """Plausible synthetic medicines consistent with the recorded treatment and history (deterministic per patient)."""
    if p.ref in SCENARIO_MEDS:
        return SCENARIO_MEDS[p.ref]
    rng = random.Random(p.ref)
    conds = {k: (v or {}).get("value") for k, v in (p.conditions or {}).items()}
    meds = []
    if p.oral_treatment == "yes":
        meds.append("metformin")
        meds += rng.sample(["glipizide", "sitagliptin", "empagliflozin", "pioglitazone", "semaglutide"], rng.choice([0, 1, 1]))
    if p.insulin == "yes":
        meds.append("insulin")
    if conds.get("systemic_hypertension") == "present":
        meds.append(rng.choice(["lisinopril", "losartan", "amlodipine", "lisinopril"]))
    if conds.get("neuropathy") == "present" and rng.random() < .5:
        meds.append("pregabalin")
    if rng.random() < .5:
        meds.append("atorvastatin")
    return meds


def backfill_medications(db):
    """Give synthetic patients seeded before medications existed a medication list (idempotent)."""
    todo = db.query(Patient).filter(Patient.synthetic.is_(True), Patient.medications.is_(None)).all()
    for p in todo:
        p.medications = demo_medications(p)
    if todo:
        db.commit()
    return len(todo)


def ensure_desk(db):
    """The demo's fictional manufacturer medical-information desk, with two example requests (idempotent)."""
    if db.query(User).filter(User.role == "medinfo").count():
        return False
    t = Tenant(name="Demo Pharma Medical Information (fictional)")
    db.add(t)
    db.flush()
    desk = User(tenant_id=t.id, name="Morgan Lee, PharmD", role="medinfo", specialty="Medical information",
                prefs={"mode": "immediate"})
    db.add(desk)
    db.flush()
    pcp = db.query(User).filter(User.role == "referring").order_by(User.id).first()
    if pcp:
        db.add(MedInfoRequest(
            tenant_id=pcp.tenant_id, requester_id=pcp.id, desk_tenant_id=t.id, drug="semaglutide", therapy_class="glp1ra",
            question="Patient with a referable DR signal: what does the label say about starting semaglutide, and how soon should retinal follow-up happen?",
            context="Adult in their 60s with diabetes; screening findings: referable diabetic retinopathy.",
            status="Answered", answered_by=desk.id, answered_at=now() - timedelta(days=2),
            answer="The label warns of diabetic retinopathy complications, seen in SUSTAIN-6 mostly in patients with existing "
                   "retinopathy and rapid glucose lowering. Patients with a history of DR should be monitored for progression. "
                   "Timing of retinal follow-up is a clinical decision; no fixed interval is specified in the label.",
            answer_source="Semaglutide injection prescribing information, Warnings and Precautions: diabetic retinopathy complications",
            created_at=now() - timedelta(days=3)))
        db.add(MedInfoRequest(
            tenant_id=pcp.tenant_id, requester_id=pcp.id, desk_tenant_id=t.id, drug="finerenone", therapy_class="finerenone",
            question="Kidney signal on retinal screening, already on lisinopril. What potassium monitoring does the label require when adding finerenone?",
            context="Adult in their 50s with diabetes; screening findings: diabetic kidney disease risk.",
            created_at=now() - timedelta(hours=5)))
    db.commit()
    return True


def seed(db, with_images=False, verbose=True):
    """Populate an empty database. Returns False if it was already seeded."""
    if db.query(Tenant).count():
        return False

    t1 = Tenant(name="Peachtree Community Health (synthetic)")
    t2 = Tenant(name="Riverside Eye Group (synthetic, other tenant)")
    db.add_all([t1, t2])
    db.flush()
    U = {}
    for key, name, role, spec, tid in [
        ("op", "Sam Rivera", "operator", "Screening operator", t1.id),
        ("pcp", "Dr. Alex Morgan", "referring", "Primary care", t1.id),
        ("pcp2", "Dr. Casey Patel", "referring", "Diabetes care", t1.id),
        ("ret", "Dr. Priya Nair", "specialist", "Retina", t1.id),
        ("ret2", "Dr. Omar Haddad", "specialist", "Comprehensive ophthalmology", t1.id),
        ("neph", "Dr. Lena Brooks", "specialist", "Nephrology", t1.id),
        ("cardio", "Dr. Ken Ito", "specialist", "Cardiology", t1.id),
        ("coord", "Taylor Brooks", "coordinator", "Care coordination", t1.id),
        ("admin", "Jordan Admin", "admin", "", t1.id),
        ("other", "Dr. Riley Other", "specialist", "Retina (other org)", t2.id),
        ("otherpcp", "Dr. Jamie Outside", "referring", "Primary care (other org)", t2.id),
    ]:
        U[key] = User(tenant_id=tid, name=name, role=role, specialty=spec, prefs={"mode": "immediate"})
        db.add(U[key])
    db.flush()

    scenarios = [
        ("negative", "RL-P0101", 58, "female", 9, "no", "yes", cond(systemic_hypertension="absent", nephropathy="absent", obesity="absent"), ["OD", "OS"]),
        ("positive", "RL-P0102", 63, "male", 17, "yes", "yes", cond(systemic_hypertension="present", nephropathy="unknown", vascular_disease="absent"), ["OD", "OS"]),
        ("unusable", "RL-P0103", 71, "female", 12, "unknown", "yes", cond(), ["OD", "OS"]),
        ("one_eye", "RL-P0104", 55, "male", 20, "yes", "no", cond(systemic_hypertension="present", neuropathy="present", diabetic_foot="present"), ["OD"]),
        ("discrepant", "RL-P0105", 49, "female", 6, "no", "yes", cond(systemic_hypertension="present", obesity="present"), ["OD", "OS"]),
    ]
    picks, table = pick_patients() if with_images else (None, None)
    for i, (key, ref, age, sex, dm, ins, oral, conds, eyes) in enumerate(scenarios):
        p = Patient(tenant_id=t1.id, ref=ref, age=age, sex=sex, dm_time=dm, insulin=ins, oral_treatment=oral, conditions=conds,
                    medications=SCENARIO_MEDS[ref])
        db.add(p)
        db.flush()
        c = Case(tenant_id=t1.id, patient_id=p.id, owner_id=U["pcp"].id, created_by=U["op"].id,
                 encounter_date="2026-09-26", device="Portable smartphone fundus camera (dilated)",
                 symptoms="" if key != "positive" else "Reports gradual blurring when reading (clinician-entered).")
        db.add(c)
        db.flush()
        wf.audit(db, t1.id, c.id, U["op"].id, "case_created", f"patient {ref} (scenario: {key})", 1)
        if picks:
            rows = table[table["patient_id"] == picks[key]]
            n = 0
            for _, r in rows.iterrows():
                if r["eye"] in eyes:
                    add_image(db, c, r["path"], r["eye"], U["op"])
                    n += 1
            c.status = "Images ready"
            wf.bump_version(db, c, U["op"], f"{n} image(s) uploaded (scenario {key})")
            if verbose:
                print(f"{ref:9s} scenario {key:10s} <- mBRSET test patient {picks[key]} ({n} images)")

    db.flush()
    if with_images:                          # four weeks of activity around real mBRSET test-split patients
        from .demo_data import populate
        n = populate(db, U, t1, exclude=list(picks.values()) if picks else ())
        if verbose:
            print(f"demo caseload: {n} real-image cases")
    db.commit()
    backfill_medications(db)
    ensure_desk(db)
    if verbose:
        print("seeded. Accounts:", ", ".join(f"{u.name} ({u.role})" for u in U.values()))
    return True
