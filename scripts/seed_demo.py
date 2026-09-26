"""
Seed the RetiLink demo workspace: two synthetic organizations, role accounts, synthetic patients, the five
acceptance scenarios (negative, positive, unusable, one eye, discrepant views) and a few historical SIMULATED
referrals so workflow analytics are not empty.

Scenario images are taken from the mBRSET *test* split (never seen in training) and stay in the local research
environment. Patient identities and histories are synthetic and are NOT linked to the mBRSET record.

  python scripts/seed_demo.py --reset [--no-images]
"""
import argparse
import hashlib
import json
import os
import random
import sys
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retilink.app import workflow as wf                                   # noqa: E402
from retilink.app.db import APP_ROOT, Base, SessionLocal, engine          # noqa: E402
from retilink.app.models import (Case, Image, Message, Patient, Referral, Review, Tenant, User, now)  # noqa: E402

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
    from retilink.ml.data import load_tables
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
    data = open(path, "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    dst = os.path.join(APP_ROOT, "images", f"t{case.tenant_id}", f"{sha}.jpg")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        open(dst, "wb").write(data)
    from PIL import Image as P
    w, h = P.open(dst).size
    db.add(Image(tenant_id=case.tenant_id, case_id=case.id, sha256=sha, path=dst, laterality=eye, view="macula-centred",
                 source=case.device, width=w, height=h, uploaded_by=user.id))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    if args.reset:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    db = SessionLocal()
    if db.query(Tenant).count():
        print("already seeded (use --reset)")
        return

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
    picks, table = (None, None) if args.no_images else pick_patients()
    for i, (key, ref, age, sex, dm, ins, oral, conds, eyes) in enumerate(scenarios):
        p = Patient(tenant_id=t1.id, ref=ref, age=age, sex=sex, dm_time=dm, insulin=ins, oral_treatment=oral, conditions=conds)
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
            print(f"{ref:9s} scenario {key:10s} <- mBRSET test patient {picks[key]} ({n} images)")

    # ---- historical SIMULATED referrals so the engagement analytics have a denominator
    stages = ["Closed", "Closed", "Closed", "Response received", "Scheduled", "Acknowledged", "Sent", "Patient declined",
              "Unreachable", "Declined"]
    rng = random.Random(3)
    for j, stage in enumerate(stages):
        p = Patient(tenant_id=t1.id, ref=f"RL-P09{j:02d}", age=rng.randint(40, 80), sex=rng.choice(["female", "male"]),
                    dm_time=rng.randint(2, 25), conditions=cond(systemic_hypertension=rng.choice(["present", "absent"])))
        db.add(p)
        db.flush()
        sent = now() - timedelta(days=rng.randint(3, 20), hours=rng.randint(0, 12))
        c = Case(tenant_id=t1.id, patient_id=p.id, owner_id=U["pcp2"].id, created_by=U["op"].id,
                 encounter_date=sent.date().isoformat(), device="Portable smartphone fundus camera (dilated)",
                 status="Signed", created_at=sent - timedelta(hours=3))
        db.add(c)
        db.flush()
        rv = Review(tenant_id=t1.id, case_id=c.id, case_version=1, hcp_id=U["pcp2"].id, decision="manual_review",
                    interpretation="SIMULATED historical case for workflow analytics.", next_action="refer_retina",
                    signature=wf.digest({"sim": j}), signed_at=sent - timedelta(hours=1))
        db.add(rv)
        db.flush()
        spec = U["ret"] if j % 3 else U["ret2"]
        pkg = {"topic": "retinal", "question": "SIMULATED", "facts": [{"text": "SIMULATED historical referral", "source": "seed"}],
               "gaps": [], "limitations": [], "evidence": {"answer": "", "references": []}, "image_ids": []}
        r = Referral(tenant_id=t1.id, case_id=c.id, review_id=rv.id, sender_id=U["pcp2"].id, recipient_id=spec.id,
                     owner_id=spec.id, topic="retinal", question="SIMULATED: assess for referable DR.", package=pkg,
                     package_hash=wf.digest(pkg), signature=wf.digest({"s": j}), idempotency_key=f"seed-{j}",
                     stage="Sent", sent_at=sent, due_at=sent + wf.DEMO_DUE["acknowledge"])
        db.add(r)
        db.flush()
        path = {"Closed": ["Acknowledged", "Scheduled", "Visit recorded", "Response received", "Closed"],
                "Response received": ["Acknowledged", "Scheduled", "Visit recorded", "Response received"],
                "Scheduled": ["Acknowledged", "Scheduled"], "Acknowledged": ["Acknowledged"], "Sent": [],
                "Patient declined": ["Acknowledged", "Patient declined"], "Unreachable": ["Acknowledged", "Unreachable"],
                "Declined": ["Declined"]}[stage]
        for s in path:
            actor = spec if s in ("Acknowledged", "Response received", "Declined", "Visit recorded") else \
                U["pcp2"] if s == "Closed" else U["coord"]
            if s == "Response received":
                db.add(Message(tenant_id=t1.id, referral_id=r.id, author_id=spec.id, kind="response",
                               body="SIMULATED signed response.", recommendation="Specialist follow-up visit",
                               signature=wf.digest({"r": j})))
                db.flush()
            if s == "Scheduled":
                r.appointment = {"date": (sent + timedelta(days=5)).date().isoformat(), "time": "10:00",
                                 "timezone": "America/New_York", "facility": "Synthetic Eye Clinic", "confirmation": "phone"}
            wf.transition(db, r, s, actor, "SIMULATED")
        if r.acknowledged_at:
            r.acknowledged_at = sent + timedelta(hours=rng.uniform(1.5, 30))
    db.commit()
    print("seeded. Accounts:", ", ".join(f"{u.name} ({u.role})" for u in U.values()))


if __name__ == "__main__":
    main()
