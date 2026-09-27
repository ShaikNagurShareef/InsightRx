"""
Populate a local Insight Rx workspace from test_data/ (for offline or portable installs without the mBRSET dataset).

Creates the demo organisations and accounts (if the database is empty), then one case per test patient in
test_data/mbrset/patients/<scenario>/ with its photos, recorded history and medicines, analysed by the models in
INSIGHTRX_MODEL_DIR (ONNX bundle on CPU, or PyTorch weights). Safe to run twice: existing test patients are skipped.

  python scripts/seed_local.py [--limit N]
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from insightrx.app import main as app  # noqa: E402  (creates tables, seeds accounts, applies upgrades)
from insightrx.app.db import SessionLocal  # noqa: E402
from insightrx.app.models import Case, Image, Patient, User  # noqa: E402

PATIENTS = os.path.join(ROOT, "test_data", "mbrset", "patients")
HISTORY = ["systemic_hypertension", "nephropathy", "vascular_disease", "acute_myocardial_infarction", "neuropathy",
           "diabetic_foot", "obesity"]


def medicines(p):
    """Plausible current medicines consistent with the recorded treatment and history (synthetic)."""
    meds = []
    if p.get("oral_diabetes_medication") == "yes":
        meds.append("metformin")
    if p.get("insulin") == "yes":
        meds.append("insulin")
    h = p.get("reported_history", {})
    if h.get("systemic_hypertension") == "present":
        meds.append("lisinopril")
    if h.get("nephropathy") == "present":
        meds.append("empagliflozin")
    if h.get("neuropathy") == "present":
        meds.append("pregabalin")
    return meds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N test patients")
    args = ap.parse_args()
    folders = sorted(d for d in glob.glob(os.path.join(PATIENTS, "*")) if os.path.isdir(d))
    if args.limit:
        folders = folders[: args.limit]
    svc = app.get_service()
    print(f"models: {svc.mode} ({svc.version}, {getattr(svc, 'backend', None) or 'n/a'} on {getattr(svc, 'device', 'n/a')})")
    with SessionLocal() as db:
        owner = db.query(User).filter(User.name == "Dr. Alex Morgan").first()
        operator = db.query(User).filter(User.name == "Sam Rivera").first()
        if not owner or not operator:
            sys.exit("demo accounts missing: start the app once (it seeds them) and rerun")
        for d in folders:
            ref = "TD-" + os.path.basename(d)[:2]
            if db.query(Patient).filter(Patient.ref == ref, Patient.tenant_id == owner.tenant_id).first():
                print(f"{ref}: already present, skipped")
                continue
            info = json.load(open(os.path.join(d, "patient.json")))
            conds = {c: {"value": info.get("reported_history", {}).get(c, "unknown"), "source": "test_data patient.json",
                         "date": "", "verification": "reported"} for c in HISTORY}
            p = Patient(tenant_id=owner.tenant_id, ref=ref, age=info.get("age"), sex=info.get("sex"),
                        dm_time=info.get("years_with_diabetes"), insulin=info.get("insulin", "unknown"),
                        oral_treatment=info.get("oral_diabetes_medication", "unknown"), conditions=conds,
                        medications=medicines(info))
            db.add(p)
            db.flush()
            case = Case(tenant_id=owner.tenant_id, patient_id=p.id, owner_id=owner.id, created_by=operator.id,
                        encounter_date=time.strftime("%Y-%m-%d"), device="Portable smartphone fundus camera (test_data)",
                        symptoms="")
            db.add(case)
            db.flush()
            n = 0
            for path in sorted(glob.glob(os.path.join(d, "*.jpg"))):
                eye = os.path.basename(path)[:2] if os.path.basename(path)[:2] in ("OD", "OS") else "unknown"
                data = open(path, "rb").read()
                img = app._validate_upload(data)
                sha, stored = app._store(case, data, img)
                db.add(Image(tenant_id=case.tenant_id, case_id=case.id, sha256=sha, path=stored, laterality=eye,
                             view="macula-centred", source=case.device, width=img.size[0], height=img.size[1],
                             uploaded_by=operator.id, data=None if stored else data))
                n += 1
            case.status = "Images ready"
            app.wf.bump_version(db, case, operator, f"{n} image(s) from test_data")
            db.flush()
            t = time.time()
            run = app.run_analysis(db, case, operator)
            db.commit()
            print(f"{ref} ({info.get('scenario')}): {n} photos -> {run.result.get('overall')} in {time.time() - t:.0f} s")
    print("done: open the app, sign in as Dr. Alex Morgan, and see Patients")


if __name__ == "__main__":
    main()
