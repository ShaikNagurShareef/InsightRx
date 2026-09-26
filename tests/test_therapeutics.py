"""Therapeutics layer: therapy mapping, interaction checks, sponsorship firewall, public-data privacy and fallbacks,
the medical-information channel and access control. Runs offline (no network)."""
import os

os.environ.setdefault("RETILINK_EXTERNAL", "offline")

import httpx  # noqa: E402
import pytest  # noqa: E402

from test_workflow import client_as, fundus_bytes, new_case, uid, upload  # noqa: E402  (seeds a temp workspace)

from retilink.app import external  # noqa: E402
from retilink.app import therapeutics as tx  # noqa: E402
from retilink.app.db import SessionLocal  # noqa: E402
from retilink.app.models import MedInfoRequest, Patient  # noqa: E402

DR_RESULT = {"overall": "Referable DR signal", "images": {"1": {"edema_flag": True}}}
KIDNEY_SNAPSHOT = [{"key": "kidneys", "state": "signal"}, {"key": "heart", "state": "recorded"}]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(external, "OFFLINE", True)
    external._cache.clear()


# ------------------------------------------------------------------ mapping and interactions
def test_findings_order_retina_first_and_diabetes_always():
    keys = [f["key"] for f in tx.findings(DR_RESULT, KIDNEY_SNAPSHOT)]
    assert keys == ["dr", "edema", "kidneys", "heart", "metabolism"]
    assert [f["key"] for f in tx.findings(None, [])] == ["metabolism"]


def test_each_finding_maps_to_guideline_classes():
    opts = {o["key"] for o in tx.options_for(tx.findings(DR_RESULT, KIDNEY_SNAPSHOT), [])}
    assert {"anti_vegf", "fenofibrate", "acei_arb", "sglt2i", "finerenone", "glp1ra"} <= opts
    nerves = tx.findings(None, [{"key": "nerves", "state": "exploratory"}])
    assert "neuropathic_pain" in {o["key"] for o in tx.options_for(nerves, [])}


def test_semaglutide_with_dr_signal_is_a_major_alert():
    alerts = tx.check_interactions(["semaglutide"], (), ["dr"])
    assert alerts and alerts[0]["severity"] == "major" and alerts[0]["title"] == "Diabetic retinopathy complications"
    assert not tx.check_interactions(["semaglutide"], (), ["kidneys"])


def test_pioglitazone_with_macular_edema_and_dual_ras_blockade():
    titles = {a["title"] for a in tx.check_interactions(["pioglitazone", "lisinopril", "losartan"], (), ["edema"])}
    assert {"Macular edema with pioglitazone", "Dual RAS blockade"} <= titles


def test_candidate_vs_current_but_not_candidate_vs_candidate():
    assert any(a["title"] == "Hyperkalaemia risk" for a in tx.check_interactions(["lisinopril"], ["finerenone"], []))
    assert not tx.check_interactions([], ["lisinopril", "losartan"], [])       # nothing prescribed yet


def test_already_on_recognises_either_ace_or_arb():
    opt = next(o for o in tx.options_for(tx.findings(None, KIDNEY_SNAPSHOT), ["losartan"]) if o["key"] == "acei_arb")
    assert opt["already_on"] == ["losartan"]
    opt = next(o for o in tx.options_for(tx.findings(None, KIDNEY_SNAPSHOT), ["fish oil"]) if o["key"] == "sglt2i")
    assert opt["already_on"] == []


def test_medicine_names_are_validated():
    assert tx.normalize_meds(["Metformin", "metformin", "<script>", "a" * 60, "valsartan, rosuvastatin"]) == \
        ["metformin", "valsartan", "rosuvastatin"]


# ------------------------------------------------------------------ sponsorship firewall
def test_ranking_is_invariant_to_sponsorship(monkeypatch):
    fnd = tx.findings(DR_RESULT, KIDNEY_SNAPSHOT)
    before = [o["key"] for o in tx.options_for(fnd, ["insulin"])]
    monkeypatch.setattr(tx, "SPONSORED", [{"sponsor": "X", "finding": k, "title": "t", "text": "x", "class": "finerenone"}
                                          for k in ("dr", "kidneys")] * 5)
    assert [o["key"] for o in tx.options_for(fnd, ["insulin"])] == before
    assert tx.sponsored_for(fnd)                                              # shown separately, after the options


# ------------------------------------------------------------------ public data: privacy and fallback
def test_outbound_queries_carry_no_patient_data(monkeypatch):
    sent = []

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"studies": [], "results": []}

    def fake_get(url, params=None, **kw):
        sent.append((url, dict(params or {})))
        return Resp()
    monkeypatch.setattr(external, "OFFLINE", False)
    monkeypatch.setattr(external.httpx, "get", fake_get)
    fnd = tx.findings(DR_RESULT, KIDNEY_SNAPSHOT)
    tx.trial_matches(fnd, age=63, sex="male", zip_code="30303")
    external.nppes("Nephrology", "30303")
    blob = repr(sent)
    assert sent and "63" not in blob and "male" not in blob and "RL-P" not in blob
    allowed = set(external.TRIAL_QUERIES.values()) | set(external.TAXONOMIES)
    assert all(p.get("query.cond", p.get("taxonomy_description")) in allowed for _, p in sent)


def test_network_failure_falls_back_to_snapshot(monkeypatch):
    def boom(*a, **kw):
        raise httpx.ConnectTimeout("offline")
    monkeypatch.setattr(external, "OFFLINE", False)
    monkeypatch.setattr(external.httpx, "get", boom)
    got = external.trials("edema", "30303")
    assert got["source"] == "snapshot" and got["items"]
    assert external.nppes("Ophthalmology", "not-a-zip")["source"] == "snapshot"
    assert external.provider("12345") is None                                # malformed NPI never queried


def test_trial_age_precheck():
    assert tx.trial_fit({"min_age": "18 Years", "max_age": "65 Years"}, 70, "male")[0] == "outside"
    assert tx.trial_fit({"min_age": "18 Years"}, 50, "female")[0] == "possible"
    assert tx.trial_fit({}, None, None)[0] == "check"


# ------------------------------------------------------------------ web flow
@pytest.fixture(scope="module")
def med_case():
    op = client_as("Sam Rivera")
    cid = new_case(op, "RL-TX1")
    assert upload(op, cid, "OD", fundus_bytes(3)).status_code == 303
    assert op.post(f"/cases/{cid}/analyze").status_code == 303
    pcp = client_as("Dr. Alex Morgan")
    r = pcp.post(f"/cases/{cid}/intake", data={"age": "63", "sex": "male", "insulin": "yes", "oral_treatment": "yes",
                                               "med": ["semaglutide", "lisinopril"], "med_other": "losartan", "tab": "systemic"})
    assert r.status_code == 303
    return cid


def test_medicines_round_trip_through_intake(med_case):
    with SessionLocal() as db:
        from retilink.app.models import Case
        p = db.get(Patient, db.get(Case, med_case).patient_id)
        assert p.medications == ["semaglutide", "lisinopril", "losartan"]


def test_therapy_page_shows_alerts_targets_and_cms(med_case):
    page = client_as("Dr. Alex Morgan").get(f"/cases/{med_case}/therapy").text
    assert "Dual RAS blockade" in page and "CPT 92228" in page and "/therapeutics/targets/" in page


def test_target_pages_and_explorer():
    c = client_as("Dr. Alex Morgan")
    assert "AlphaFold" in c.get("/therapeutics").text
    page = c.get("/therapeutics/targets/VEGFA").text
    assert "/static/structures/1CZ8.pdb" in page and "ranibizumab" in page
    assert c.get("/therapeutics/targets/NOPE").status_code == 404


def test_medinfo_request_is_deidentified_and_desk_cannot_open_cases(med_case):
    pcp = client_as("Dr. Alex Morgan")
    r = pcp.post("/medinfo", data={"case_id": med_case, "drug": "semaglutide", "therapy_class": "glp1ra",
                                   "question": "Label guidance on retinopathy monitoring?"})
    assert r.status_code == 303
    with SessionLocal() as db:
        req = db.query(MedInfoRequest).order_by(MedInfoRequest.id.desc()).first()
        assert "RL-TX1" not in req.context and "63" not in req.context and "60s" in req.context
        rid = req.id
    desk = client_as("Morgan Lee, PharmD")
    assert desk.get(f"/cases/{med_case}").status_code in (403, 404)
    assert desk.get(f"/cases/{med_case}/therapy").status_code in (403, 404)
    assert "RL-TX1" not in desk.get("/medinfo").text
    assert desk.post(f"/medinfo/{rid}/answer", data={"answer": "See the label warning.", "answer_source": ""}).status_code == 422
    assert desk.post(f"/medinfo/{rid}/answer", data={"answer": "See the label warning section.",
                                                     "answer_source": "Label, Warnings"}).status_code == 303
    assert pcp.post(f"/medinfo/{rid}/answer", data={"answer": "x" * 20, "answer_source": "y"}).status_code == 403


def test_medinfo_rejects_unknown_drug(med_case):
    r = client_as("Dr. Alex Morgan").post("/medinfo", data={"case_id": med_case, "drug": "aspirin; drop table",
                                                            "question": "Anything at all here?"})
    assert r.status_code == 422


def test_refer_out_and_letter_use_registry_snapshot(med_case):
    pcp = client_as("Dr. Alex Morgan")
    page = pcp.get(f"/cases/{med_case}/refer-out", params={"topic": "nephropathy", "zip": "30303"}).text
    assert "Nephrology" in page and "Referral letter" in page
    npi = str(external._snapshot("nppes_snapshot.json")["Nephrology"][0]["npi"])
    letter = pcp.get(f"/cases/{med_case}/letter", params={"npi": npi, "topic": "nephropathy"})
    assert letter.status_code == 200 and npi in letter.text and "RL-TX1" in letter.text
    assert pcp.get(f"/cases/{med_case}/letter", params={"npi": "0000000000"}).status_code == 404


def test_quality_page():
    assert "CMS131" in client_as("Dr. Alex Morgan").get("/quality").text


def test_other_tenant_cannot_open_therapy(med_case):
    assert client_as("Dr. Jamie Outside").get(f"/cases/{med_case}/therapy").status_code in (403, 404)
    assert uid("Dr. Jamie Outside")
