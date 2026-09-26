"""End-to-end workflow, access and safety tests (simulated vision mode, temporary workspace)."""
import io
import os
import re
import subprocess
import sys
import tempfile

import pytest
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="retilink_test_")
os.environ["RETILINK_APP_ROOT"] = TMP
os.environ["RETILINK_MODEL_DIR"] = os.path.join(TMP, "no_model")     # -> SIMULATED vision
os.environ["RETILINK_SECRET"] = "test"
sys.path.insert(0, ROOT)
subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "seed_demo.py"), "--reset", "--no-images"], check=True,
               env=os.environ.copy(), capture_output=True)

from fastapi.testclient import TestClient  # noqa: E402

from retilink.app import main  # noqa: E402
from retilink.app.db import SessionLocal  # noqa: E402
from retilink.app.llm import RestrictedPayload, guard  # noqa: E402
from retilink.app.models import Referral, User  # noqa: E402


def fundus_bytes(seed=0, fmt="JPEG"):
    im = Image.new("RGB", (800, 800), (0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((40, 40, 760, 760), fill=(190 - seed, 90, 40))
    d.ellipse((500, 350, 580, 430), fill=(240, 200, 120))
    b = io.BytesIO()
    im.save(b, fmt)
    return b.getvalue()


def uid(name):
    with SessionLocal() as db:
        return db.query(User).filter(User.name == name).one().id


def client_as(name):
    c = TestClient(main.app, follow_redirects=False)
    c.post("/login", data={"user_id": uid(name)})
    return c


def new_case(op, ref="RL-T1"):
    r = op.post("/cases", data={"patient_ref": ref, "encounter_date": "2026-09-26", "device": "portable camera",
                                "owner_id": uid("Dr. Alex Morgan")})
    assert r.status_code == 303
    return int(r.headers["location"].split("?")[0].rsplit("/", 1)[1])


def upload(c, cid, eye, data, name="a.jpg"):
    return c.post(f"/cases/{cid}/images", data={"laterality": eye}, files=[("files", (name, data, "image/jpeg"))])


def sign(pcp, cid, decision="accept", reason=""):
    return pcp.post(f"/cases/{cid}/reviews", data={"decision": decision, "interpretation": "reviewed",
                                                  "override_reason": reason, "next_action": "refer_retina", "attest": "1"})


def send(pcp, cid, question="Please assess.", recipient="Dr. Priya Nair", key=None):
    page = pcp.get(f"/cases/{cid}", params={"tab": "consult", "topic": "retinal", "question": question,
                                           "recipient_id": uid(recipient)}).text
    token = re.search(r'name="approval_token" value="([0-9a-f]+)"', page).group(1)
    idem = key or re.search(r'name="idempotency_key" value="([0-9a-f]+)"', page).group(1)
    return pcp.post("/referrals", data={"case_id": cid, "topic": "retinal", "question": question,
                                        "recipient_id": uid(recipient), "approval_token": token,
                                        "idempotency_key": idem, "attest": "1"}), token, idem


@pytest.fixture(scope="module")
def ready_case():
    op, pcp = client_as("Sam Rivera"), client_as("Dr. Alex Morgan")
    cid = new_case(op)
    assert upload(op, cid, "OD", fundus_bytes(0)).status_code == 303
    assert upload(op, cid, "OS", fundus_bytes(5)).status_code == 303
    assert op.post(f"/cases/{cid}/analyze").status_code == 303
    assert sign(pcp, cid).status_code == 303
    return cid


def test_missing_identity_blocks_case():
    op = client_as("Sam Rivera")
    r = op.post("/cases", data={"encounter_date": "2026-09-26", "device": "x"})
    assert r.status_code == 422


def test_rejects_non_image_and_duplicate():
    op = client_as("Sam Rivera")
    cid = new_case(op, "RL-T2")
    assert upload(op, cid, "OD", b"not an image at all").status_code == 415
    assert upload(op, cid, "OD", fundus_bytes(1)).status_code == 303
    r = upload(op, cid, "OD", fundus_bytes(1))
    assert "Duplicate" in r.headers["location"]


def test_operator_cannot_sign(ready_case):
    op = client_as("Sam Rivera")
    assert sign(op, ready_case).status_code == 403


def test_disagree_requires_reason(ready_case):
    pcp = client_as("Dr. Alex Morgan")
    assert sign(pcp, ready_case, "disagree", "").status_code == 422


def test_cross_tenant_and_unassigned_denied(ready_case):
    other = client_as("Dr. Riley Other")
    assert other.get(f"/cases/{ready_case}").status_code == 404
    with SessionLocal() as db:
        from retilink.app.models import Image as Img
        iid = db.query(Img).filter(Img.case_id == ready_case).first().id
    assert other.get(f"/images/{iid}").status_code == 404
    assert client_as("Dr. Priya Nair").get(f"/cases/{ready_case}").status_code == 403    # not yet a recipient
    assert client_as("Jordan Admin").get(f"/images/{iid}").status_code == 403           # admin is not clinical access


def test_idempotent_referral_and_full_handoff(ready_case):
    pcp = client_as("Dr. Alex Morgan")
    r1, token, idem = send(pcp, ready_case)
    assert r1.status_code == 303
    rid = int(r1.headers["location"].split("/")[2].split("?")[0])
    for _ in range(3):                                    # double click / retry -> same referral
        again = pcp.post("/referrals", data={"case_id": ready_case, "topic": "retinal", "question": "Please assess.",
                                             "recipient_id": uid("Dr. Priya Nair"), "approval_token": token,
                                             "idempotency_key": idem, "attest": "1"})
        assert f"/referrals/{rid}" in again.headers["location"]
    with SessionLocal() as db:
        assert db.query(Referral).filter(Referral.idempotency_key == idem).count() == 1

    spec, coord = client_as("Dr. Priya Nair"), client_as("Taylor Brooks")
    act = lambda c, **d: c.post(f"/referrals/{rid}/action", data=d)
    assert act(pcp, action="close").status_code == 409                       # illegal: Sent -> Closed
    assert act(coord, action="acknowledge").status_code == 403               # only recipient acknowledges
    assert act(spec, action="acknowledge").status_code == 303
    assert act(coord, action="schedule", date="2026-10-02", time="09:00", tz="America/New_York",
               facility="Eye clinic").status_code == 303
    assert act(spec, action="visit").status_code == 303
    assert act(spec, action="respond", body="Moderate NPDR confirmed.", recommendation="Specialist follow-up visit").status_code == 422
    assert act(spec, action="respond", body="Moderate NPDR confirmed.", recommendation="Specialist follow-up visit",
               attest="1").status_code == 303
    assert act(pcp, action="close").status_code == 303
    with SessionLocal() as db:
        ref = db.get(Referral, rid)
        assert ref.stage == "Closed" and ref.closure_code == "completed"


def test_signature_invalidated_by_new_image(ready_case):
    op, pcp = client_as("Sam Rivera"), client_as("Dr. Alex Morgan")
    page = pcp.get(f"/cases/{ready_case}", params={"tab": "consult", "topic": "retinal", "question": "Q2",
                                                  "recipient_id": uid("Dr. Omar Haddad")}).text
    token = re.search(r'name="approval_token" value="([0-9a-f]+)"', page).group(1)
    assert upload(op, ready_case, "OS", fundus_bytes(9)).status_code == 303      # case changes after preview
    r = pcp.post("/referrals", data={"case_id": ready_case, "topic": "retinal", "question": "Q2",
                                     "recipient_id": uid("Dr. Omar Haddad"), "approval_token": token,
                                     "idempotency_key": "k-stale", "attest": "1"})
    assert r.status_code == 409                                                   # stale signature -> no dispatch


def test_recipient_must_be_granted_specialist(ready_case):
    pcp = client_as("Dr. Alex Morgan")
    assert sign(pcp, ready_case).status_code == 303
    page = pcp.get(f"/cases/{ready_case}", params={"tab": "consult", "topic": "retinal", "question": "Q3",
                                                  "recipient_id": uid("Dr. Riley Other")}).text
    token = re.search(r'name="approval_token" value="([0-9a-f]+)"', page).group(1)
    r = pcp.post("/referrals", data={"case_id": ready_case, "topic": "retinal", "question": "Q3",
                                     "recipient_id": uid("Dr. Riley Other"), "approval_token": token,
                                     "idempotency_key": "k-other", "attest": "1"})
    assert r.status_code == 422


def test_unusable_images_never_reassure():
    from retilink.app.vision import VisionService
    svc = VisionService(model_dir="/nonexistent")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "white.jpg")
    Image.new("RGB", (800, 800), (240, 240, 240)).save(p)
    res = svc.analyze([{"id": 1, "path": p, "laterality": "OD"}, {"id": 2, "path": p, "laterality": "OS"}], {})
    assert res["overall"] == "Unable to assess" and not res["complete"]
    q = os.path.join(d, "f.jpg")
    open(q, "wb").write(fundus_bytes(2))
    one = svc.analyze([{"id": 3, "path": q, "laterality": "OD"}], {})
    assert one["overall"] != "No model finding" and not one["complete"]       # one eye only -> never reassuring


def test_gemini_guard_blocks_restricted_payload():
    assert guard({"question": "Which DR findings warrant referral?", "passages": []})
    for bad in ({"question": "patient_id 17 has DR", "passages": []}, {"question": "see 1003.1.jpg", "passages": []},
                {"question": "q", "passages": [], "case": 3}, {"question": "case #12 result", "passages": []}):
        with pytest.raises(RestrictedPayload):
            guard(bad)


def test_explanations_unavailable_in_simulated_mode(ready_case):
    pcp = client_as("Dr. Alex Morgan")
    with SessionLocal() as db:
        from retilink.app.models import Image as Img
        iid = db.query(Img).filter(Img.case_id == ready_case).first().id
    assert pcp.get(f"/images/{iid}/explain?head=dr_referable").status_code == 404     # never a fabricated map
    assert client_as("Dr. Riley Other").get(f"/images/{iid}/explain").status_code == 404   # tenant check first
    page = pcp.get(f"/cases/{ready_case}", params={"tab": "systemic"}).text
    assert "Not evaluated" in page and "Cardiovascular composite" in page


def test_oculomics_views_and_role_scoping(ready_case):
    pcp = client_as("Dr. Alex Morgan")
    panel = pcp.get("/oculomics")
    assert panel.status_code == 200 and "Organ by organ" in panel.text and "Whole-body" not in panel.text[:200]
    science = pcp.get("/oculomics?view=science")
    assert science.status_code == 200 and "Evidence, stated honestly" in science.text and "Future research" in science.text
    assert "not a diagnosis" in pcp.get(f"/cases/{ready_case}?tab=systemic").text.replace("never a diagnosis", "not a diagnosis")
    admin = client_as("Jordan Admin").get("/oculomics")
    assert admin.status_code == 200 and "Patients screened" in admin.text and "RL-T1" not in admin.text   # no clinical access
    other = client_as("Dr. Jamie Outside").get("/oculomics")
    assert "RL-T1" not in other.text                                                                    # other tenant


def test_inbox_shows_role_stats_and_activity(ready_case):
    page = client_as("Dr. Alex Morgan").get("/inbox").text
    assert "Results reviewed" in page and "Recent activity" in page and "Your last 14 days" in page
    assert "signed an interpretation for RL-T1" in page or "opened a screening for RL-T1" in page
    spec = client_as("Dr. Priya Nair").get("/inbox").text
    assert "Median time to accept" in spec and "RL-T2" not in spec          # specialists only see referred cases


def test_access_gate(monkeypatch):
    monkeypatch.setattr(main, "ACCESS_CODE", "open-sesame")
    c = TestClient(main.app, follow_redirects=False)
    assert c.get("/login").headers["location"].startswith("/access")
    assert c.get("/api/health").status_code == 200                      # worker health stays reachable
    assert "error=1" in c.post("/access", data={"code": "wrong", "next": "/login"}).headers["location"]
    r = c.post("/access", data={"code": "open-sesame", "next": "//evil.example"})
    assert r.headers["location"] == "/login"                            # no open redirect
    assert c.get("/login").status_code == 200
