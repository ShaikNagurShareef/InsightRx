"""Queued analyses on a slow remote worker (e.g. a free Hugging Face CPU Space): Screen queues and polls, saving a
screened patient reuses its result, case analyses settle when viewed, and lost jobs are resubmitted."""
import pytest

from test_workflow import client_as, fundus_bytes, new_case, upload  # noqa: E402  (seeds a temp workspace)

from insightrx.app import vision  # noqa: E402
from insightrx.app.db import SessionLocal  # noqa: E402
from insightrx.app.models import Image, ModelRun  # noqa: E402


class FakeAsyncWorker:
    """Behaves like RemoteVisionService with INSIGHTRX_VISION_ASYNC=1; results come from the SIMULATED service."""
    is_async, backend, mode, version, metrics, calib = True, "remote", "live", "fake-remote-v1", {}, None

    def __init__(self):
        self.sim = vision.VisionService(model_dir="/nonexistent")
        self.jobs, self.submitted, self.ready = {}, 0, False

    def submit(self, images, patient):
        self.submitted += 1
        jid = f"job{self.submitted}"
        self.jobs[jid] = (images, patient)
        return jid

    def job(self, jid):
        if jid not in self.jobs:
            return {"status": "lost"}
        if not self.ready:
            return {"status": "queued", "position": 1, "eta_s": 120}
        images, patient = self.jobs[jid]
        return {"status": "done", "result": self.sim.analyze(images, patient)}

    def analyze(self, images, patient):
        raise AssertionError("async worker must not be called synchronously")

    def explain(self, *a, **k):
        return None


@pytest.fixture
def worker(monkeypatch):
    w = FakeAsyncWorker()
    monkeypatch.setattr(vision, "_service", w)
    return w


def test_screen_queues_then_renders_and_save_reuses_the_result(worker):
    pcp = client_as("Dr. Alex Morgan")
    r = pcp.post("/screen", data={"age": "61"}, files=[("files_OD", ("od.jpg", fundus_bytes(21), "image/jpeg")),
                                                       ("files_OS", ("os.jpg", fundus_bytes(22), "image/jpeg"))])
    assert r.status_code == 303 and r.headers["location"].startswith("/screen/result?token=")
    url = r.headers["location"]
    pending = pcp.get(url)
    assert pending.status_code == 200 and "Analysing 2 photos" in pending.text and 'http-equiv="refresh"' in pending.text
    worker.ready = True
    done = pcp.get(url)
    assert "Screening result" in done.text and "Personalised therapeutics" in done.text
    token = url.split("token=")[1]
    s = pcp.post("/screen/save", data={"token": token, "patient_ref": "RL-ASYNC1", "age": "61"})
    assert s.status_code == 303 and worker.submitted == 1          # no second analysis
    cid = int(s.headers["location"].split("/cases/")[1].split("/")[0])
    with SessionLocal() as db:
        run = db.query(ModelRun).filter(ModelRun.case_id == cid).one()
        ids = {str(i.id) for i in db.query(Image).filter(Image.case_id == cid)}
        assert run.status == "completed" and set(run.result["images"]) == ids   # re-keyed to saved image ids


def test_case_analysis_is_queued_and_settles_when_viewed(worker):
    op, pcp = client_as("Sam Rivera"), client_as("Dr. Alex Morgan")
    cid = new_case(op, "RL-ASYNC2")
    assert upload(op, cid, "OD", fundus_bytes(23)).status_code == 303
    assert op.post(f"/cases/{cid}/analyze").status_code == 303
    page = pcp.get(f"/cases/{cid}")
    assert "Analysing on the Insight Rx vision service" in page.text and 'http-equiv="refresh"' in page.text
    worker.ready = True
    page = pcp.get(f"/cases/{cid}")
    assert "Analysing on the Insight Rx vision service" not in page.text
    with SessionLocal() as db:
        run = db.query(ModelRun).filter(ModelRun.case_id == cid).order_by(ModelRun.id.desc()).first()
        assert run.status == "completed" and run.result.get("overall")


def test_lost_job_is_resubmitted(worker):
    op = client_as("Sam Rivera")
    cid = new_case(op, "RL-ASYNC3")
    upload(op, cid, "OD", fundus_bytes(24))
    op.post(f"/cases/{cid}/analyze")
    before = worker.submitted
    worker.jobs.clear()                                   # worker restarted: it no longer knows the job
    client_as("Dr. Alex Morgan").get(f"/cases/{cid}")
    assert worker.submitted == before + 1
