"""Insight Rx Copilot (Backboard.io): de-identified briefs, per-clinician memory, transcript, memory controls, roles.
Uses a fake Backboard client; no network."""
import os
import types
import uuid

import pytest

from test_workflow import client_as, fundus_bytes, new_case, upload  # noqa: E402  (seeds a temp workspace)

from insightrx.app import copilot  # noqa: E402
from insightrx.app.llm import RestrictedPayload  # noqa: E402


class _Assistants(dict):
    def __missing__(self, key):                      # unknown ids behave like the real API
        from backboard import BackboardNotFoundError
        raise BackboardNotFoundError(f"assistant {key} not found")


class FakeBackboard:
    """In-memory stand-in for backboard.BackboardClient (async API, async context manager)."""
    state = None

    def __init__(self):
        s = FakeBackboard.state
        self.s = s

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def create_assistant(self, name, **kw):
        aid = str(uuid.uuid4())
        self.s["assistants"][aid] = {"name": name, "docs": [], "memories": [], **kw}
        return types.SimpleNamespace(assistant_id=aid)

    async def upload_document_to_assistant(self, aid, path):
        self.s["assistants"][aid]["docs"].append(open(path).read())
        return types.SimpleNamespace(document_id=str(uuid.uuid4()))

    async def get_document_status(self, doc_id):
        return types.SimpleNamespace(status="completed")

    async def clone_assistant(self, aid, name=None, copy_documents=True, copy_memories=True):
        src = self.s["assistants"][aid]
        new = str(uuid.uuid4())
        self.s["assistants"][new] = {"name": name, "docs": list(src["docs"]) if copy_documents else [],
                                     "memories": list(src["memories"]) if copy_memories else []}
        return types.SimpleNamespace(assistant=types.SimpleNamespace(assistant_id=new))

    async def create_thread(self, aid):
        tid = str(uuid.uuid4())
        self.s["threads"][tid] = aid
        return types.SimpleNamespace(thread_id=tid)

    async def add_message(self, thread_id, content, **kw):
        self.s["sent"].append(content)
        aid = self.s["threads"][thread_id]
        mems = [{"memory": m["content"]} for m in self.s["assistants"][aid]["memories"]]
        return types.SimpleNamespace(messages=[{"content": f"Answer to: {content[-40:]}", "retrieved_memories": mems,
                                                "retrieved_files": ["insightrx_interactions.md"], "model_name": "fake"}])

    async def add_memory(self, aid, content, metadata=None):
        self.s["assistants"][aid]["memories"].append({"id": str(uuid.uuid4()), "content": content})
        return {"ok": True}

    async def get_memories(self, aid, page=1, page_size=100):
        return types.SimpleNamespace(memories=[types.SimpleNamespace(id=m["id"], content=m["content"], created_at=None)
                                               for m in self.s["assistants"][aid]["memories"]])

    async def delete_memory(self, aid, mid):
        self.s["assistants"][aid]["memories"] = [m for m in self.s["assistants"][aid]["memories"] if m["id"] != mid]
        return {"ok": True}


@pytest.fixture
def fake_backboard(monkeypatch):
    FakeBackboard.state = {"assistants": _Assistants(), "threads": _Assistants(), "sent": []}   # fresh backend each test
    monkeypatch.setenv("BACKBOARD_API_KEY", "test-key")
    monkeypatch.setattr(copilot, "_client", lambda: FakeBackboard())
    return FakeBackboard.state


@pytest.fixture(scope="module")
def cp_case():
    op = client_as("Sam Rivera")
    cid = new_case(op, "RL-CP1")
    assert upload(op, cid, "OD", fundus_bytes(11)).status_code == 303
    assert op.post(f"/cases/{cid}/analyze").status_code == 303
    client_as("Dr. Alex Morgan").post(f"/cases/{cid}/intake", data={"age": "58", "med": ["semaglutide"], "tab": "systemic"})
    return cid


def test_outbound_guard_blocks_identifiers():
    for bad in ("What about RL-P0102?", "Seen on 09/26/2026", "see photo OD_1.jpg", "patient_id 44"):
        with pytest.raises(RestrictedPayload):
            copilot.check_outbound(bad)
    assert copilot.check_outbound("Adult in their 50s with diabetes")


def test_knowledge_base_is_curated_content_only():
    docs = dict(copilot.knowledge_docs())
    assert {"insightrx_therapy_classes.md", "insightrx_interactions.md", "insightrx_targets.md",
            "insightrx_cms_quality_billing.md", "insightrx_guideline_passages.md"} == set(docs)
    blob = " ".join(docs.values())
    assert "VEGFA" in blob and "CPT 92228" in blob and "RL-P" not in blob


def test_case_question_sends_only_deidentified_brief_and_saves_transcript(fake_backboard, cp_case):
    pcp = client_as("Dr. Alex Morgan")
    r = pcp.post(f"/cases/{cp_case}/copilot", data={"question": "What should I start first?"})
    assert r.status_code == 303 and r.headers["location"].endswith("#copilot")
    sent = fake_backboard["sent"][-1]
    assert "De-identified case brief" in sent and "semaglutide" in sent
    assert "RL-CP1" not in sent and "2026" not in sent and "58" not in sent and "50s" in sent
    page = pcp.get(f"/cases/{cp_case}/therapy").text
    assert "What should I start first?" in page and "Answer to:" in page


def test_each_clinician_gets_private_memory(fake_backboard, cp_case):
    a, b = client_as("Dr. Alex Morgan"), client_as("Dr. Priya Nair")
    assert a.post("/copilot/remember", data={"text": "Prefers dapagliflozin first when eGFR is borderline"}).status_code == 303
    assert "dapagliflozin first when eGFR" in a.get("/copilot").text
    assert "dapagliflozin first when eGFR" not in b.get("/copilot").text          # another clinician's clone
    r = a.post("/copilot/ask", data={"question": "Which class first in kidney disease?"})
    assert r.status_code == 303
    assert "Used what it remembers about you" in a.get("/copilot").text
    kb = [x for x in fake_backboard["assistants"].values() if x["name"] == copilot.BASE_NAME]
    clones = [x for x in fake_backboard["assistants"].values() if x["name"].startswith("Insight Rx Copilot (clinician")]
    assert len(kb) == 1 and len(clones) == 2 and all(len(c["docs"]) == 5 for c in clones)


def test_forget_memory(fake_backboard):
    a = client_as("Dr. Alex Morgan")
    a.post("/copilot/remember", data={"text": "Refer DME to Dr. Nair within two weeks"})
    aid = [k for k, v in fake_backboard["assistants"].items() if v["memories"]][0]
    mid = fake_backboard["assistants"][aid]["memories"][0]["id"]
    assert a.post(f"/copilot/memories/{mid}/delete").status_code == 303
    assert "Refer DME to Dr. Nair" not in a.get("/copilot").text


def test_identifying_question_is_refused(fake_backboard, cp_case):
    r = client_as("Dr. Alex Morgan").post(f"/cases/{cp_case}/copilot", data={"question": "Compare with RL-P0102"})
    assert r.status_code == 303 and "identifying" in r.headers["location"]
    assert not any("RL-P0102" in s for s in fake_backboard["sent"])


def test_roles_and_disconnected_state(monkeypatch, cp_case):
    monkeypatch.delenv("BACKBOARD_API_KEY", raising=False)
    assert "not connected" in client_as("Dr. Alex Morgan").get("/copilot").text
    r = client_as("Dr. Alex Morgan").post("/copilot/ask", data={"question": "hello there"})
    assert r.status_code == 303 and "not+connected" in r.headers["location"].replace("%20", "+")
    assert client_as("Morgan Lee, PharmD").post("/copilot/ask", data={"question": "hi"}).status_code == 403
    assert client_as("Morgan Lee, PharmD").post(f"/cases/{cp_case}/copilot", data={"question": "hi"}).status_code in (403, 404)
    assert os.environ.get("BACKBOARD_API_KEY") is None


def test_billing_notice_is_reported_as_unavailable():
    from insightrx.app import copilot
    assert copilot.BILLING_NOTICE.search("Your free credit is reserved for Memory & RAG, so it can't cover LLM chat.")
    assert not copilot.BILLING_NOTICE.search("Semaglutide needs a retinal exam before starting in patients with DR.")
