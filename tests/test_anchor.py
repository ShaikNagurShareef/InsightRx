"""Solana anchoring: only a digest leaves, the transaction is a valid signed Memo, failures never block signing."""
import base64
import json

import httpx
import pytest
from solders.keypair import Keypair
from solders.transaction import Transaction

from insightrx.app import anchor

DIGEST = "ab" * 32
BLOCKHASH = "EkSnNWid2cvwEVnVx9aBqawnmiCNiDgp3gUdkDPTKN1N"
FAKE_TX = "5" * 88


def fake_rpc(sent: list, fail: bool = False) -> httpx.Client:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if fail:
            return httpx.Response(503)
        if body["method"] == "getLatestBlockhash":
            return httpx.Response(200, json={"result": {"value": {"blockhash": BLOCKHASH}}})
        sent.append(body["params"][0])
        return httpx.Response(200, json={"result": FAKE_TX})
    return httpx.Client(transport=httpx.MockTransport(handle))


@pytest.fixture
def payer(monkeypatch):
    kp = Keypair()
    monkeypatch.setenv("SOLANA_ANCHOR_KEYPAIR", str(kp))
    return kp


def test_memo_carries_only_kind_and_digest():
    assert anchor.memo_text("review", DIGEST) == f"insightrx:v1:review:{DIGEST}"
    for kind, dg in (("patient", DIGEST), ("review", "RL-0001 Maria"), ("review", DIGEST[:-1])):
        with pytest.raises(ValueError):
            anchor.memo_text(kind, dg)


def test_anchor_sends_a_signed_memo_transaction(payer):
    sent = []
    assert anchor.anchor("referral", DIGEST, fake_rpc(sent)) == FAKE_TX
    tx = Transaction.from_bytes(base64.b64decode(sent[0]))
    tx.verify()                                                  # signature valid for the fee payer
    assert tx.message.account_keys[0] == payer.pubkey()
    ix = tx.message.instructions[0]
    assert str(tx.message.account_keys[ix.program_id_index]) == anchor.MEMO_PROGRAM
    assert bytes(ix.data) == f"insightrx:v1:referral:{DIGEST}".encode()


def test_disabled_without_a_keypair(monkeypatch):
    monkeypatch.delenv("SOLANA_ANCHOR_KEYPAIR", raising=False)
    sent = []
    assert anchor.anchor("review", DIGEST, fake_rpc(sent)) is None and sent == []


def test_unreachable_rpc_degrades_to_none(payer):
    assert anchor.anchor("review", DIGEST, fake_rpc([], fail=True)) is None


def test_explorer_link_round_trip(monkeypatch):
    monkeypatch.delenv("SOLANA_CLUSTER", raising=False)
    detail = f"review sha256 {DIGEST[:12]}… on Solana devnet; tx {FAKE_TX}"
    assert anchor.tx_in(detail) == FAKE_TX and anchor.tx_in("sig abc") is None
    assert anchor.explorer_url(FAKE_TX).endswith(f"/tx/{FAKE_TX}?cluster=devnet")


def test_signing_a_review_records_the_anchor_on_the_timeline(payer, monkeypatch):
    from test_workflow import client_as, fundus_bytes, new_case, sign, upload  # seeds a temp workspace
    monkeypatch.setattr(anchor, "anchor", lambda kind, digest, client=None: FAKE_TX)
    op, pcp = client_as("Sam Rivera"), client_as("Dr. Alex Morgan")
    cid = new_case(op, "RL-ANCHOR1")
    upload(op, cid, "OD", fundus_bytes(31))
    op.post(f"/cases/{cid}/analyze")
    assert sign(pcp, cid).status_code == 303
    page = pcp.get(f"/cases/{cid}?tab=timeline").text
    assert "Anchored on solana" in page and f"/tx/{FAKE_TX}?cluster=devnet" in page
