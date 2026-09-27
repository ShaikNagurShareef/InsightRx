"""
Tamper-evidence on Solana (MLH "Best Use of Solana").

Every signed review and every sent consultation package already has a SHA-256 digest (workflow.review_signature,
workflow.digest). This module writes that digest, and nothing else, to the Solana Memo program, so anyone can later
prove the signed record existed unchanged at that time without trusting our database:

    memo = "insightrx:v1:<kind>:<sha256 hex>"        no patient data, no names, no ids that identify anyone

  SOLANA_ANCHOR_KEYPAIR   base58 64-byte secret key of the fee payer (devnet: fund it with `airdrop()`); unset = off
  SOLANA_RPC_URL          default https://api.devnet.solana.com
  SOLANA_CLUSTER          default devnet (explorer links)

Anchoring never blocks clinical work: failures are logged and the record is saved as usual.
"""
import base64
import logging
import os
import re

import httpx

MEMO_PROGRAM = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"
DEFAULT_RPC = "https://api.devnet.solana.com"
TIMEOUT_S = 6
KINDS = ("review", "referral")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TX = re.compile(r"\btx ([1-9A-HJ-NP-Za-km-z]{64,90})\b")
log = logging.getLogger("insightrx.anchor")


def cluster() -> str:
    return os.environ.get("SOLANA_CLUSTER", "devnet")


def enabled() -> bool:
    return bool(os.environ.get("SOLANA_ANCHOR_KEYPAIR"))


def memo_text(kind: str, digest: str) -> str:
    if kind not in KINDS or not _HEX64.match(digest or ""):
        raise ValueError("anchor memo takes a known kind and a sha256 hex digest")
    return f"insightrx:v1:{kind}:{digest}"


def explorer_url(tx: str) -> str:
    return f"https://explorer.solana.com/tx/{tx}?cluster={cluster()}"


def tx_in(detail: str) -> str | None:
    """The transaction signature recorded in an 'anchored_on_solana' audit detail."""
    m = _TX.search(detail or "")
    return m.group(1) if m else None


def _rpc(method: str, params: list, client: httpx.Client):
    r = client.post(os.environ.get("SOLANA_RPC_URL", DEFAULT_RPC),
                    json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, timeout=TIMEOUT_S)
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise RuntimeError(f"{method}: {body['error'].get('message', body['error'])}")
    return body["result"]


def build_transaction(memo: str, secret_b58: str, blockhash: str) -> str:
    """A Memo-program transaction signed by the fee payer, base64-encoded for sendTransaction."""
    from solders.hash import Hash
    from solders.instruction import AccountMeta, Instruction
    from solders.keypair import Keypair
    from solders.message import Message
    from solders.pubkey import Pubkey
    from solders.transaction import Transaction

    payer = Keypair.from_base58_string(secret_b58)
    ix = Instruction(Pubkey.from_string(MEMO_PROGRAM), memo.encode(), [AccountMeta(payer.pubkey(), True, False)])
    bh = Hash.from_string(blockhash)
    tx = Transaction([payer], Message.new_with_blockhash([ix], payer.pubkey(), bh), bh)
    return base64.b64encode(bytes(tx)).decode()


def anchor(kind: str, digest: str, client: httpx.Client | None = None) -> str | None:
    """Write the digest to Solana -> transaction signature, or None when disabled or unreachable."""
    if not enabled():
        return None
    memo = memo_text(kind, digest)
    own = client is None
    client = client or httpx.Client()
    try:
        blockhash = _rpc("getLatestBlockhash", [{"commitment": "confirmed"}], client)["value"]["blockhash"]
        raw = build_transaction(memo, os.environ["SOLANA_ANCHOR_KEYPAIR"], blockhash)
        return _rpc("sendTransaction", [raw, {"encoding": "base64", "preflightCommitment": "confirmed"}], client)
    except (httpx.HTTPError, RuntimeError, KeyError, ValueError, ImportError) as e:
        log.warning("solana anchor failed for %s: %s", kind, e)
        return None
    finally:
        if own:
            client.close()


def record(db, case, actor_id, kind: str, digest: str) -> str | None:
    """Anchor and append the proof to the case's audit trail."""
    from . import workflow as wf
    tx = anchor(kind, digest)
    if tx:
        wf.audit(db, case.tenant_id, case.id, actor_id, "anchored_on_solana",
                 f"{kind} sha256 {digest[:12]}… on Solana {cluster()}; tx {tx}", case.version)
    return tx


def airdrop(pubkey: str, lamports: int = 1_000_000_000) -> str:
    """Devnet only: fund the fee payer (one anchor costs 5,000 lamports)."""
    with httpx.Client() as c:
        return _rpc("requestAirdrop", [pubkey, lamports], c)
