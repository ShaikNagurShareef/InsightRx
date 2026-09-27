"""
Insight Rx Copilot: an evidence-grounded assistant with persistent memory for each clinician, built on Backboard.io.

- Knowledge (RAG): Insight Rx's curated catalogues (guideline therapy classes, interaction rules, protein targets with
  structure insights, CMS quality and billing rules, guideline passages) are uploaded once to a base assistant.
- Memory: each clinician gets a private clone of that assistant, so Backboard's persistent memory learns *their*
  preferences, practice patterns and follow-up intentions across sessions and patients, never mixing clinicians.
- Threads: one conversation thread per clinician per case (or a general thread), so context carries across visits.
- Privacy: only a de-identified brief reaches Backboard (age band, finding labels, medicines, therapy options, alerts,
  ranked targets). It is checked against the same restricted-pattern guard as the Gemini path before sending.

Answering: with GEMINI_API_KEY set, Backboard runs memory only (send_to_llm=false: it extracts and recalls this
clinician's memories), the matching knowledge passages are retrieved locally from the same documents, and Gemini writes a
cited answer. Without a Gemini key, Backboard's own LLM answers (needs Backboard LLM credits).

Configure with BACKBOARD_API_KEY (and optionally BACKBOARD_LLM_PROVIDER / BACKBOARD_MODEL). Without it the copilot
reports itself as unavailable and nothing is sent anywhere.
"""
import asyncio
import functools
import os
import re
import tempfile

from sqlalchemy import select

from . import therapeutics as tx
from .llm import FORBIDDEN, GeminiUnavailable, RestrictedPayload, gemini_generate, gemini_key
from .models import AppSetting, CopilotMessage, now

PROVIDER = os.environ.get("BACKBOARD_LLM_PROVIDER", "google")
MODEL = os.environ.get("BACKBOARD_MODEL", "gemini-2.5-flash")
BASE_NAME = "Insight Rx Copilot (knowledge base)"
KNOWLEDGE_VERSION = "kb-v1"

SYSTEM_PROMPT = """You are Insight Rx Copilot, decision support for clinicians who screen people with diabetes using
retinal photographs. Answer from the Insight Rx knowledge documents (guideline therapy classes, interaction rules,
protein targets and structure insights, CMS quality and billing rules, guideline passages) and name the source document
or guideline you relied on. Use the de-identified case brief when one is given.
Rules: you support the clinician's decision and never replace it; do not diagnose or prescribe, and do not give doses
beyond what the cited label or guideline states; say plainly when the knowledge base does not cover a question;
flag interaction alerts first; keep answers under 180 words unless asked for more; use plain clinical English.
Use what you remember about this clinician (their stated preferences, practice patterns and follow-up intentions)
to tailor the answer, and say when you are doing so."""

FACT_PROMPT = """Extract only durable facts about the CLINICIAN: stated treatment preferences and practice patterns,
preferred specialists or referral habits, follow-up intentions, and topics they care about (for example
"prefers SGLT2 inhibitors before GLP-1 receptor agonists in kidney disease"). Never extract patient details:
no ages, dates, identifiers, image descriptions or individual results."""


class CopilotUnavailable(Exception):
    pass


BILLING_NOTICE = re.compile(r"free credit is reserved|Billing page|add credits", re.I)
TOP_PASSAGES = 6
_WORD = re.compile(r"[a-z0-9][a-z0-9-]+")
STOP = frozenset("the and for with what when which does how are is of to in on a an or be it this that should do i my".split())


def enabled():
    return bool(os.environ.get("BACKBOARD_API_KEY"))


CASE_ROLES = ("referring", "specialist")                                  # patient-level copilot (case owners)
GENERAL_ROLES = CASE_ROLES + ("operator", "coordinator", "admin")         # never the medical-information desk


def can_use(user, scope="general"):
    """RBAC for the copilot: one rule for access and for visibility (links, pages, panels)."""
    return bool(user) and enabled() and user.role in (CASE_ROLES if scope == "case" else GENERAL_ROLES)


def _client():
    from backboard import BackboardClient
    key = os.environ.get("BACKBOARD_API_KEY")
    if not key:
        raise CopilotUnavailable("Copilot is not connected (BACKBOARD_API_KEY is not set).")
    return BackboardClient(api_key=key, timeout=60)


def _run(coro):
    return asyncio.run(coro)


def _with_recovery(db, user, make):
    """Run make() (a coroutine factory); if Backboard no longer knows a cached assistant or thread (account reset,
    deleted assistant), drop this clinician's cached bindings and the base binding, then retry once."""
    from backboard import BackboardNotFoundError
    try:
        return _run(make())
    except BackboardNotFoundError:
        for s in db.scalars(select(AppSetting).where(AppSetting.key.like("backboard:%"))).all():
            if s.key.startswith((f"backboard:user:{user.id}:", f"backboard:thread:{user.id}:", "backboard:base:")):
                db.delete(s)
        db.commit()
        return _run(make())


# ------------------------------------------------------------------ knowledge base (public, curated content only)
def knowledge_docs():
    """[(filename, markdown)] built from the curated catalogues. No patient data."""
    from .evidence import REFS
    cat, ix, cms = tx.catalogue(), tx.interactions_db(), tx.cms()
    tiers = cat["tiers"]
    out = []
    lines = ["# Insight Rx guideline therapy classes", "", "Classes a clinician may consider for each retinal or systemic "
             "finding, ranked by guideline strength (sponsorship never enters the ranking).", ""]
    for c in cat["classes"]:
        lines += [f"## {c['label']}", f"- Findings: {', '.join(cat['findings'][k]['label'] for k in c['findings'])}",
                  f"- Evidence tier: {tiers[c['tier']]['label']}", f"- Example drugs: {', '.join(c['drugs']) or 'none'}",
                  f"- Protein targets: {', '.join(c['targets']) or 'none'}", f"- Summary: {c['summary']}",
                  f"- Source: {c['source']}", ""]
    out.append(("insightrx_therapy_classes.md", "\n".join(lines)))
    lines = ["# Insight Rx interaction and drug-disease rules", "", ix["_about"], ""]
    for r in ix["pairs"]:
        lines += [f"## {r['title']} ({r['severity']})", f"- Between: {', '.join(r['a'])} and {', '.join(r['b'])}",
                  f"- {r['detail']}", f"- Source: {r['source']}", ""]
    for r in ix["disease"]:
        lines += [f"## {r['title']} ({r['severity']})",
                  f"- Drug classes {', '.join(r['class'])} with finding '{cat['findings'][r['finding']]['label']}'",
                  f"- {r['detail']}", f"- Source: {r['source']}", ""]
    lines += ["## Drug to class map", ", ".join(f"{d}: {c}" for d, c in sorted(ix["drug_classes"].items()))]
    out.append(("insightrx_interactions.md", "\n".join(lines)))
    from .personalize import structure_insights
    si = structure_insights()
    lines = ["# Insight Rx protein targets linked to retinal findings", ""]
    for t0 in tx.targets():
        t = tx.target(t0["gene"])
        s = si.get(t["gene"]) or {}
        pocket = s.get("pocket") or {}
        approved = [d["name"] for d in t["drugs"] if d["phase"] == "Approved"]
        pipeline = [f"{d['name']} ({d['phase']})" for d in t["drugs"] if d["phase"] != "Approved"]
        lines += [f"## {t['gene']}: {t['name']} (UniProt {t['uniprot']})", f"- Role: {t['role']}",
                  f"- Why the eye points here: {t['eye_link']}",
                  f"- Linked findings: {', '.join(cat['findings'][k]['label'] for k in t['findings'])}",
                  f"- Approved drugs (ChEMBL): {', '.join(approved) or 'none recorded'}",
                  f"- Pipeline drugs: {', '.join(pipeline) or 'none recorded'}"]
        if t.get("plddt"):
            lines.append(f"- AlphaFold model: mean pLDDT {t['plddt']['mean']:.0f}, {t['plddt']['confident'] * 100:.0f}% of "
                         f"residues confident")
        if pocket.get("residues"):
            lines.append(f"- Drug-contact site: {len(pocket['residues'])} residues within {pocket['cutoff_A']} A of the "
                         f"{pocket['drug']} in PDB {pocket['pdb']}; AlphaFold confidence there {pocket['mean_plddt']:.0f}")
        lines.append("")
    out.append(("insightrx_targets.md", "\n".join(lines)))
    lines = ["# CMS quality measures and billing for diabetic retinal screening", "", cms["_about"], ""]
    for m in cms["measures"]:
        lines += [f"## {m['id']}: {m['name']} ({m['owner']})", f"- {m['what']}", f"- With Insight Rx: {m['how']}", ""]
    for c in cms["codes"]:
        lines += [f"## CPT {c['code']}: {c['name']}", f"- Fit: {c['fit']}", ""]
    lines += ["## ICD-10 coding hints"] + [f"- {d['code']}: {d['when']}" for d in cms["icd10"]]
    out.append(("insightrx_cms_quality_billing.md", "\n".join(lines)))
    refs = REFS if isinstance(REFS, list) else REFS.get("references", [])
    lines = ["# Guideline passages (curated evidence set)", ""]
    for r in refs:
        lines += [f"## {r['title']}", f"- Publisher: {r['publisher']}, {r['version']}", f"- {r['passage']}",
                  f"- URL: {r['url']}", ""]
    out.append(("insightrx_guideline_passages.md", "\n".join(lines)))
    return out


# ------------------------------------------------------------------ bindings (assistant per clinician, thread per case)
def _get(db, key):
    s = db.get(AppSetting, key)
    return s.value if s else None


def _put(db, key, value):
    s = db.get(AppSetting, key) or AppSetting(key=key, value="")
    s.value = value
    db.merge(s)
    db.commit()


async def _ensure_base(client, db):
    key = f"backboard:base:{KNOWLEDGE_VERSION}"
    aid = _get(db, key)
    if aid:
        return aid
    a = await client.create_assistant(name=BASE_NAME, system_prompt=SYSTEM_PROMPT, tok_k=12,
                                      custom_fact_extraction_prompt=FACT_PROMPT)
    with tempfile.TemporaryDirectory() as d:
        for name, text in knowledge_docs():
            path = os.path.join(d, name)
            with open(path, "w") as fh:
                fh.write(text)
            doc = await client.upload_document_to_assistant(a.assistant_id, path)
            for _ in range(60):                               # wait until indexed so clones copy the embeddings
                st = await client.get_document_status(doc.document_id)
                if str(getattr(st.status, "value", st.status)).lower().rsplit(".", 1)[-1] in (
                        "completed", "indexed", "ready", "processed", "failed", "error"):
                    break
                await asyncio.sleep(2)
    _put(db, key, str(a.assistant_id))
    return str(a.assistant_id)


async def _ensure_user(client, db, user):
    key = f"backboard:user:{user.id}:{KNOWLEDGE_VERSION}"
    aid = _get(db, key)
    if aid:
        return aid
    base = await _ensure_base(client, db)
    clone = await client.clone_assistant(base, name=f"Insight Rx Copilot (clinician {user.id})", copy_documents=True,
                                         copy_memories=False)
    aid = str(clone.assistant.assistant_id)
    _put(db, key, aid)
    return aid


async def _ensure_thread(client, db, user, scope):
    key = f"backboard:thread:{user.id}:{scope}"
    tid = _get(db, key)
    if tid:
        return tid
    aid = await _ensure_user(client, db, user)
    t = await client.create_thread(aid)
    _put(db, key, str(t.thread_id))
    return str(t.thread_id)


# ------------------------------------------------------------------ de-identified case brief
def check_outbound(text):
    for p in FORBIDDEN:
        if p.search(text):
            raise RestrictedPayload(f"copilot payload matches restricted pattern {p.pattern}")
    return text


def case_brief(bundle, res, age, sex):
    """What the copilot may know about a case: no reference, dates, images or identifiers."""
    fnd = [f"{f['label']} ({f['strength'].lower()})" for f in bundle["fnd"] if f["key"] != "metabolism"]
    alerts = {a["title"]: a for o in bundle["options"] for a in o["alerts"]}
    for a in bundle["current_alerts"]:
        alerts.setdefault(a["title"], a)
    lines = [f"Patient: {tx.age_band(age)}{', ' + sex if sex else ''} with diabetes.",
             f"Retina: {(res or {}).get('overall') or 'no current analysis'}.",
             "Findings: " + ("; ".join(fnd) or "none beyond diabetes") + ".",
             "Current medicines: " + (", ".join(bundle["meds"]) or "none recorded") + ".",
             "Guideline therapy options: " + "; ".join(f"{o['label']} ({o['tier_label']})" for o in bundle["options"]) + ".",
             "Interaction alerts: " + ("; ".join(f"{a['title']} [{a['severity']}] involving {', '.join(a['drugs'])}"
                                                 for a in alerts.values()) or "none") + ".",
             "Top protein targets: " + "; ".join(f"{p['gene']} priority {p['priority']}" for p in bundle["priorities"][:5]) + "."]
    return check_outbound(" ".join(lines))


# ------------------------------------------------------------------ local retrieval over the same knowledge documents
@functools.lru_cache(maxsize=1)
def passages():
    """[(doc, heading, text)]: each '## ' section of the knowledge documents."""
    out = []
    for name, text in knowledge_docs():
        for block in text.split("\n## ")[1:]:
            head, _, body = block.partition("\n")
            out.append((name, head.strip(), body.strip()))
    return out


def _terms(text):
    return {w for w in _WORD.findall(text.lower()) if w not in STOP}


def retrieve(query, k=TOP_PASSAGES):
    q = _terms(query)
    scored = []
    for doc, head, body in passages():
        h, b = _terms(head), _terms(body)
        score = 3 * len(q & h) + len(q & b)
        if score:
            scored.append((score, doc, head, body))
    return [(doc, head, body) for _, doc, head, body in sorted(scored, key=lambda x: -x[0])[:k]]


def _gemini_answer(question, brief, mems):
    found = retrieve(f"{question} {brief or ''}")
    ctx = "\n\n".join(f"[{n + 1}] ({doc.removesuffix('.md').replace('_', ' ')}) {head}\n{body[:900]}"
                       for n, (doc, head, body) in enumerate(found)) or "(no matching passage)"
    remembered = "\n".join(f"- {m}" for m in mems) or "- (nothing yet)"
    prompt = (f"Knowledge passages:\n{ctx}\n\nWhat you remember about this clinician:\n{remembered}\n\n"
              + (f"De-identified case brief: {brief}\n\n" if brief else "")
              + f"Clinician question: {question}\n\nCite passages as [n]. If the passages do not cover it, say so.")
    text, model = gemini_generate(prompt, system=SYSTEM_PROMPT)
    return text, sorted({doc for doc, _, _ in found}), model


# ------------------------------------------------------------------ public API
def ask(db, user, question, scope="general", brief=None, case_id=None):
    """Ask the copilot. Returns {'answer', 'memories', 'files', 'model'}; stores both turns locally for display."""
    question = (question or "").strip()[:1500]
    if not question:
        raise ValueError("empty question")
    check_outbound(question)
    content = (f"De-identified case brief: {brief}\n\nClinician question: {question}" if brief else question)
    own_llm = bool(gemini_key())

    async def go():
        async with _client() as client:
            tid = await _ensure_thread(client, db, user, scope)
            if own_llm:                                      # Backboard: memory extraction + recall only
                return await client.add_message(thread_id=tid, content=content, memory="Auto", memory_citation=True,
                                                send_to_llm="false", stream=False)
            return await client.add_message(thread_id=tid, content=content, llm_provider=PROVIDER, model_name=MODEL,
                                            memory="Auto", memory_citation=True, stream=False)
    r = _with_recovery(db, user, go)
    m = (r.messages or [{}])[-1] if hasattr(r, "messages") else r.__dict__
    memories = [x.get("memory") or x.get("content") or "" for x in (m.get("retrieved_memories") or [])]
    memories = [x for x in memories if x]
    if own_llm:
        try:
            answer, files, model = _gemini_answer(question, brief, memories)
        except GeminiUnavailable as e:
            raise CopilotUnavailable("Gemini is busy right now; your question and memories were saved. Try again.") from e
        model = f"{model} + Backboard memory"
    else:
        answer = m.get("content") or m.get("message") or ""
        files = list(m.get("retrieved_files") or [])
        model = m.get("model_name") or MODEL
        if BILLING_NOTICE.search(answer):                  # Backboard answers with a billing notice, not an error
            raise CopilotUnavailable("The copilot's Backboard account needs LLM credits (memory and retrieval are "
                                     "working). Add credits in Backboard or set GEMINI_API_KEY, then try again.")
    db.add(CopilotMessage(user_id=user.id, case_id=case_id, scope=scope, role="clinician", content=question))
    db.add(CopilotMessage(user_id=user.id, case_id=case_id, scope=scope, role="copilot", content=answer,
                          meta={"memories": memories, "files": files, "model": model}))
    db.commit()
    return {"answer": answer, "memories": memories, "files": files, "model": model}


def remember(db, user, text):
    text = check_outbound((text or "").strip()[:500])
    if not text:
        raise ValueError("empty memory")

    async def go():
        async with _client() as client:
            aid = await _ensure_user(client, db, user)
            return await client.add_memory(aid, text, metadata={"source": "clinician", "at": now().isoformat()})
    return _with_recovery(db, user, go)


def memories(db, user):
    async def go():
        async with _client() as client:
            aid = await _ensure_user(client, db, user)
            r = await client.get_memories(aid, page=1, page_size=100)
            return [{"id": m.id, "content": m.content, "created_at": m.created_at} for m in r.memories]
    return _with_recovery(db, user, go)


def forget(db, user, memory_id):
    async def go():
        async with _client() as client:
            aid = await _ensure_user(client, db, user)
            return await client.delete_memory(aid, memory_id)
    return _with_recovery(db, user, go)


def transcript(db, user, scope, limit=20):
    rows = db.scalars(select(CopilotMessage).where(CopilotMessage.user_id == user.id, CopilotMessage.scope == scope)
                      .order_by(CopilotMessage.id.desc()).limit(limit)).all()
    rows = list(reversed(rows))
    bad = {i for i, r in enumerate(rows) if r.role == "copilot" and BILLING_NOTICE.search(r.content or "")}
    return [r for i, r in enumerate(rows) if i not in bad and i + 1 not in bad]   # drop billing notices + their question


def recent_scopes(db, user, limit=8):
    rows = db.scalars(select(CopilotMessage).where(CopilotMessage.user_id == user.id, CopilotMessage.role == "clinician")
                      .order_by(CopilotMessage.id.desc()).limit(50)).all()
    seen, out = set(), []
    for r in rows:
        if r.scope not in seen:
            seen.add(r.scope)
            out.append(r)
    return out[:limit]



