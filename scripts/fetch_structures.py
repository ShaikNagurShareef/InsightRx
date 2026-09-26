"""
Fetch the static reference data behind the Therapeutics layer (run once; outputs are committed):

  public/static/structures/AF-<uniprot>.pdb      AlphaFold DB models (CC-BY 4.0), B-factor = pLDDT
  public/static/structures/<pdb>.pdb             experimental drug complexes (RCSB PDB)
  public/static/vendor/3Dmol-min.js              3Dmol.js viewer (BSD-3)
  retilink/app/data/chembl_drugs.json            drugs acting on each target (ChEMBL mechanisms)
  retilink/app/data/trials_snapshot.json         recruiting trials per topic (ClinicalTrials.gov v2), offline fallback
  retilink/app/data/nppes_snapshot.json          NPI Registry providers per taxonomy (CMS NPPES), offline fallback

Usage: python scripts/fetch_structures.py [--skip-structures | --snapshots-only]
"""
import json
import os
import sys

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from retilink.app import external  # noqa: E402

STRUCT = os.path.join(ROOT, "public", "static", "structures")
VENDOR = os.path.join(ROOT, "public", "static", "vendor")
DATA = os.path.join(ROOT, "retilink", "app", "data")
THREEDMOL = "https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"
MAX_DRUGS = 10


def get(client, url, **kw):
    r = client.get(url, timeout=60, follow_redirects=True, **kw)
    r.raise_for_status()
    return r


def keep_atoms(pdb_text: str, chains=None) -> str:
    """Coordinates only (ATOM/HETATM/TER/END), optionally only some chains: smaller files, same rendering."""
    lines = [ln for ln in pdb_text.splitlines() if ln.startswith(("ATOM", "HETATM", "TER", "END"))
             and (not chains or ln.startswith("END") or ln[21:22] in chains)]
    return "\n".join(lines) + "\n"


def fetch_structures(client, targets):
    os.makedirs(STRUCT, exist_ok=True)
    for t in targets:
        dst = os.path.join(STRUCT, f"AF-{t['uniprot']}.pdb")
        if not os.path.exists(dst):
            meta = get(client, f"https://alphafold.ebi.ac.uk/api/prediction/{t['uniprot']}").json()[0]
            open(dst, "w").write(keep_atoms(get(client, meta["pdbUrl"]).text))
        print(f"{t['gene']:8s} AlphaFold {os.path.getsize(dst) // 1024} KB")
        cx = t.get("complex")
        if cx:
            dst = os.path.join(STRUCT, f"{cx['pdb']}.pdb")
            if not os.path.exists(dst):
                open(dst, "w").write(keep_atoms(get(client, f"https://files.rcsb.org/download/{cx['pdb']}.pdb").text,
                                                   cx.get("keep_chains")))
            print(f"{'':8s} PDB {cx['pdb']} {os.path.getsize(dst) // 1024} KB")


def chembl_drugs(client, targets):
    """Drugs with a recorded mechanism on each target (ChEMBL), approved first."""
    base = "https://www.ebi.ac.uk/chembl/api/data"
    out = {}
    for t in targets:
        tids = [x["target_chembl_id"] for x in get(client, f"{base}/target.json", params={
            "target_components__accession": t["uniprot"], "target_type": "SINGLE PROTEIN", "limit": 5}).json()["targets"]]
        drugs = {}
        for tid in tids[:1]:
            mechs = get(client, f"{base}/mechanism.json", params={"target_chembl_id": tid, "limit": 200}).json()["mechanisms"]
            for m in mechs:
                mid = m["molecule_chembl_id"]
                if mid in drugs:
                    continue
                mol = get(client, f"{base}/molecule/{mid}.json").json()
                name = (mol.get("pref_name") or "").lower()
                if not name:
                    continue
                drugs[mid] = {"chembl_id": mid, "name": name, "max_phase": float(mol.get("max_phase") or 0),
                              "action": (m.get("action_type") or "").lower(), "mechanism": m.get("mechanism_of_action", ""),
                              "type": (mol.get("molecule_type") or "").lower()}
        ranked = sorted(drugs.values(), key=lambda d: (-d["max_phase"], d["name"]))
        out[t["gene"]] = {"target_chembl_id": tids[0] if tids else None, "drugs": ranked[:MAX_DRUGS]}
        print(f"{t['gene']:8s} ChEMBL {len(drugs)} drugs -> kept {len(out[t['gene']]['drugs'])}")
    json.dump(out, open(os.path.join(DATA, "chembl_drugs.json"), "w"), indent=1)


def snapshots():
    trials = {topic: external.trials_live(topic, external.DEFAULT_ZIP) for topic in external.TRIAL_QUERIES}
    json.dump(trials, open(os.path.join(DATA, "trials_snapshot.json"), "w"), indent=1)
    print("trials snapshot:", {k: len(v) for k, v in trials.items()})
    nppes = {tax: external.nppes_live(tax, external.DEFAULT_ZIP) for tax in external.TAXONOMIES}
    json.dump(nppes, open(os.path.join(DATA, "nppes_snapshot.json"), "w"), indent=1)
    print("nppes snapshot:", {k: len(v) for k, v in nppes.items()})


def main():
    targets = json.load(open(os.path.join(DATA, "targets.json")))["targets"]
    with httpx.Client(headers={"User-Agent": "RetiLink-hackathon/1.0"}) as client:
        if not {"--skip-structures", "--snapshots-only"} & set(sys.argv):
            fetch_structures(client, targets)
            os.makedirs(VENDOR, exist_ok=True)
            open(os.path.join(VENDOR, "3Dmol-min.js"), "wb").write(get(client, THREEDMOL).content)
        if "--snapshots-only" not in sys.argv:
            chembl_drugs(client, targets)
    snapshots()


if __name__ == "__main__":
    main()
