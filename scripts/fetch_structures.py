"""
Fetch the static reference data behind the Therapeutics layer (run once; outputs are committed):

  public/static/structures/AF-<uniprot>.pdb      AlphaFold DB models (CC-BY 4.0), B-factor = pLDDT
  public/static/structures/<pdb>.pdb             experimental drug complexes (RCSB PDB)
  public/static/vendor/3Dmol-min.js              3Dmol.js viewer (BSD-3)
  insightrx/app/data/chembl_drugs.json            drugs acting on each target (ChEMBL mechanisms)
  insightrx/app/data/trials_snapshot.json         recruiting trials per topic (ClinicalTrials.gov v2), offline fallback
  insightrx/app/data/nppes_snapshot.json          NPI Registry providers per taxonomy (CMS NPPES), offline fallback
  insightrx/app/data/structure_insights.json      per-target structure analysis: pLDDT per residue, disordered regions,
                                                 UniProt domains/binding sites, drug-contact (pocket/epitope) residues

Usage: python scripts/fetch_structures.py [--skip-structures | --snapshots-only | --insights-only]
"""
import json
import os
import sys

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from insightrx.app import external  # noqa: E402

STRUCT = os.path.join(ROOT, "public", "static", "structures")
VENDOR = os.path.join(ROOT, "public", "static", "vendor")
DATA = os.path.join(ROOT, "insightrx", "app", "data")
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


AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
       "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
       "MSE": "M"}
IGNORE_HET = {"HOH", "SO4", "CL", "NA", "GOL", "EDO", "PEG", "NAG", "ZN", "NAP", "PLM", "MG", "CA", "K", "ACT", "PO4"}
CONTACT_A = 4.5
UNIPROT_FEATURES = {"Domain", "Transmembrane", "Signal", "Binding site", "Active site", "Topological domain"}


def _atoms(path):
    """(record, resn, chain, resseq+icode, atom, xyz, bfactor) for every atom line."""
    import numpy as np
    out = []
    for ln in open(path):
        if ln.startswith(("ATOM", "HETATM")):
            out.append((ln[:6].strip(), ln[17:20].strip(), ln[21], ln[22:27].strip(), ln[12:16].strip(),
                        np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]), float(ln[60:66] or 0)))
    return out


def _chain_residues(atoms, chain):
    seen, res = set(), []
    for rec, resn, ch, rid, *_ in atoms:
        if rec == "ATOM" and ch == chain and rid not in seen and resn in AA3:
            seen.add(rid)
            res.append((rid, AA3[resn]))
    return res


def _map_to_uniprot(chain_res, useq):
    """Chain residue id -> UniProt position via sequence alignment (robust to numbering offsets and gaps)."""
    from difflib import SequenceMatcher
    cseq = "".join(a for _, a in chain_res)
    m = SequenceMatcher(None, cseq, useq, autojunk=False)
    out = {}
    for blk in m.get_matching_blocks():
        if blk.size >= 4:
            for k in range(blk.size):
                out[chain_res[blk.a + k][0]] = blk.b + k + 1
    return out, (sum(b.size for b in m.get_matching_blocks()) / max(1, len(cseq)))


def _pocket(t, useq):
    """Target residues within 4.5 A of the drug (ligand, antibody or peptide) in the experimental complex."""
    import numpy as np
    cx = t.get("complex")
    path = os.path.join(STRUCT, f"{cx['pdb']}.pdb") if cx else None
    if not path or not os.path.exists(path):
        return None
    atoms = _atoms(path)
    partners = set(cx.get("partner_chains") or [])
    chains = sorted({a[2] for a in atoms if a[0] == "ATOM"} - partners)
    target_chains = {}
    for ch in chains:
        res = _chain_residues(atoms, ch)
        mapping, ident = _map_to_uniprot(res, useq)
        if ident > 0.6 and len(res) > 30:
            target_chains[ch] = mapping
    if partners:
        drug = np.array([a[5] for a in atoms if a[2] in partners])
        drug_label = "antibody or peptide chains " + ", ".join(sorted(partners))
    else:
        lig = cx.get("ligand")
        drug = np.array([a[5] for a in atoms if a[0] == "HETATM" and (a[1] == lig if lig else a[1] not in IGNORE_HET)])
        drug_label = f"ligand {lig}"
    if not len(drug):
        return None
    hits = set()
    for rec, resn, ch, rid, _, xyz, _ in atoms:
        if rec == "ATOM" and ch in target_chains and rid in target_chains[ch]:
            if float(np.min(np.linalg.norm(drug - xyz, axis=1))) <= CONTACT_A:
                hits.add(target_chains[ch][rid])
    return {"pdb": cx["pdb"], "drug": drug_label, "cutoff_A": CONTACT_A, "residues": sorted(hits)}


def structure_insights(client, targets):
    out = {}
    for t in targets:
        u = get(client, f"https://rest.uniprot.org/uniprotkb/{t['uniprot']}.json",
                params={"fields": "ft_domain,ft_transmem,ft_signal,ft_binding,ft_act_site,ft_topo_dom,sequence,cc_function"}).json()
        useq = u["sequence"]["value"]
        func = next((c["texts"][0]["value"] for c in u.get("comments", []) if c.get("commentType") == "FUNCTION"), "")
        feats = [{"type": f["type"], "start": f["location"]["start"]["value"], "end": f["location"]["end"]["value"],
                  "desc": f.get("description", "") or (f.get("ligand") or {}).get("name", "")}
                 for f in u.get("features", []) if f["type"] in UNIPROT_FEATURES]
        af = _atoms(os.path.join(STRUCT, f"AF-{t['uniprot']}.pdb"))
        plddt = [round(a[6]) for a in af if a[0] == "ATOM" and a[4] == "CA"]
        dis, start = [], None
        for i, v in enumerate(plddt + [100]):
            if v < 50 and start is None:
                start = i
            elif v >= 50 and start is not None:
                if i - start >= 10:
                    dis.append([start + 1, i])
                start = None
        pocket = _pocket(t, useq)
        if pocket:
            vals = [plddt[p - 1] for p in pocket["residues"] if p - 1 < len(plddt)]
            pocket["mean_plddt"] = round(sum(vals) / len(vals), 1) if vals else None
            pocket["aa"] = [f"{useq[p - 1]}{p}" for p in pocket["residues"] if p - 1 < len(useq)]
        out[t["gene"]] = {"length": len(useq), "function": func[:600], "features": feats, "plddt": plddt,
                          "disordered": dis, "pocket": pocket}
        print(f"{t['gene']:8s} {len(useq)} aa, {len(feats)} features, {len(dis)} disordered regions, "
              f"pocket {len(pocket['residues']) if pocket else '-'} residues"
              f"{' (mean pLDDT %s)' % pocket['mean_plddt'] if pocket else ''}")
    json.dump(out, open(os.path.join(DATA, "structure_insights.json"), "w"), separators=(",", ":"))


def snapshots():
    trials = {topic: external.trials_live(topic, external.DEFAULT_ZIP) for topic in external.TRIAL_QUERIES}
    json.dump(trials, open(os.path.join(DATA, "trials_snapshot.json"), "w"), indent=1)
    print("trials snapshot:", {k: len(v) for k, v in trials.items()})
    nppes = {tax: external.nppes_live(tax, external.DEFAULT_ZIP) for tax in external.TAXONOMIES}
    json.dump(nppes, open(os.path.join(DATA, "nppes_snapshot.json"), "w"), indent=1)
    print("nppes snapshot:", {k: len(v) for k, v in nppes.items()})


def main():
    targets = json.load(open(os.path.join(DATA, "targets.json")))["targets"]
    with httpx.Client() as client:
        if not {"--skip-structures", "--snapshots-only", "--insights-only"} & set(sys.argv):
            fetch_structures(client, targets)
            os.makedirs(VENDOR, exist_ok=True)
            open(os.path.join(VENDOR, "3Dmol-min.js"), "wb").write(get(client, THREEDMOL).content)
        if "--insights-only" in sys.argv:
            structure_insights(client, targets)
            return
        if "--snapshots-only" not in sys.argv:
            chembl_drugs(client, targets)
        structure_insights(client, targets)
    snapshots()


if __name__ == "__main__":
    main()
