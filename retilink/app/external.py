"""
Public-data clients for the Therapeutics layer: ClinicalTrials.gov v2, the CMS NPPES NPI Registry and openFDA labels.

Privacy rule: outbound requests carry only a fixed topic/taxonomy key and a clinic ZIP code, never patient data.
Every call has a short timeout, a TTL cache and a checked-in snapshot fallback (data/*_snapshot.json) so a demo
never depends on the network. Results say where they came from ('live' or 'snapshot').
"""
import json
import os
import re
import time

import httpx

DATA = os.path.join(os.path.dirname(__file__), "data")
TIMEOUT = float(os.environ.get("RETILINK_EXTERNAL_TIMEOUT", "6"))
TTL = 6 * 3600
DEFAULT_ZIP = os.environ.get("RETILINK_CLINIC_ZIP", "30303")          # clinic location (Atlanta), not the patient's
UA: dict = {}                                                       # ClinicalTrials.gov rejects custom user agents
OFFLINE = os.environ.get("RETILINK_EXTERNAL", "live") == "offline"

# topic key -> ClinicalTrials.gov condition query (fixed strings: nothing patient-derived is ever sent)
TRIAL_QUERIES = {
    "dr": "diabetic retinopathy",
    "edema": "diabetic macular edema",
    "heart": "hypertension AND diabetes",
    "kidneys": "diabetic kidney disease",
    "nerves": "diabetic neuropathy",
}
# NPPES taxonomy descriptions by referral topic
TAXONOMIES = {
    "Ophthalmology": "retinal",
    "Nephrology": "nephropathy",
    "Cardiovascular Disease": "systemic_hypertension",
    "Neurology": "neuropathy",
}
_cache: dict = {}


def _snapshot(name):
    try:
        return json.load(open(os.path.join(DATA, name)))
    except (OSError, ValueError):
        return {}


def _cached(key, fn, fallback):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < TTL:
        return hit[1]
    if not OFFLINE:
        try:
            val = {"source": "live", "items": fn()}
            _cache[key] = (time.time(), val)
            return val
        except (httpx.HTTPError, ValueError, KeyError):
            pass
    return {"source": "snapshot", "items": fallback()}


def clean_zip(z) -> str:
    z = (z or "").strip()[:5]
    return z if re.fullmatch(r"\d{5}", z) else DEFAULT_ZIP


# ------------------------------------------------------------------ ClinicalTrials.gov
def _trial(study):
    ps = study["protocolSection"]
    ident, status = ps["identificationModule"], ps.get("statusModule", {})
    elig = ps.get("eligibilityModule", {})
    design = ps.get("designModule", {})
    locs = (ps.get("contactsLocationsModule") or {}).get("locations") or []
    sponsor = ((ps.get("sponsorCollaboratorsModule") or {}).get("leadSponsor") or {}).get("name", "")
    interventions = [i.get("name", "") for i in (ps.get("armsInterventionsModule") or {}).get("interventions", [])][:4]
    us = [l for l in locs if l.get("country") == "United States"]
    return {"nct": ident["nctId"], "title": ident.get("briefTitle", ""), "status": status.get("overallStatus", ""),
            "phase": ", ".join(design.get("phases") or []) or "N/A", "sponsor": sponsor, "interventions": interventions,
            "min_age": elig.get("minimumAge", ""), "max_age": elig.get("maximumAge", ""), "sex": elig.get("sex", "ALL"),
            "sites": len(locs), "site": (f"{us[0].get('facility', '')}, {us[0].get('city', '')}, {us[0].get('state', '')}"
                                         if us else (locs[0].get("country", "") if locs else "")),
            "url": f"https://clinicaltrials.gov/study/{ident['nctId']}"}


def trials_live(topic, zip_code, n=8):
    params = {"query.cond": TRIAL_QUERIES[topic], "filter.overallStatus": "RECRUITING", "pageSize": n,
              "fields": "protocolSection"}
    if zip_code:                                              # near the clinic, 500 miles
        params["filter.geo"] = f"distance({_zip_point(zip_code)},500mi)"
    r = httpx.get("https://clinicaltrials.gov/api/v2/studies", params=params, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    return [_trial(s) for s in r.json().get("studies", [])]


ZIP_POINTS = {"30303": "33.7527,-84.3915"}                   # clinic ZIPs we geocode offline; others: no geo filter


def _zip_point(z):
    return ZIP_POINTS.get(z, ZIP_POINTS["30303"])


def trials(topic, zip_code=None):
    if topic not in TRIAL_QUERIES:
        return {"source": "none", "items": []}
    z = clean_zip(zip_code)
    return _cached(("trials", topic, z), lambda: trials_live(topic, z), lambda: _snapshot("trials_snapshot.json").get(topic, []))


# ------------------------------------------------------------------ CMS NPPES NPI Registry
def _provider(r):
    basic = r.get("basic", {})
    addr = next((a for a in r.get("addresses", []) if a.get("address_purpose") == "LOCATION"), (r.get("addresses") or [{}])[0])
    tax = next((t for t in r.get("taxonomies", []) if t.get("primary")), (r.get("taxonomies") or [{}])[0])
    name = " ".join(x for x in [basic.get("first_name", "").title(), basic.get("last_name", "").title()] if x) \
        or basic.get("organization_name", "").title()
    cred = basic.get("credential", "")
    return {"npi": r.get("number"), "name": name + (f", {cred}" if cred else ""), "taxonomy": tax.get("desc", ""),
            "address": f"{addr.get('address_1', '').title()}, {addr.get('city', '').title()}, {addr.get('state', '')} "
                       f"{(addr.get('postal_code') or '')[:5]}",
            "phone": addr.get("telephone_number", ""),
            "url": f"https://npiregistry.cms.hhs.gov/provider-view/{r.get('number')}"}


def nppes_live(taxonomy, zip_code, n=8):
    params = {"version": "2.1", "taxonomy_description": taxonomy, "postal_code": zip_code[:3] + "*",
              "enumeration_type": "NPI-1", "limit": n}
    r = httpx.get("https://npiregistry.cms.hhs.gov/api/", params=params, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    return [_provider(x) for x in r.json().get("results", [])]


def nppes(taxonomy, zip_code=None):
    if taxonomy not in TAXONOMIES:
        return {"source": "none", "items": []}
    z = clean_zip(zip_code)
    return _cached(("nppes", taxonomy, z), lambda: nppes_live(taxonomy, z),
                   lambda: _snapshot("nppes_snapshot.json").get(taxonomy, []))


def provider(npi):
    """One provider by NPI (10 digits), live registry first, then the snapshot."""
    if not re.fullmatch(r"\d{10}", npi or ""):
        return None

    def live():
        r = httpx.get("https://npiregistry.cms.hhs.gov/api/", params={"version": "2.1", "number": npi},
                      headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        return [_provider(x) for x in r.json().get("results", [])]
    got = _cached(("npi", npi), live, lambda: [p for items in _snapshot("nppes_snapshot.json").values()
                                               for p in items if str(p["npi"]) == npi])
    return got["items"][0] if got["items"] else None


def taxonomy_for(topic):
    return next((t for t, k in TAXONOMIES.items() if k == topic), "Ophthalmology")


# ------------------------------------------------------------------ openFDA label sections
def label_section(generic, section="warnings_and_cautions"):
    """First 600 characters of a label section for a generic drug name (curated names only)."""
    if not re.fullmatch(r"[a-z][a-z \-]{2,40}", generic or ""):
        return {"source": "none", "items": []}

    def live():
        r = httpx.get("https://api.fda.gov/drug/label.json", params={"search": f'openfda.generic_name:"{generic}"', "limit": 1},
                      headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        res = r.json()["results"][0]
        text = " ".join(res.get(section) or res.get("warnings") or [])
        return [{"text": text[:600] + ("..." if len(text) > 600 else ""), "set_id": res.get("set_id", ""),
                 "url": f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={res.get('set_id', '')}"}]
    return _cached(("label", generic, section), live, lambda: [])
