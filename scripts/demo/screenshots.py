"""
Capture documentation screenshots of every Insight Rx screen (production, live models) into docs/screenshots/,
plus the three PDF reports into docs/reports/. The access code is entered off-screen and never captured.

  python scripts/demo/screenshots.py [--base URL]
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from record import MEDS, ROOT, SAMPLE, env_code  # noqa: E402

SHOTS = os.path.join(ROOT, "docs", "screenshots")
REPORTS = os.path.join(ROOT, "docs", "reports")
W, H = 1440, 900
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/data/users3/nshaik3/Projects/Oculomics/RetiLink/tools/pw")


def sign_in(page, base, name):
    page.goto(base + "/login")
    page.locator("form[action='/login']", has_text=name).locator("button").first.click()
    page.wait_for_load_state("networkidle")


def shot(page, name, full=True, wait=600, selector=None):
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(wait)
    path = os.path.join(SHOTS, f"{name}.png")
    if selector:
        page.locator(selector).first.screenshot(path=path)
    else:
        page.screenshot(path=path, full_page=full)
    print("  ", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://insightrx-hcp.vercel.app")
    base = ap.parse_args().base
    os.makedirs(SHOTS, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
        ctx = b.new_context(viewport={"width": W, "height": H}, device_scale_factor=1)
        page = ctx.new_page()
        page.goto(base + "/access")
        page.fill("#code", env_code())
        page.click("form[action='/access'] button")
        page.wait_for_load_state("networkidle")
        shot(page, "01_sign_in", full=False)

        sign_in(page, base, "Dr. Alex Morgan")
        page.goto(base + "/screen"); shot(page, "02_screen")
        page.set_input_files("input[name=files_OD]", os.path.join(SAMPLE, "OD_1.jpg"))
        page.set_input_files("input[name=files_OS]", os.path.join(SAMPLE, "OS_1.jpg"))
        page.fill("#in-age", "64")
        for m in MEDS:
            page.locator(f"label.medchip:has(input[value={m}])").click()
        page.locator("aside form button[type=submit]").click()
        page.wait_for_selector("section.personal", timeout=180_000)
        page.locator("[data-attn-toggle]:not([hidden])").first.click()
        page.wait_for_timeout(2500)
        shot(page, "03_screen_result")
        shot(page, "04_screen_treatment", selector="section[aria-labelledby=tx-h]")
        shot(page, "05_screen_personal_targets", selector="section.personal")
        token = re.search(r"/screen/report\.pdf\?token=([\w-]+)", page.content()).group(1)

        page.goto(base + f"/therapeutics/targets/VEGFA?screen={token}"); shot(page, "06_target_vegfa_complex", wait=4000)
        page.goto(base + f"/therapeutics/targets/VEGFA?view=model&screen={token}"); shot(page, "07_target_vegfa_alphafold", wait=4000)
        page.goto(base + "/therapeutics/targets/SLC5A2"); shot(page, "08_target_sglt2_empagliflozin", wait=4000)
        page.goto(base + "/therapeutics"); shot(page, "09_therapeutics_explorer")

        page.goto(base + "/patients"); shot(page, "10_patients")
        case_id = re.findall(r'href="/cases/(\d+)"', page.content())[0]
        page.goto(base + f"/cases/{case_id}"); shot(page, "11_case_screening", wait=1500)
        page.goto(base + f"/cases/{case_id}?tab=systemic"); shot(page, "12_case_whole_body")
        page.goto(base + f"/cases/{case_id}/therapy"); shot(page, "13_case_therapy_trials")
        page.goto(base + f"/cases/{case_id}/consult"); shot(page, "14_consult_specialist")
        page.goto(base + f"/cases/{case_id}/refer-out?topic=nephropathy"); shot(page, "15_refer_out_npi_registry")
        npi = re.search(r"/letter\?npi=(\d{10})", page.content())
        if npi:
            page.goto(base + f"/cases/{case_id}/letter?npi={npi.group(1)}&topic=nephropathy"); shot(page, "16_referral_letter")
        page.goto(base + "/consults"); shot(page, "17_consults_inbox")
        ref = re.search(r'href="/referrals/(\d+)"', page.content())
        if ref:
            page.goto(base + f"/referrals/{ref.group(1)}"); shot(page, "18_referral_relay")
        page.goto(base + "/quality"); shot(page, "19_cms_quality_billing")
        page.goto(base + "/performance"); shot(page, "20_model_performance")
        page.goto(base + "/oculomics?view=science"); shot(page, "21_how_oculomics_works")
        page.goto(base + "/medinfo"); shot(page, "22_medinfo_clinician")
        page.goto(base + f"/cases/{case_id}/explain"); shot(page, "27_patient_explainer_en")
        page.goto(base + f"/cases/{case_id}/explain?lang=pt"); shot(page, "28_patient_explainer_pt")
        page.goto(base + "/performance?view=trends"); shot(page, "29_finding_trends")
        page.goto(base + f"/cases/{case_id}?tab=timeline"); shot(page, "30_audit_trail_solana")
        page.goto(base + "/copilot#cp-end"); shot(page, "31_copilot", full=False)

        for name, url in [("patient_therapy_report", f"/screen/report.pdf?token={token}"),
                          ("target_dossier_VEGFA", f"/therapeutics/targets/VEGFA/report.pdf?screen={token}"),
                          ("target_dossier_ACE", "/therapeutics/targets/ACE/report.pdf"),
                          ("portfolio_report", "/therapeutics/report.pdf")]:
            open(os.path.join(REPORTS, f"{name}.pdf"), "wb").write(ctx.request.get(base + url, timeout=180_000).body())
            print("   report", name)

        sign_in(page, base, "Dr. Priya Nair")
        page.goto(base + "/consults"); shot(page, "23_specialist_inbox")
        sign_in(page, base, "Taylor Brooks")
        page.goto(base + "/consults"); shot(page, "24_coordinator_worklist")
        sign_in(page, base, "Morgan Lee")
        page.goto(base + "/medinfo"); shot(page, "25_medinfo_manufacturer_desk")

        mobile = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2,
                               storage_state=ctx.storage_state())
        mp = mobile.new_page()
        sign_in(mp, base, "Dr. Alex Morgan")
        mp.goto(base + f"/cases/{case_id}/therapy"); mp.wait_for_timeout(800)
        mp.screenshot(path=os.path.join(SHOTS, "26_mobile_therapy.png"))
        print("   26_mobile_therapy")
        b.close()


if __name__ == "__main__":
    main()
