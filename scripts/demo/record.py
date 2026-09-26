"""
Record the Insight Rx demo against production with Playwright, paced to the narration (timings.json).

Phase A (no video): unlock the access code, sign in, run an off-camera screening to render the PDF pages for the
reports slide, and save the session. The access code never appears on camera.
Phase B (video): one continuous recording; each scene's start time is logged for compose.py.

  python scripts/demo/record.py [--base URL] [--out DIR]
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from script import GAP_S, SCENES, roi_numbers  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT = "/data/users3/nshaik3/Projects/Oculomics/RetiLink/demo/out"
SAMPLE = os.path.join(ROOT, "test_data", "mbrset", "patients", "03_referable_both_eyes")
MEDS = ["semaglutide", "pioglitazone"]
W, H = 1536, 864
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/data/users3/nshaik3/Projects/Oculomics/RetiLink/tools/pw")

CURSOR_JS = """
(() => {
  const mk = () => {
    if (document.getElementById('demo-cursor')) return;
    const c = document.createElement('div'); c.id = 'demo-cursor';
    c.style.cssText = 'position:fixed;left:-40px;top:-40px;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;' +
      'border:3px solid #e0782b;background:rgba(224,120,43,.18);z-index:2147483647;pointer-events:none;transition:transform .12s';
    document.documentElement.appendChild(c);
    addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
    addEventListener('mousedown', e => {
      c.style.transform = 'scale(.7)';
      const r = document.createElement('div');
      r.style.cssText = `position:fixed;left:${e.clientX}px;top:${e.clientY}px;width:10px;height:10px;margin:-5px 0 0 -5px;border-radius:50%;` +
        'border:3px solid #e0782b;z-index:2147483646;pointer-events:none;transition:all .5s ease-out;opacity:1';
      document.documentElement.appendChild(r);
      requestAnimationFrame(() => { r.style.width = r.style.height = '60px'; r.style.margin = '-30px 0 0 -30px'; r.style.opacity = '0'; });
      setTimeout(() => r.remove(), 600);
    }, true);
    addEventListener('mouseup', () => { c.style.transform = ''; }, true);
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mk); else mk();
})();
"""


def env_code():
    for line in open(os.path.join(ROOT, ".env")):
        if line.startswith("DEMO_ACCESS_CODE="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("DEMO_ACCESS_CODE missing from .env")


# ------------------------------------------------------------------ phase A: off-camera preparation
def prepare(browser, base, out):
    ctx = browser.new_context(viewport={"width": W, "height": H})
    page = ctx.new_page()
    page.goto(base + "/access")
    page.fill("#code", env_code())
    page.click("form[action='/access'] button")
    page.wait_for_load_state("networkidle")
    page.goto(base + "/login")
    page.locator("form[action='/login']", has_text="Dr. Alex Morgan").locator("button").click()
    page.wait_for_load_state("networkidle")
    state = os.path.join(out, "state.json")
    ctx.storage_state(path=state)
    # a case for the workflow scene (first referable case in Dr. Morgan's list)
    page.goto(base + "/patients")
    case_id = re.findall(r'href="/cases/(\d+)"', page.content())[0]
    # off-camera screening -> PDF pages for the reports slide
    files = {f"files_{e}": {"name": f"{e}_1.jpg", "mimeType": "image/jpeg",
                            "buffer": open(os.path.join(SAMPLE, f"{e}_1.jpg"), "rb").read()} for e in ("OD", "OS")}
    r = ctx.request.post(base + "/screen", multipart={"age": "64", "med": MEDS[0], **files}, timeout=180_000)
    token = re.search(r"/screen/report\.pdf\?token=([\w-]+)", r.text()).group(1)
    pdfs = {"patient": (f"/screen/report.pdf?token={token}", 2), "dossier": (f"/therapeutics/targets/VEGFA/report.pdf?screen={token}", 0),
            "portfolio": ("/therapeutics/report.pdf", 0)}
    slides = os.path.join(out, "slides")
    os.makedirs(slides, exist_ok=True)
    import pymupdf
    for name, (url, page_no) in pdfs.items():
        doc = pymupdf.open(stream=ctx.request.get(base + url, timeout=180_000).body(), filetype="pdf")
        doc[min(page_no, doc.page_count - 1)].get_pixmap(dpi=110).save(os.path.join(slides, f"page_{name}.png"))
    ctx.close()
    build_slides(slides)
    return state, case_id


def build_slides(slides):
    n = roi_numbers()
    html = open(os.path.join(HERE, "slides.html")).read()
    for k, v in {"GAPS": f"{n['gaps']:,}", "DR": f"{n['dr']:,}", "VTDR": f"{n['vtdr']:,}", "RATE": f"{n['rate_92228']:.2f}",
                 "BILLING_K": f"{n['billing'] / 1000:,.0f}"}.items():
        html = html.replace("{{" + k + "}}", v)
    sprite = open(os.path.join(ROOT, "public", "static", "icons.svg")).read()      # inline: file:// blocks external <use>
    html = html.replace('href="icons.svg#', 'href="#').replace("<body>", "<body>" + sprite, 1)
    open(os.path.join(slides, "index.html"), "w").write(html)
    shutil.copy(os.path.join(ROOT, "public", "static", "fonts", "PublicSans.woff2"), slides)


# ------------------------------------------------------------------ phase B helpers
class Clock:
    def __init__(self):
        self.t0 = time.monotonic()

    def now(self):
        return time.monotonic() - self.t0


def glide(page, locator, click=True, steps=28):
    locator.scroll_into_view_if_needed()
    box = locator.bounding_box()
    if not box:
        return
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y, steps=steps)
    page.wait_for_timeout(180)
    if click:
        page.mouse.click(x, y)


def scroll_to(page, selector, offset=90, wait=900):
    page.evaluate("""([s, o]) => { const el = document.querySelector(s); if (!el) return;
        window.scrollTo({top: el.getBoundingClientRect().top + window.scrollY - o, behavior: 'smooth'}); }""", [selector, offset])
    page.wait_for_timeout(wait)


def cues(scene_id, timings):
    """Start time of each sentence within the scene, so actions can land on the words."""
    t, out = 0.0, []
    for s in timings[scene_id]["sentences"]:
        out.append(t)
        t += s["dur"] + GAP_S
    return out


def wait_until(page, clock, t_abs):
    rest = t_abs - clock.now()
    if rest > 0:
        page.wait_for_timeout(int(rest * 1000))


# ------------------------------------------------------------------ scenes
def run_scene(page, sc, clock, start, timings, base, slides_url, case_id):
    cue = cues(sc["id"], timings)
    at = lambda i, extra=0.0: wait_until(page, clock, start + cue[min(i, len(cue) - 1)] + extra)
    sid = sc["id"]
    if "slide" in sc:
        if page.url.startswith("file://"):
            page.evaluate("h => { location.hash = h; }", sc["slide"])
        else:
            page.goto(f"{slides_url}#{sc['slide']}")
        return
    if sid == "screen":
        page.goto(base + "/screen")
        page.wait_for_load_state("networkidle")
        page.set_input_files("input[name=files_OD]", os.path.join(SAMPLE, "OD_1.jpg"))
        page.wait_for_timeout(500)
        page.set_input_files("input[name=files_OS]", os.path.join(SAMPLE, "OS_1.jpg"))
        glide(page, page.locator("#in-age"))
        page.keyboard.type("64", delay=90)
        for m in MEDS:
            glide(page, page.locator(f"label.medchip:has(input[value={m}])"))
        glide(page, page.locator("aside form button[type=submit]"))
        page.wait_for_selector("section.personal", timeout=180_000)
        page.wait_for_timeout(600)
        scroll_to(page, ".photo-grid", 150)
        toggle = page.locator("[data-attn-toggle]:not([hidden])").first
        if toggle.count():
            at(2, 2.5)
            glide(page, toggle)
            page.wait_for_timeout(2500)
        at(3)
        scroll_to(page, ".oc-patient", 80, 1400)
    elif sid == "therapy":
        scroll_to(page, "#tx-h", 80, 1200)
        at(1, 1.0)
        alerts = page.locator(".ixlist .ix.major")
        for i in range(min(2, alerts.count())):
            glide(page, alerts.nth(i), click=False)
            page.wait_for_timeout(1800)
    elif sid == "targets":
        scroll_to(page, "section.personal", 80, 1200)
        rows = page.locator("section.personal .prio-row")
        at(1)
        for i in range(min(4, rows.count())):
            glide(page, rows.nth(i), click=False, steps=20)
            page.wait_for_timeout(1300)
    elif sid == "vegfa":
        link = page.locator("section.personal .prio-row[href*='VEGFA']")
        glide(page, link)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1500)
        canvas = page.locator(".mol-canvas")
        at(1, 1.0)
        glide(page, canvas, click=False)
        at(2)
        glide(page, page.locator("nav.seg a", has_text="AlphaFold"))
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2500)
        at(3, 0.5)
        scroll_to(page, "#si-h", 120, 1400)
        at(4)
        scroll_to(page, "#d-h", 90, 1400)
    elif sid == "sglt2":
        page.goto(base + "/therapeutics/targets/SLC5A2")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1800)
        glide(page, page.locator(".mol-canvas"), click=False, steps=12)      # auto-rotation does the rest
    elif sid == "workflow":
        page.goto(base + f"/cases/{case_id}/therapy")
        page.wait_for_load_state("networkidle")
        scroll_to(page, "#tr-h", 90, 1200)
        wait_until(page, clock, start + 4.2)
        page.goto(base + f"/cases/{case_id}/refer-out?topic=nephropathy")
        page.wait_for_load_state("networkidle")
        wait_until(page, clock, start + 8.4)
        page.goto(base + "/medinfo")
        page.wait_for_load_state("networkidle")


def record(browser, base, out, state, case_id, timings):
    vids = os.path.join(out, "video")
    shutil.rmtree(vids, ignore_errors=True)
    ctx = browser.new_context(viewport={"width": W, "height": H}, storage_state=state,
                              record_video_dir=vids, record_video_size={"width": W, "height": H})
    ctx.add_init_script(CURSOR_JS)
    page = ctx.new_page()
    clock = Clock()
    slides_url = "file://" + os.path.join(out, "slides", "index.html")
    log = []
    for sc in SCENES:
        start = clock.now()
        run_scene(page, sc, clock, start, timings, base, slides_url, case_id)
        wait_until(page, clock, start + timings[sc["id"]]["min_len"])
        end = clock.now()
        log.append({"id": sc["id"], "start": round(start, 3), "end": round(end, 3)})
        print(f"{sc['id']:11s} {start:7.2f} -> {end:7.2f}  ({end - start:5.1f} s, narration {timings[sc['id']]['speech']:.1f})")
    page.wait_for_timeout(800)
    video = page.video.path()
    ctx.close()
    json.dump({"scenes": log, "video": video}, open(os.path.join(out, "scenes.json"), "w"), indent=1)
    print("video:", video, f"total {clock.now():.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://insightrx-hcp.vercel.app")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    timings = json.load(open(os.path.join(args.out, "timings.json")))
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
        state, case_id = prepare(browser, args.base, args.out)
        record(browser, args.base, args.out, state, case_id, timings)
        browser.close()


if __name__ == "__main__":
    main()
