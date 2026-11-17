"""Browser smoke test for the React dashboard (optional; needs `pip install playwright` + a Chromium/Edge browser).

Usage: start the app (backend serving frontend/dist on :8000), then
       python scripts/e2e_smoke.py [http://localhost:8000] [--channel msedge]
Checks every page renders with no JS errors / failed API calls, the predict + learning-path forms work,
and the mobile menu opens.
"""
import sys

from playwright.sync_api import sync_playwright

BASE = next((a for a in sys.argv[1:] if a.startswith("http")), "http://localhost:8000")
CHANNEL = sys.argv[sys.argv.index("--channel") + 1] if "--channel" in sys.argv else None
PAGES = {"overview": "YouTube educator network study", "graph": "Concept relationship graph", "clusters": "Topic clusters",
         "path": "Learning path generator", "educators": "Pedagogical influence", "predict": "Predict level", "add": "Add a video"}

failures = []
with sync_playwright() as p:
    browser = p.chromium.launch(channel=CHANNEL) if CHANNEL else p.chromium.launch()
    page = browser.new_page(viewport={"width": 1366, "height": 900})
    errors, bad = [], []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("response", lambda r: bad.append(f"{r.status} {r.url}") if "/api/" in r.url and r.status >= 400 else None)

    for slug, heading in PAGES.items():
        page.goto(f"{BASE}/#/{slug}", wait_until="networkidle")
        page.wait_for_selector("h1", timeout=60000)
        h1 = page.inner_text("h1")
        ok = heading.lower() in h1.lower()
        print(f"{'OK ' if ok else 'BAD'} /#/{slug:10s} h1={h1!r}")
        if not ok:
            failures.append(slug)

    # sidebar navigation by clicking
    page.goto(f"{BASE}/#/overview", wait_until="networkidle")
    page.click("nav >> text=Educators")
    page.wait_for_selector("text=Pedagogical influence")
    print("OK  sidebar navigation")

    # learning path form
    page.goto(f"{BASE}/#/path", wait_until="networkidle")
    page.fill("input[aria-label='Goal concept']", "random forest")
    page.click("text=Generate path")
    page.wait_for_selector("text=Path to “random forest”", timeout=30000)
    print("OK  learning path form:", page.locator(".step").count(), "steps")

    # predict form
    page.goto(f"{BASE}/#/predict", wait_until="networkidle")
    page.click("button:has-text('Predict')")
    page.wait_for_selector("text=Topic cluster", timeout=60000)
    print("OK  predict form")

    # mobile menu
    m = browser.new_page(viewport={"width": 390, "height": 844})
    m.goto(f"{BASE}/#/overview", wait_until="networkidle")
    m.click("button[aria-label='Toggle menu']")
    m.click("nav >> text=Clusters")
    m.wait_for_selector("text=Topic clusters")
    width_ok = m.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    print(("OK " if width_ok else "BAD"), " mobile menu + no horizontal overflow")
    if not width_ok:
        failures.append("mobile-overflow")
    browser.close()

if errors:
    failures.append("js-errors")
    print("JS errors:", errors)
if bad:
    failures.append("api-errors")
    print("Failed API calls:", bad)
print("RESULT:", "PASS" if not failures else f"FAIL {failures}")
sys.exit(1 if failures else 0)
