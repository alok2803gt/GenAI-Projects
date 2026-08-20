import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width":1440,"height":900})
        errors = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(f"PAGE ERROR: {exc}"))

        await page.goto("http://localhost:8001", wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(1500)

        await page.click("text=Live Trading")
        await page.wait_for_timeout(500)
        tabs = await page.query_selector_all("button")
        clicked = False
        for t in tabs:
            text = await t.inner_text()
            if text.strip() == "GOOG Condor":
                await t.click()
                clicked = True
                break
        print("Clicked GOOG Condor tab:", clicked)
        await page.wait_for_timeout(1500)

        body_text = await page.inner_text("body")
        print("Contains 'GOOG Weekly Condor':", "GOOG Weekly Condor" in body_text)
        print("Contains 'Decision Log':", "decision log" in body_text.lower())
        print("Contains 'PRICING_ABORT':", "PRICING_ABORT" in body_text)
        print("Contains 'None open right now':", "None open right now" in body_text)

        real_errors = [e for e in errors if "deoptimised" not in e]
        print("Console errors:", real_errors if real_errors else "none")

        await page.screenshot(path="googcondor_tab_screenshot.png")
        await browser.close()

asyncio.run(main())
