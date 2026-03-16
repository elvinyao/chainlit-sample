"""Browser service — uses Playwright to fill and submit a local HTML form."""

from pathlib import Path
from typing import Any, Dict

from playwright.async_api import async_playwright

# Project root is two levels up from this file (services/browser_service.py -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def playwright_fill_form(data: Dict[str, Any]) -> Dict[str, str]:
    """Open sample form in headless Chromium, fill fields, submit, and take a screenshot."""

    html_path = _PROJECT_ROOT / "sample_site" / "index.html"
    url = html_path.as_uri()

    output_dir = _PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    screenshot_path = output_dir / "last_run.png"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)
        await page.fill("input[name='name']", data["name"])
        await page.fill("input[name='email']", data["email"])
        await page.fill("textarea[name='message']", data["message"])
        await page.click("button[type='submit']")
        result_text = await page.locator("#result").inner_text()
        await page.screenshot(path=str(screenshot_path), full_page=True)
        await browser.close()

    return {
        "result_text": result_text,
        "screenshot_path": str(screenshot_path),
    }
