from contextlib import contextmanager

from playwright.sync_api import Browser, sync_playwright


@contextmanager
def launch_browser(headless: bool = False):
    with sync_playwright() as playwright:
        browser: Browser = playwright.chromium.launch(headless=headless)
        try:
            yield browser
        finally:
            try:
                browser.close()
            except Exception:
                pass
