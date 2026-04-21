"""Small guards so scan/apply loops stop cleanly when the user closes the browser."""

from playwright.sync_api import Page


def page_is_open(page: Page) -> bool:
    try:
        return not page.is_closed()
    except Exception:
        return False


def playwright_fatal(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = str(exc).lower()
    return (
        "TargetClosed" in name
        or "closed" in msg
        or "connection closed" in msg
        or "browser has been closed" in msg
    )
