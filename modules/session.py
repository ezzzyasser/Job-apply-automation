import re
import time
from pathlib import Path

from playwright.sync_api import Browser, Locator, Page, TimeoutError as PlaywrightTimeout


def _url_suggests_logged_out(page: Page) -> bool:
    u = (page.url or "").lower()
    return any(
        x in u
        for x in (
            "/login",
            "/uas/login",
            "signin",
            "/checkpoint",
            "/challenge",
        )
    )


def _welcome_back_visible(page: Page) -> bool:
    """LinkedIn 'Welcome back, {name}' re-auth — usually password field only."""
    try:
        loc = page.get_by_text(re.compile(r"welcome\s+back", re.I)).first
        return loc.count() > 0 and loc.is_visible()
    except Exception:
        return False


def _find_visible_password_input(page: Page) -> Locator | None:
    """Primary password box (full login or welcome-back card)."""
    ordered = (
        'input#password',
        'input[name="session_password"]',
        'input[id*="password"]',
    )
    for sel in ordered:
        loc = page.locator(sel).first
        try:
            if loc.count() > 0 and loc.is_visible():
                return loc
        except Exception:
            continue
    try:
        loc = page.get_by_label(re.compile(r"^\s*password\s*$", re.I)).first
        if loc.count() > 0 and loc.is_visible():
            return loc
    except Exception:
        pass
    try:
        n = page.locator('input[type="password"]').count()
        for i in range(min(n, 8)):
            loc = page.locator('input[type="password"]').nth(i)
            if loc.is_visible():
                return loc
    except Exception:
        pass
    return None


def _find_visible_username_input(page: Page) -> Locator | None:
    loc = page.locator('input[name="session_key"], input#username').first
    try:
        if loc.count() > 0 and loc.is_visible():
            return loc
    except Exception:
        pass
    return None


def _still_on_login_wall(page: Page) -> bool:
    """
    Any LinkedIn surface where the user must authenticate (email+password or welcome-back).
    """
    pwd = _find_visible_password_input(page)
    if pwd is None:
        return False
    if _url_suggests_logged_out(page):
        return True
    if _welcome_back_visible(page):
        return True
    if _find_visible_username_input(page) is not None:
        return True
    try:
        if page.get_by_text(re.compile(r"forgot\s+password", re.I)).first.count() > 0:
            if page.get_by_text(re.compile(r"forgot\s+password", re.I)).first.is_visible():
                return True
    except Exception:
        pass
    return False


def session_looks_valid(page: Page) -> bool:
    """True if we appear to be on a normal signed-in LinkedIn surface."""
    if _still_on_login_wall(page):
        return False
    try:
        if page.locator("nav.global-nav").first.count() > 0:
            try:
                if page.locator("nav.global-nav").first.is_visible():
                    return True
            except Exception:
                return True
    except Exception:
        pass
    try:
        if page.locator("#global-nav").first.count() > 0:
            return True
    except Exception:
        pass
    u = (page.url or "").lower()
    return "linkedin.com/feed" in u or "linkedin.com/jobs" in u


def _email_field_prefilled_or_partial(page: Page) -> str:
    user = _find_visible_username_input(page)
    if user is None:
        return ""
    try:
        return (user.input_value() or "").strip()
    except Exception:
        return ""


def _click_sign_in(page: Page) -> bool:
    selectors = (
        'button[type="submit"]',
        "button.sign-in-form__submit-btn",
        "button.artdeco-button--primary",
        'input[type="submit"]',
    )
    for sel in selectors:
        btn = page.locator(sel).filter(has_text=re.compile(r"sign\s*in", re.I)).first
        try:
            if btn.count() > 0 and btn.is_visible():
                btn.click(timeout=8000)
                return True
        except Exception:
            continue
    for sel in selectors:
        btn = page.locator(sel).first
        try:
            if btn.count() > 0 and btn.is_visible():
                txt = (btn.inner_text(timeout=500) or "").lower()
                if "sign in" in txt or "signin" in txt.replace(" ", ""):
                    btn.click(timeout=8000)
                    return True
        except Exception:
            continue
    try:
        page.get_by_role("button", name=re.compile(r"^\s*sign\s*in\s*$", re.I)).first.click(
            timeout=8000
        )
        return True
    except Exception:
        pass
    return False


def _try_password_login(page: Page, email: str, password: str, login_url: str) -> bool:
    target = (login_url or "https://www.linkedin.com/login").strip()
    try:
        page.goto(target, wait_until="domcontentloaded", timeout=35000)
    except PlaywrightTimeout:
        return False
    time.sleep(1.4)

    pwd_field = _find_visible_password_input(page)
    if pwd_field is None:
        print("[SESSION] No visible password field on the login page.")
        return False
    if not (password or "").strip():
        return False

    user_field = _find_visible_username_input(page)
    use_config_email = (email or "").strip()

    try:
        if user_field is not None:
            existing = (user_field.input_value() or "").strip()
            if existing and (
                not use_config_email or existing.lower() == use_config_email.lower()
            ):
                print("[SESSION] Email already on the form — filling password only.")
            elif use_config_email:
                user_field.click(timeout=5000)
                user_field.fill(use_config_email, timeout=5000)
            else:
                if not existing:
                    print(
                        "[SESSION] Email field is empty and `email` is missing in config — add it."
                    )
                    return False
        else:
            # "Welcome back" / remembered account — no email input
            print("[SESSION] Welcome-back screen — filling password only (no email field).")
        pwd_field.click(timeout=5000)
        pwd_field.fill("", timeout=2000)
        pwd_field.fill(password.strip(), timeout=5000)
    except Exception as exc:
        print(f"[SESSION] Could not fill password: {exc}")
        return False

    time.sleep(0.35)
    if not _click_sign_in(page):
        print("[SESSION] Could not find a Sign in button to click.")
        return False

    for _ in range(55):
        time.sleep(0.55)
        u = (page.url or "").lower()
        if "checkpoint" in u or "challenge" in u:
            print(
                "[SESSION] LinkedIn wants verification (2FA / captcha). Complete it in the browser."
            )
            return False
        try:
            err = page.locator(
                ".form__label--error, .alert-content, #error-for-password, .form__error"
            ).first
            if err.count() > 0 and err.is_visible():
                print("[SESSION] Login error on page — check password in config.")
                return False
        except Exception:
            pass
        if session_looks_valid(page):
            return True
        if not _still_on_login_wall(page) and "linkedin.com" in u:
            return True
    return session_looks_valid(page)


def open_or_create_session(
    browser: Browser,
    storage_state_path: str,
    login_url: str,
    email: str = "",
    password: str = "",
) -> Page:
    """
    Load cookies from storage_state_path if present; if session is dead, sign in again.
    Supports full login and LinkedIn 'Welcome back' (password-only) screens.
    """
    storage_file = Path(storage_state_path)
    context = None
    try:
        if storage_file.exists():
            context = browser.new_context(storage_state=str(storage_file))
        else:
            context = browser.new_context()
    except Exception as exc:
        print(f"[SESSION] Could not load {storage_state_path} ({exc}). Using a fresh browser context.")
        context = browser.new_context()

    page = context.new_page()
    try:
        page.goto(
            "https://www.linkedin.com/feed/",
            wait_until="domcontentloaded",
            timeout=35000,
        )
    except Exception as exc:
        print(f"[SESSION] Initial navigation issue: {exc}")

    time.sleep(1.2)

    if session_looks_valid(page):
        print("[SESSION] Loaded saved cookies — already signed in.")
        return page

    print("[SESSION] Session missing or expired — signing in again.")

    have_password = bool((password or "").strip())
    if have_password:
        if _try_password_login(page, (email or "").strip(), (password or "").strip(), login_url):
            try:
                context.storage_state(path=storage_state_path)
                print(f"[SESSION] Updated cookies saved to {storage_state_path}")
            except Exception as exc:
                print(f"[SESSION] Could not save cookies: {exc}")
            try:
                page.goto(
                    "https://www.linkedin.com/jobs/",
                    wait_until="domcontentloaded",
                    timeout=25000,
                )
            except Exception:
                pass
            return page
        print("[SESSION] Automatic sign-in did not finish.")

    try:
        page.goto(
            (login_url or "https://www.linkedin.com/login").strip(),
            wait_until="domcontentloaded",
            timeout=30000,
        )
    except Exception:
        pass
    input("[SESSION] Finish login in the browser, then press Enter here to save cookies…")
    try:
        context.storage_state(path=storage_state_path)
        print(f"[SESSION] Saved session state to {storage_state_path}")
    except Exception as exc:
        print(f"[SESSION] Could not save cookies: {exc}")

    try:
        page.goto(
            "https://www.linkedin.com/jobs/",
            wait_until="domcontentloaded",
            timeout=25000,
        )
    except Exception:
        pass
    return page
