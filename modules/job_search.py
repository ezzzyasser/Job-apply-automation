import random
import re
import time
from dataclasses import dataclass
from urllib.parse import quote_plus

from playwright.sync_api import Page, TimeoutError

from modules.ai_job_content import prepare_job_application_ai
from modules.apply_flow import ApplyConfig, resolve_cv_path_for_job, run_easy_apply
from modules.memory_store import MemoryStore
from modules.safe_page import page_is_open, playwright_fatal


@dataclass(frozen=True)
class JobCardResult:
    keyword: str
    title: str
    company: str
    easy_apply: bool
    status: str


ROLE_SYNONYMS: dict[str, list[str]] = {
    "ai_engineer": ["ai", "artificial intelligence", "ai engineer", "ai developer"],
    "ml_engineer": ["ml", "mle", "machine learning", "machine learning engineer"],
    "computer_vision": ["computer vision", "cv", "image processing"],
    "data_science": ["data science", "data scientist", "ds"],
    "nlp": ["nlp", "nlu", "nlg", "natural language processing"],
    "automation": ["automation", "rpa", "python developer", "automation developer"],
}


def _normalize_role_text(value: str) -> str:
    normalized = " ".join(value.lower().split())
    normalized = normalized.replace("artificial intelligence", "ai")
    normalized = normalized.replace("machine learning", "ml")
    return normalized


def _detect_keyword_group(keyword: str) -> str | None:
    keyword_n = _normalize_role_text(keyword)
    for group, terms in ROLE_SYNONYMS.items():
        if any(_normalize_role_text(term) in keyword_n for term in terms):
            return group
    return None


def _matches_keyword_equivalent(title: str, keyword: str) -> bool:
    normalized_title = _normalize_role_text(title)
    keyword_group = _detect_keyword_group(keyword)
    if keyword_group is not None:
        return any(_normalize_role_text(term) in normalized_title for term in ROLE_SYNONYMS[keyword_group])
    normalized_keyword = _normalize_role_text(keyword)
    return normalized_keyword in normalized_title


def build_jobs_url(
    keyword: str,
    location: str,
    easy_apply_only: bool = True,
    entry_level_only: bool = False,
) -> str:
    encoded_keyword = quote_plus(keyword)
    encoded_location = quote_plus(location)
    base_url = (
        "https://www.linkedin.com/jobs/search/"
        f"?keywords={encoded_keyword}&location={encoded_location}"
    )
    if easy_apply_only:
        base_url += "&f_AL=true"
    if entry_level_only:
        base_url += "&f_E=1,2"
    return base_url


def _human_delay(min_seconds: float = 1.0, max_seconds: float = 2.0) -> None:
    time.sleep(random.uniform(min_seconds, max_seconds))


def scan_jobs_for_keyword(
    page: Page,
    keyword: str,
    location: str,
    max_jobs: int,
    max_accepted_min_experience_years: int = 1,
    memory: MemoryStore | None = None,
    apply_cfg: ApplyConfig | None = None,
) -> list[JobCardResult]:
    url = build_jobs_url(
        keyword=keyword,
        location=location,
        easy_apply_only=True,
        entry_level_only=False,
    )
    page.goto(url, wait_until="domcontentloaded")
    _human_delay()

    card_selectors = [
        "ul.scaffold-layout__list-container li",
        "div.job-card-container",
        "li.jobs-search-results__list-item",
    ]

    matched_selector = ""
    for selector in card_selectors:
        try:
            page.wait_for_selector(selector, timeout=7000)
            if page.locator(selector).count() > 0:
                matched_selector = selector
                break
        except TimeoutError:
            continue

    if not matched_selector:
        print(f"[{keyword}] No job cards found on page. Skipping keyword.")
        return []

    results: list[JobCardResult] = []
    seen_jobs: set[tuple[str, str]] = set()
    page_index = 1

    while True:
        if not page_is_open(page):
            print(f"[{keyword}] Browser closed; stopping keyword scan.")
            return results

        remaining = 0 if max_jobs <= 0 else max(0, max_jobs - len(results))
        if max_jobs > 0 and remaining == 0:
            break

        # Do NOT bulk-load all cards before processing — that scrolls the whole page,
        # virtualizes the list, and can make it look like "next page" before any apply.
        print(f"[{keyword}] Page {page_index}: processing cards one-by-one (list scroll only).")

        page_matches = 0
        idx = 0
        while True:
            if not page_is_open(page):
                print(f"[{keyword}] Browser closed; stopping.")
                return results
            if max_jobs > 0 and len(results) >= max_jobs:
                return results

            if not _ensure_job_card_index_ready(page, matched_selector, idx):
                break

            try:
                total_cards = page.locator(matched_selector).count()
            except Exception as exc:
                if playwright_fatal(exc):
                    print(f"[{keyword}] Browser closed during scan.")
                    return results
                raise
            if idx >= total_cards:
                break

            try:
                card = page.locator(matched_selector).nth(idx)
                card.scroll_into_view_if_needed()
                card.click(timeout=5000)
                _human_delay(1.0, 1.8)

                title_locator = page.locator(
                    "h1.t-24.t-bold.inline, h1.jobs-unified-top-card__job-title, h1"
                ).first
                company_locator = page.locator(
                    "div.job-details-jobs-unified-top-card__company-name a, "
                    "a.jobs-unified-top-card__company-name, "
                    "div.jobs-unified-top-card__primary-description a"
                ).first
                title = title_locator.inner_text(timeout=3000).strip()
                company = company_locator.inner_text(timeout=3000).strip()

                dedupe_key = (title.lower(), company.lower())
                if dedupe_key in seen_jobs:
                    idx += 1
                    continue
                seen_jobs.add(dedupe_key)

                title_l = title.lower()
                company_l = company.lower()
                matches_keyword = _matches_keyword_equivalent(title=title_l, keyword=keyword)
                description_text = _extract_job_description(page)
                min_required = _extract_min_required_years(description_text)
                # Unknown years in description: still allow apply if keyword (and rest) match.
                experience_ok = min_required is None or (
                    min_required <= max_accepted_min_experience_years
                )

                easy_apply_button = page.locator(
                    "button.jobs-apply-button, button:has-text('Easy Apply')"
                ).first
                easy_apply = easy_apply_button.count() > 0 and easy_apply_button.is_visible()

                if not matches_keyword:
                    status = "skipped_not_keyword_match"
                    easy_apply_flag = False
                elif not experience_ok:
                    status = "skipped_experience_too_high"
                    easy_apply_flag = False
                else:
                    page_matches += 1
                    if memory is not None and apply_cfg is not None:
                        print(
                            f"[FREEZE] Eligible: {title_l} @ {company_l} — "
                            "applying now (no next job / no next page until done)."
                        )
                        try:
                            pre_tailored: dict | None = None
                            cv_excerpt = ""
                            if apply_cfg.ollama is not None:
                                print(
                                    "[AI] Preparing cover letter and CV/keyword text from job + resume "
                                    "before Easy Apply…"
                                )
                                cv_path = resolve_cv_path_for_job(title_l, apply_cfg)
                                pre_tailored, cv_excerpt = prepare_job_application_ai(
                                    apply_cfg.ollama,
                                    apply_cfg.user_profile,
                                    title_l,
                                    company_l,
                                    description_text or "",
                                    cv_path,
                                )
                            apply_status = run_easy_apply(
                                page=page,
                                memory=memory,
                                cfg=apply_cfg,
                                job_title=title_l,
                                company=company_l,
                                job_description=description_text or "",
                                pre_tailored=pre_tailored,
                                cv_excerpt=cv_excerpt,
                            )
                            if apply_status == "page_closed":
                                results.append(
                                    JobCardResult(
                                        keyword=keyword,
                                        title=title_l,
                                        company=company_l,
                                        easy_apply=False,
                                        status="page_closed",
                                    )
                                )
                                print("[APPLY] Page closed; ending scan.")
                                return results
                            status = apply_status
                            easy_apply_flag = apply_status == "applied"
                            print(f"[APPLY] Done -> {apply_status}")
                        except Exception as exc:
                            status = "apply_error"
                            easy_apply_flag = False
                            print(f"[APPLY] Error: {exc}")
                    else:
                        status = "ready_for_apply" if easy_apply else "skipped_not_easy_apply"
                        easy_apply_flag = easy_apply

                results.append(
                    JobCardResult(
                        keyword=keyword,
                        title=title_l,
                        company=company_l,
                        easy_apply=easy_apply_flag,
                        status=status,
                    )
                )
                exp_note = (
                    "unknown_allowed"
                    if min_required is None
                    else f"max_ok<={max_accepted_min_experience_years}"
                )
                print(
                    f"[{keyword}] {title_l} @ {company_l} | "
                    f"keyword_match={matches_keyword} | "
                    f"min_required_years={min_required} | experience_ok={experience_ok} ({exp_note}) | "
                    f"easy_apply={easy_apply}"
                )
            except Exception as exc:
                if playwright_fatal(exc):
                    print(f"[{keyword}] Browser closed during job card handling.")
                    return results
                results.append(
                    JobCardResult(
                        keyword=keyword,
                        title="unknown",
                        company="unknown",
                        easy_apply=False,
                        status="scan_error",
                    )
                )
                print(f"[{keyword}] Failed to parse a job card.")

            idx += 1

        print(f"[{keyword}] Page {page_index}: matched jobs ready/checkable = {page_matches}")

        if max_jobs > 0 and len(results) >= max_jobs:
            break
        if not page_is_open(page):
            return results
        if not _go_to_next_page(page):
            break
        page_index += 1
        _human_delay(0.8, 1.6)

    return results


def _jobs_list_scroll_root(page: Page):
    """Scroll only inside the jobs list column — never the full-page scroll that hits pagination."""
    for sel in (
        "div.jobs-search-results-list",
        "div.scaffold-layout__list",
        "ul.scaffold-layout__list-container",
    ):
        loc = page.locator(sel).first
        if loc.count() > 0 and loc.is_visible():
            return loc
    return None


def _click_see_more_jobs_if_present(page: Page) -> bool:
    show_more = page.locator(
        "button:has-text('See more jobs'), button:has-text('Show more jobs')"
    ).first
    if show_more.count() == 0 or not show_more.is_visible():
        return False
    try:
        show_more.click(timeout=1500)
        _human_delay(0.6, 1.2)
        return True
    except Exception:
        return False


def _ensure_job_card_index_ready(page: Page, card_selector: str, target_index: int) -> bool:
    """
    Gently load list until nth(target_index) exists. Uses small wheels on the jobs list only.
    Returns False if no more cards can be revealed for this index.
    """
    stagnant = 0
    max_attempts = 40
    for _ in range(max_attempts):
        if not page_is_open(page):
            return False
        try:
            count = page.locator(card_selector).count()
        except Exception as exc:
            if playwright_fatal(exc):
                return False
            raise
        if count > target_index:
            return True

        _click_see_more_jobs_if_present(page)

        root = _jobs_list_scroll_root(page)
        try:
            if root is not None:
                root.hover()
                page.mouse.wheel(0, 900)
            else:
                page.mouse.wheel(0, 600)
        except Exception as exc:
            if playwright_fatal(exc):
                return False
            try:
                page.mouse.wheel(0, 600)
            except Exception:
                return False

        _human_delay(0.35, 0.7)
        try:
            new_count = page.locator(card_selector).count()
        except Exception as exc:
            if playwright_fatal(exc):
                return False
            raise
        if new_count <= count:
            stagnant += 1
            if stagnant >= 6:
                return new_count > target_index
        else:
            stagnant = 0

    try:
        return page.locator(card_selector).count() > target_index
    except Exception as exc:
        if playwright_fatal(exc):
            return False
        raise


def _go_to_next_page(page: Page) -> bool:
    if not page_is_open(page):
        return False
    # Scope strictly to LinkedIn jobs search pagination — never generic :has-text('Next')
    # (Easy Apply and other modals use "Next" and would break ordering).
    pag = page.locator(
        "div.jobs-search-pagination, "
        "nav.jobs-search-pagination, "
        "div[class*='jobs-search-pagination']"
    )
    candidates = pag.locator(
        "button[aria-label='View next page'], "
        "button[aria-label='Next'], "
        "button.jobs-search-pagination__button--next"
    )
    try:
        btn = candidates.first
        if btn.count() == 0:
            return False
        if not btn.is_visible() or btn.is_disabled():
            return False
        btn.click(timeout=2000)
        page.wait_for_load_state("domcontentloaded", timeout=8000)
        print("[SCAN] Moved to next results page (pagination only)")
        return True
    except Exception as exc:
        if playwright_fatal(exc):
            print("[SCAN] Browser closed; cannot change results page.")
            return False
        return False


def _extract_job_description(page: Page) -> str:
    show_more = page.locator(
        "button.jobs-description__footer-button, "
        "button:has-text('Show more'), "
        "button:has-text('see more')"
    ).first
    if show_more.count() > 0 and show_more.is_visible():
        try:
            show_more.click(timeout=1500)
            time.sleep(0.4)
        except Exception:
            pass

    selectors = [
        "div.jobs-description__container",
        "div.jobs-box__html-content",
        "div#job-details",
    ]
    for selector in selectors:
        locator = page.locator(selector).first
        if locator.count() == 0:
            continue
        try:
            text = locator.inner_text(timeout=4000).strip()
            if text:
                return " ".join(text.split())
        except Exception:
            continue
    return ""


def _extract_min_required_years(description_text: str) -> int | None:
    text = description_text.lower()
    if not text:
        return None

    if any(token in text for token in ["no experience", "zero experience", "fresh graduate", "fresh grad"]):
        return 0

    range_patterns = [
        r"(\d+)\s*(?:\+)?\s*[-–]\s*(\d+)\s*(?:years|year|yrs|yr)",
        r"(\d+)\s*(?:\+)?\s*(?:to)\s*(\d+)\s*(?:years|year|yrs|yr)",
    ]
    for pattern in range_patterns:
        match = re.search(pattern, text)
        if match:
            first = int(match.group(1))
            second = int(match.group(2))
            return min(first, second)

    single_patterns = [
        r"(?:at least|min(?:imum)?|minimum of|with)\s*(\d+)\s*(?:\+)?\s*(?:years|year|yrs|yr)",
        r"(\d+)\s*(?:\+)?\s*(?:years|year|yrs|yr)\s*(?:of)?\s*experience",
        r"experience\s*[:\-]?\s*(\d+)\s*(?:\+)?\s*(?:years|year|yrs|yr)",
    ]
    for pattern in single_patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))

    return None
