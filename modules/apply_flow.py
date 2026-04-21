import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import Locator, Page, TimeoutError

from modules.ai_job_content import (
    AiContentState,
    OllamaSettings,
    ensure_tailored_package,
    pick_text_answer_from_ai,
    write_tailored_artifacts,
)
from modules.memory_store import MemoryStore
from modules.safe_page import page_is_open, playwright_fatal


@dataclass(frozen=True)
class ApplyConfig:
    ai_cv_path: str
    general_cv_path: str
    min_delay_seconds: float
    max_delay_seconds: float
    ollama: OllamaSettings | None = None
    user_profile: str = ""


def _normalize_role_text(value: str) -> str:
    normalized = " ".join(value.lower().split())
    normalized = normalized.replace("artificial intelligence", "ai")
    normalized = normalized.replace("machine learning", "ml")
    return normalized


def resolve_cv_path_for_job(job_title: str, cfg: ApplyConfig) -> str:
    """Same rules as resume upload (AI/ML role → ai_cv_path)."""
    job_lower = _normalize_role_text(job_title)
    preferred = cfg.ai_cv_path if ("ai" in job_lower or "ml" in job_lower) else cfg.general_cv_path
    fallback = cfg.general_cv_path or cfg.ai_cv_path
    return (preferred or fallback or "").strip()


def _sleep(min_seconds: float, max_seconds: float) -> None:
    time.sleep(random.uniform(min_seconds, max_seconds))


def _safe_text(locator: Locator) -> str:
    try:
        return locator.inner_text(timeout=500).strip()
    except Exception:
        return ""


def _parse_choice_index(raw: str, lines: list[str]) -> int | None:
    """
    Map user input to a zero-based index.
    Accepts: 1,2,3…, full option text, yes/no/y/n (when exactly two options look binary).
    """
    if not lines or not raw.strip():
        return None
    raw = raw.strip()
    if raw.isdigit():
        n = int(raw)
        if 1 <= n <= len(lines):
            return n - 1
        return None

    raw_l = raw.lower()
    # Exact line match
    for i, line in enumerate(lines):
        if line.strip().lower() == raw_l:
            return i

    # Yes / No shortcuts (common LinkedIn questions)
    if len(lines) == 2:
        a, b = lines[0].strip().lower(), lines[1].strip().lower()
        yn_first = a in {"yes", "y"} and b in {"no", "n"}
        yn_second = a in {"no", "n"} and b in {"yes", "y"}
        if raw_l in {"yes", "y", "true", "1"}:
            if yn_first:
                return 0
            if yn_second:
                return 1
        if raw_l in {"no", "n", "false", "2"}:
            if yn_first:
                return 1
            if yn_second:
                return 0
        # Options shown as "1) Yes" style already handled by digit; allow word on either side
        for i, line in enumerate(lines):
            if raw_l in {"yes", "y", "true"} and "yes" in line.lower():
                return i
            if raw_l in {"no", "n", "false"} and "no" in line.lower():
                return i

    # Substring fallback (last resort)
    for i, line in enumerate(lines):
        if raw_l and raw_l in line.lower():
            return i
    return None


def _prompt_numbered_choice(question: str, lines: list[str]) -> int | None:
    """
    Show 1-based numbered choices; user may reply with 1, 2, 3, yes/no, or option text.
    Returns zero-based index, or None if empty / invalid.
    """
    if not lines:
        return None
    print(f"\n{question}")
    for i, line in enumerate(lines, start=1):
        print(f"  {i}) {line}")
    raw = input(
        f"Choose 1-{len(lines)} (number), or type yes/no / option text: "
    ).strip()
    return _parse_choice_index(raw, lines)


def _question_for_input(modal: Locator, field: Locator, fallback: str) -> str:
    input_id = field.get_attribute("id") or ""
    aria = field.get_attribute("aria-label") or ""
    placeholder = field.get_attribute("placeholder") or ""
    name = field.get_attribute("name") or ""

    label_text = ""
    if input_id:
        label_text = _safe_text(modal.locator(f"label[for='{input_id}']").first)

    question = label_text or aria or placeholder or name or fallback
    question = " ".join(question.split())
    return question or fallback


def _fill_text_like_fields(
    modal: Locator,
    memory: MemoryStore,
    cfg: ApplyConfig,
    job_title: str,
    company: str,
    job_description: str,
    ai_state: AiContentState,
) -> None:
    text_selector = (
        "input:not([type='hidden']):not([type='radio']):not([type='checkbox']):"
        "not([type='file']):not([type='submit']):not([type='button']):not([disabled]),"
        "textarea:not([disabled])"
    )
    fields = modal.locator(text_selector)
    count = fields.count()
    ollama = cfg.ollama
    for idx in range(count):
        field = fields.nth(idx)
        tag = (field.evaluate("el => el.tagName.toLowerCase()") or "input").strip()
        input_type = (field.get_attribute("type") or "text").lower()

        current_value = (field.input_value() or "").strip()
        if current_value:
            continue

        question = _question_for_input(modal, field, f"{tag}_{idx}")
        key_type = f"{tag}:{input_type}"
        answer = memory.get(question, key_type)
        if answer is None and ollama is not None:
            ai_answer = pick_text_answer_from_ai(
                ollama,
                ai_state,
                question,
                cfg.user_profile,
                job_title,
                company,
                job_description,
            )
            if ai_answer:
                answer = ai_answer
        if answer is None:
            answer = input(f"Answer for '{question}': ").strip()
            if answer:
                memory.set(question, key_type, answer)
        if answer:
            field.fill(answer)


def _collect_select_option_rows(select: Locator) -> list[dict]:
    """Each row: display (shown in list), value (for select_option), label (fallback)."""
    rows: list[dict] = []
    options = select.locator("option")
    for opt_idx in range(options.count()):
        opt = options.nth(opt_idx)
        value = (opt.get_attribute("value") or "").strip()
        text = _safe_text(opt)
        display = (text or value).strip()
        low = display.lower()
        if not display or low in {"select an option", "choose an option", ""}:
            continue
        rows.append({"display": display, "value": value, "label": text})
    return rows


def _fill_select_fields(modal: Locator, memory: MemoryStore) -> None:
    selects = modal.locator("select:not([disabled])")
    for idx in range(selects.count()):
        select = selects.nth(idx)
        if (select.input_value() or "").strip():
            continue

        question = _question_for_input(modal, select, f"select_{idx}")
        answer = memory.get(question, "select")
        rows = _collect_select_option_rows(select)
        lines = [r["display"] for r in rows]
        if answer is None:
            pick = _prompt_numbered_choice(f"Dropdown: {question}", lines)
            if pick is None:
                continue
            row = rows[pick]
            token = row["value"] if row["value"] else (row["label"] or row["display"])
            answer = token
            if answer:
                memory.set(question, "select", answer)
        elif lines:
            pick = _parse_choice_index(answer, lines)
            if pick is not None:
                row = rows[pick]
                answer = row["value"] if row["value"] else (row["label"] or row["display"])
        if answer:
            try:
                select.select_option(answer)
            except Exception:
                try:
                    select.select_option(label=answer)
                except Exception:
                    for row in _collect_select_option_rows(select):
                        if row["display"] == answer or row["value"] == answer:
                            if row["value"]:
                                select.select_option(value=row["value"])
                            else:
                                select.select_option(label=row["label"] or row["display"])
                            break


def _fill_radio_fields(modal: Locator, memory: MemoryStore) -> None:
    radios = modal.locator("input[type='radio']:not([disabled])")
    handled_groups: set[str] = set()
    for idx in range(radios.count()):
        radio = radios.nth(idx)
        group_name = (radio.get_attribute("name") or f"group_{idx}").strip()
        if group_name in handled_groups:
            continue
        handled_groups.add(group_name)

        group = modal.locator(f"input[type='radio'][name='{group_name}']")
        if group.count() == 0:
            continue
        if modal.locator(f"input[type='radio'][name='{group_name}']:checked").count() > 0:
            continue

        question = _question_for_input(modal, group.first, group_name)
        answer = memory.get(question, "radio")

        lines: list[str] = []
        values_ordered: list[str] = []
        for opt_idx in range(group.count()):
            option = group.nth(opt_idx)
            value = (option.get_attribute("value") or "").strip()
            display = ""
            try:
                display = (
                    option.evaluate(
                        """el => {
                            const lab = el.closest('label');
                            if (lab) return lab.innerText.trim().replace(/\\s+/g, ' ');
                            return (el.getAttribute('value') || el.value || '').trim();
                          }"""
                    )
                    or ""
                ).strip()
            except Exception:
                display = ""
            if not display:
                display = value or f"Option {opt_idx + 1}"
            lines.append(display)
            values_ordered.append(value or display)

        if answer is None:
            pick = _prompt_numbered_choice(f"Radio: {question}", lines)
            if pick is None:
                continue
            answer = values_ordered[pick]
            memory.set(question, "radio", answer)
        else:
            pick = _parse_choice_index(answer, lines)
            if pick is not None:
                answer = values_ordered[pick]

        if not answer:
            continue

        selected = False
        for opt_idx in range(group.count()):
            option = group.nth(opt_idx)
            value = (option.get_attribute("value") or "").strip()
            lab = lines[opt_idx] if opt_idx < len(lines) else ""
            if (
                value.lower() == answer.lower()
                or lab.lower() == answer.lower()
                or answer.lower() == (value or lab).lower()
            ):
                option.check()
                selected = True
                break
        if not selected and answer.strip().isdigit():
            n = int(answer.strip())
            if 1 <= n <= group.count():
                group.nth(n - 1).check()
                selected = True
        if not selected and group.count() > 0:
            group.nth(0).check()


def _fill_role_radiogroups(modal: Locator, memory: MemoryStore) -> None:
    """LinkedIn-style [role=radiogroup] with [role=radio] tiles (not always <input type=radio>)."""
    groups = modal.locator('[role="radiogroup"]')
    for gi in range(groups.count()):
        g = groups.nth(gi)
        opts = g.locator('[role="radio"]')
        if opts.count() < 2:
            continue
        answered = False
        for j in range(opts.count()):
            if (opts.nth(j).get_attribute("aria-checked") or "").lower() == "true":
                answered = True
                break
        if answered:
            continue

        question = _safe_text(g.locator("legend, [data-test-form-element-label]").first) or f"radiogroup_{gi}"
        lines = [_safe_text(opts.nth(j)) or f"Option {j + 1}" for j in range(opts.count())]
        if not lines:
            continue

        answer = memory.get(question, "role_radio")
        if answer is None:
            pick = _prompt_numbered_choice(f"Choice (tiles): {question}", lines)
            if pick is None:
                continue
            memory.set(question, "role_radio", lines[pick])
        else:
            pick = _parse_choice_index(answer, lines)
            if pick is None:
                continue
        opts.nth(pick).click(timeout=5000)


def _fill_yes_no_fieldsets(modal: Locator, memory: MemoryStore) -> None:
    """Yes / No as separate buttons inside a fieldset (common in Easy Apply)."""
    count = modal.locator("fieldset").count()
    for i in range(count):
        fs = modal.locator("fieldset").nth(i)
        yes_btn = fs.get_by_role("button", name=re.compile(r"^\s*yes\s*$", re.I)).first
        no_btn = fs.get_by_role("button", name=re.compile(r"^\s*no\s*$", re.I)).first
        if yes_btn.count() == 0 or no_btn.count() == 0:
            continue
        try:
            if yes_btn.is_disabled() and no_btn.is_disabled():
                continue
        except Exception:
            pass

        if (yes_btn.get_attribute("aria-pressed") or "") == "true":
            continue
        if (no_btn.get_attribute("aria-pressed") or "") == "true":
            continue

        question = _safe_text(fs.locator("legend").first) or f"yes_no_{i}"
        lines = ["Yes", "No"]
        answer = memory.get(question, "yes_no")
        if answer is None:
            pick = _prompt_numbered_choice(f"Yes / No: {question}", lines)
            if pick is None:
                continue
            memory.set(question, "yes_no", lines[pick])
        else:
            pick = _parse_choice_index(answer, lines)
            if pick is None:
                continue
        if pick == 0:
            yes_btn.click(timeout=5000)
        else:
            no_btn.click(timeout=5000)


def _upload_cv_if_needed(
    modal: Locator, job_title: str, cfg: ApplyConfig, ai_state: AiContentState
) -> None:
    file_input = modal.locator("input[type='file']").first
    if file_input.count() == 0:
        return

    cv_path = resolve_cv_path_for_job(job_title, cfg)
    if not cv_path:
        print("CV upload requested but cv path is missing in config.")
        return

    full_path = Path(cv_path).expanduser().resolve()
    if not full_path.exists():
        print(f"CV file not found: {full_path}")
        return

    file_input.set_input_files(str(full_path))
    print(f"Uploaded CV: {full_path.name}")
    if (
        cfg.ollama
        and ai_state._package
        and (ai_state._package.get("keyword_phrases") or ai_state._package.get("professional_summary"))
    ):
        print(
            "[AI] Suggested resume keywords for this job were written under output_tailored_dir "
            "(if set) — mirror those terms in your PDF/LI profile; the uploader used your file as-is."
        )


def _apply_modal_locator(page: Page) -> Locator:
    """Visible apply dialog — prefer Easy Apply shell."""
    scoped = page.locator('[role="dialog"]:visible').filter(
        has=page.locator(".jobs-easy-apply-modal, .jobs-easy-apply-content, form")
    )
    if scoped.count() > 0:
        return scoped.last
    return page.locator('[role="dialog"]:visible').last


def _click_button_in_scope(scope: Locator, labels: list[str]) -> bool:
    for label in labels:
        btn = scope.locator(f"button:has-text('{label}')").first
        if btn.count() > 0 and btn.is_visible():
            btn.click()
            return True
    return False


def _click_modal_footer_next_or_review(modal: Locator) -> bool:
    footer_selectors = (
        ".jobs-easy-apply-modal__footer",
        ".jobs-easy-apply-content__footer",
        "footer",
        ".artdeco-modal__actionbar",
    )
    labels_order = ("Continue to next step", "Continue", "Next", "Review")
    for fs in footer_selectors:
        footer = modal.locator(fs).first
        if footer.count() == 0:
            continue
        try:
            if not footer.is_visible():
                continue
        except Exception:
            continue
        for label in labels_order:
            try:
                btn = footer.get_by_role(
                    "button", name=re.compile("^" + re.escape(label) + "$", re.I)
                ).first
                if btn.count() > 0 and btn.is_visible():
                    btn.click(timeout=5000)
                    return True
            except Exception:
                pass
        for label in labels_order:
            primary = footer.locator(f"button.artdeco-button--primary:has-text('{label}')").first
            if primary.count() > 0 and primary.is_visible():
                primary.click(timeout=5000)
                return True
            loose = footer.locator(f"button:has-text('{label}')").first
            if loose.count() > 0 and loose.is_visible():
                loose.click(timeout=5000)
                return True
    return False


def _close_modal_without_submit(page: Page, modal: Locator) -> None:
    if _click_button_in_scope(modal, ["Dismiss", "Cancel", "Close"]):
        _sleep(0.3, 0.8)
    discard = page.locator("button:has-text('Discard')").first
    if discard.count() > 0 and discard.is_visible():
        discard.click()


def _dismiss_stuck_overlays(page: Page) -> None:
    try:
        page.keyboard.press("Escape")
        time.sleep(0.35)
    except Exception:
        pass


def _easy_apply_overlay_visible(page: Page) -> bool:
    if not page_is_open(page):
        return False
    try:
        if page.locator(".jobs-easy-apply-modal:visible").count() > 0:
            return True
        return (
            page.locator(
                '[role="dialog"]:visible:has(.jobs-easy-apply-modal), '
                '[role="dialog"]:visible:has(.jobs-easy-apply-content)'
            ).count()
            > 0
        )
    except Exception:
        return False


def _ensure_easy_apply_popup_closed(page: Page) -> None:
    """After submit or any terminal apply path, clear overlays so the next job card is usable."""
    if not page_is_open(page):
        return
    for _ in range(12):
        if not _easy_apply_overlay_visible(page):
            return
        body = page.locator("body")
        if _click_button_in_scope(body, ["Done", "Dismiss", "Not now", "Close"]):
            _sleep(0.35, 0.75)
            continue
        try:
            _close_modal_without_submit(page, _apply_modal_locator(page))
        except Exception:
            pass
        _dismiss_stuck_overlays(page)


def _dismiss_visible_confirmation_dialogs(page: Page, max_rounds: int = 14) -> None:
    """
    LinkedIn often shows a separate success/confirmation [role=dialog] after submit
    (no .jobs-easy-apply-modal). Close it so the job list is usable again.
    """
    if not page_is_open(page):
        return
    no_click_streak = 0
    for _ in range(max_rounds):
        if not page_is_open(page):
            return
        try:
            dialogs = page.locator('[role="dialog"]:visible')
            if dialogs.count() == 0:
                return
        except Exception:
            return

        dlg = dialogs.last
        clicked = False
        for pattern in (
            r"^\s*Done\s*$",
            r"^\s*OK\s*$",
            r"^\s*Got it\s*$",
            r"^\s*Close\s*$",
            r"^\s*Dismiss\s*$",
            r"^\s*No thanks\s*$",
            r"^\s*Not now\s*$",
        ):
            try:
                btn = dlg.get_by_role("button", name=re.compile(pattern, re.I)).first
                if btn.count() > 0 and btn.is_visible():
                    btn.click(timeout=5000)
                    clicked = True
                    _sleep(0.35, 0.7)
                    break
            except Exception:
                pass

        if not clicked:
            for sel in (
                "button.artdeco-modal__dismiss",
                "button[data-test-modal-close-btn]",
                'button[aria-label="Dismiss"]',
                "button.modal__dismiss",
            ):
                try:
                    b = dlg.locator(sel).first
                    if b.count() > 0 and b.is_visible():
                        b.click(timeout=5000)
                        clicked = True
                        _sleep(0.35, 0.7)
                        break
                except Exception:
                    pass

        if not clicked:
            body = page.locator("body")
            if _click_button_in_scope(body, ["Done", "OK", "Got it", "Dismiss", "Close", "Not now"]):
                clicked = True
                _sleep(0.35, 0.7)

        if not clicked:
            no_click_streak += 1
            _dismiss_stuck_overlays(page)
            if no_click_streak >= 3:
                return
        else:
            no_click_streak = 0


def _ensure_apply_ui_fully_closed(page: Page) -> None:
    """Easy Apply shell + post-submit confirmation dialogs."""
    if not page_is_open(page):
        return
    _ensure_easy_apply_popup_closed(page)
    _dismiss_visible_confirmation_dialogs(page)


def _locate_easy_apply_on_job_page(page: Page) -> Locator:
    """Job detail panel apply button only (not list duplicates)."""
    return (
        page.locator(
            "div.jobs-search__job-details--wrapper button.jobs-apply-button--top-card, "
            "div.jobs-details-top-card__main-actions button.jobs-apply-button, "
            "button.jobs-apply-button--top-card"
        )
        .filter(has_text=re.compile(r"easy\s*apply", re.I))
        .first
    )


def _wait_apply_dialog(page: Page, timeout_ms: int = 28000) -> bool:
    """Wait until an apply dialog is visible (attached + painted)."""
    try:
        page.get_by_role("dialog").first.wait_for(state="visible", timeout=timeout_ms)
        return True
    except TimeoutError:
        pass
    try:
        page.locator(".jobs-easy-apply-modal").first.wait_for(state="visible", timeout=5000)
        return True
    except TimeoutError:
        return False


def run_easy_apply(
    page: Page,
    memory: MemoryStore,
    cfg: ApplyConfig,
    job_title: str,
    company: str,
    job_description: str = "",
    pre_tailored: dict | None = None,
    cv_excerpt: str = "",
) -> str:
    if not page_is_open(page):
        return "page_closed"

    easy_apply_btn = _locate_easy_apply_on_job_page(page)
    try:
        easy_apply_btn.wait_for(state="visible", timeout=12000)
    except TimeoutError:
        easy_apply_btn = page.locator("button.jobs-apply-button").filter(
            has_text=re.compile(r"easy\s*apply", re.I)
        ).first
        try:
            easy_apply_btn.wait_for(state="visible", timeout=5000)
        except TimeoutError:
            return "skipped_not_easy_apply"

    print("[APPLY] Clicking Easy Apply on job detail (then modal Next only inside dialog).")
    try:
        easy_apply_btn.scroll_into_view_if_needed()
        easy_apply_btn.click(timeout=10000)
    except Exception as exc:
        if playwright_fatal(exc):
            return "page_closed"
        return "skipped_not_easy_apply"

    time.sleep(0.9)
    if not page_is_open(page):
        return "page_closed"

    if not _wait_apply_dialog(page):
        _dismiss_stuck_overlays(page)
        return "apply_modal_not_opened"

    ai_state = AiContentState()
    ai_state.cv_excerpt = (cv_excerpt or "").strip()
    if pre_tailored is not None:
        ai_state._package = pre_tailored
        ai_state._tried = True
    elif cfg.ollama and (cfg.ollama.output_tailored_dir or "").strip() and not ai_state._package:
        ensure_tailored_package(
            ai_state,
            cfg.ollama,
            cfg.user_profile,
            job_title,
            company,
            job_description,
            cv_excerpt=ai_state.cv_excerpt,
        )
    try:
        time.sleep(0.45)
        print(f"[APPLY] Dialog visible — {job_title} @ {company}")

        max_steps = 12
        for step in range(1, max_steps + 1):
            if not page_is_open(page):
                return "page_closed"

            try:
                modal = _apply_modal_locator(page)
                print(f"[APPLY] Step {step}: fill (modal scope only)")
                _sleep(cfg.min_delay_seconds, cfg.max_delay_seconds)
                _fill_text_like_fields(
                    modal, memory, cfg, job_title, company, job_description, ai_state
                )
                _fill_select_fields(modal, memory)
                _fill_radio_fields(modal, memory)
                _fill_role_radiogroups(modal, memory)
                _fill_yes_no_fieldsets(modal, memory)
                _upload_cv_if_needed(modal, job_title, cfg, ai_state)
                _sleep(0.35, 0.9)

                modal = _apply_modal_locator(page)
                submit_btn = modal.locator("button:has-text('Submit application')").first
                if submit_btn.count() > 0 and submit_btn.is_visible():
                    decision = input(
                        f"Submit application for '{job_title}' at '{company}'? (y/n): "
                    ).strip().lower()
                    if decision != "y":
                        _close_modal_without_submit(page, modal)
                        return "stopped_by_user_before_submit"
                    submit_btn.click()
                    _sleep(1.0, 2.2)
                    _click_button_in_scope(page.locator("body"), ["Done", "OK", "Got it"])
                    _dismiss_visible_confirmation_dialogs(page, max_rounds=8)
                    return "applied"

                modal = _apply_modal_locator(page)
                if _click_modal_footer_next_or_review(modal):
                    print("[APPLY] Modal footer: Next / Continue / Review")
                    _sleep(0.45, 1.0)
                    continue

                modal = _apply_modal_locator(page)
                if _click_button_in_scope(modal, ["Save application"]):
                    _sleep(0.6, 1.2)
                    _click_button_in_scope(page.locator("body"), ["Done", "OK", "Got it"])
                    _dismiss_visible_confirmation_dialogs(page, max_rounds=8)
                    return "saved_not_submitted"

                _close_modal_without_submit(page, modal)
                return "unknown_modal_state"
            except Exception as exc:
                if playwright_fatal(exc):
                    return "page_closed"
                print(f"[APPLY] Step error: {exc}")
                _dismiss_stuck_overlays(page)
                return "apply_error"

        _dismiss_stuck_overlays(page)
        return "max_steps_reached"
    finally:
        if cfg.ollama and ai_state._package is not None:
            write_tailored_artifacts(cfg.ollama, job_title, company, ai_state)
        if page_is_open(page):
            _ensure_apply_ui_fully_closed(page)
