"""
Ollama HTTP client + prompts for per-job cover letter and resume keyword alignment.
Uses stdlib only (urllib); run Ollama locally (e.g. ollama run llama3.1).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str
    model: str
    timeout_seconds: float
    max_description_chars: int
    auto_cover_letter: bool
    auto_tailor_fields: bool
    output_tailored_dir: str


@dataclass
class AiContentState:
    """One generation per run_easy_apply: cover letter, summary, keyword phrases."""

    _package: dict[str, Any] | None = None
    _tried: bool = False
    last_error: str | None = None
    cv_excerpt: str = ""


def _slug_for_filename(company: str, title: str) -> str:
    raw = f"{company}_{title}"
    s = re.sub(r"[^\w\-.]+", "_", raw, flags=re.I)
    return s[:120] or "job"


def _parse_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", raw)
    if not m:
        raise ValueError("no JSON object in model output")
    return json.loads(m.group(0))


def ollama_chat(
    base_url: str, model: str, messages: list[dict[str, str]], timeout_seconds: float
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
    content = (data.get("message") or {}).get("content") or ""
    if not str(content).strip():
        raise ValueError("empty Ollama message content")
    return str(content).strip()


def extract_text_from_cv_file(cv_path: str, max_chars: int = 12000) -> str:
    """
    Read plain text from .txt, or from PDF with optional `pypdf` (`pip install pypdf`).
    """
    p = Path(cv_path).expanduser()
    if not p.exists() or not p.is_file():
        return ""
    suf = p.suffix.lower()
    if suf == ".txt":
        return p.read_text(encoding="utf-8", errors="replace")[:max_chars]
    if suf == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print(
                "[AI] For PDF text extraction, install: pip install pypdf  "
                "(using profile only until then.)"
            )
            return ""
        try:
            reader = PdfReader(str(p))
            parts: list[str] = []
            for page in reader.pages[:8]:
                t = page.extract_text() or ""
                if t:
                    parts.append(t)
            blob = "\n".join(parts)
            return (blob or "")[:max_chars]
        except Exception as exc:
            print(f"[AI] Could not read PDF: {exc}")
            return ""
    return ""


def build_tailored_package(
    ollama: OllamaSettings,
    user_profile: str,
    job_title: str,
    company: str,
    job_description: str,
    cv_excerpt: str = "",
) -> dict[str, Any]:
    cap = ollama.max_description_chars
    desc = (job_description or "").strip()
    if cap > 0 and len(desc) > cap:
        desc = desc[:cap] + "\n[…truncated]"

    cv = (cv_excerpt or "").strip()
    if len(cv) > 6000:
        cv = cv[:6000] + "\n[…]"

    system = (
        "You help job candidates. Reply with ONLY a single JSON object, no markdown fences, "
        "no commentary. Use double-quoted keys and string values. "
        "The JSON must have exactly these keys: "
        '"cover_letter" (string, 3 short paragraphs, plain text, no emojis), '
        '"professional_summary" (string, 2–3 sentences matching the job and true to the candidate), '
        '"keyword_phrases" (JSON array of 8 to 12 short strings: ATS skills/phrases; align with the job '
        "and the candidate's real CV or profile — do not invent degrees or jobs they do not have).\n"
        "If a resume excerpt is present, use it to suggest concrete wording the candidate can add to their PDF resume."
    )
    up = (user_profile or "").strip() or "No profile file; use resume excerpt and job only."
    user = (
        f"Job title: {job_title}\n"
        f"Company: {company}\n"
        f"Candidate profile / notes:\n{up}\n"
    )
    if cv:
        user += f"\nResume or CV text (excerpt; may be partial):\n{cv}\n"
    else:
        user += "\n(Resume text not available — use profile and job only.)\n"
    user += f"\nJob description (may be partial):\n{desc}\n"
    raw = ollama_chat(
        ollama.base_url,
        ollama.model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        ollama.timeout_seconds,
    )
    return _parse_json_object(raw)


def ensure_tailored_package(
    state: AiContentState,
    ollama: OllamaSettings,
    user_profile: str,
    job_title: str,
    company: str,
    job_description: str,
    cv_excerpt: str = "",
) -> dict[str, Any] | None:
    if state._package is not None:
        return state._package
    if state._tried:
        return None
    state._tried = True
    try:
        ex = (cv_excerpt or state.cv_excerpt or "").strip()
        state._package = build_tailored_package(
            ollama, user_profile, job_title, company, job_description, cv_excerpt=ex
        )
        state.last_error = None
        return state._package
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        state._package = None
        state.last_error = str(exc)
        print(f"[AI] Ollama error: {exc}")
        return None


def field_is_cover_letter(question: str) -> bool:
    q = (question or "").lower()
    if any(
        s in q
        for s in (
            "cover letter",
            "letter to the hiring",
            "message to the hiring",
            "message to hiring",
            "introduce yourself",
        )
    ):
        return True
    if "why are you" in q or "why do you" in q:
        return "interested" in q or "applying" in q or "role" in q or "position" in q
    if "additional information" in q and "cover" in q:
        return True
    if re.search(
        r"\b(write|include)\b.*\b(letter|message)\b", q, re.I
    ) or re.search(r"\b(cover|motivation)\b.*\b(letter|statement)\b", q, re.I):
        return True
    return False


def field_wants_tailor_summary(question: str) -> bool:
    q = (question or "").lower()
    if any(
        s in q
        for s in (
            "how do you match",
            "relevant experience",
            "describe your experience",
            "describe your background",
            "summarize your",
            "professional summary",
            "brief summary",
            "why are you a good fit",
            "what makes you",
            "key skills and",
            "top skills",
        )
    ):
        return True
    if "at least" in q and "character" in q and ("summary" in q or "describe" in q):
        return True
    return False


def _field_should_not_use_ai(question: str) -> bool:
    q = (question or "").lower()
    deny = (
        "phone",
        "email",
        "e-mail",
        "url",
        "http",
        "linkedin",
        "twitter",
        "github.com",
        "website",
        "salary",
        "compensation",
        "sponsor",
        "visa",
        "authorization",
        "are you legal",
        "full name",
        "first name",
        "last name",
        "address",
        "postal",
        "zip",
        "city,",
        "country of",
    )
    return any(d in q for d in deny)


def pick_text_answer_from_ai(
    ollama: OllamaSettings,
    state: AiContentState,
    question: str,
    user_profile: str,
    job_title: str,
    company: str,
    job_description: str,
) -> str | None:
    field_lower = (question or "").lower()
    if ollama.auto_cover_letter and field_is_cover_letter(question):
        pkg = ensure_tailored_package(
            state,
            ollama,
            user_profile,
            job_title,
            company,
            job_description,
            cv_excerpt=state.cv_excerpt,
        )
        if pkg:
            s = (pkg.get("cover_letter") or "").strip()
            if s:
                print(f"[AI] Filled cover letter for field: {question[:80]}…")
            return s or None
        return None

    if ollama.auto_tailor_fields and field_wants_tailor_summary(question):
        if _field_should_not_use_ai(question) and "summary" not in field_lower:
            return None
        pkg = ensure_tailored_package(
            state,
            ollama,
            user_profile,
            job_title,
            company,
            job_description,
            cv_excerpt=state.cv_excerpt,
        )
        if pkg:
            s = (pkg.get("professional_summary") or "").strip()
            if s:
                print(f"[AI] Filled professional summary for field: {question[:80]}…")
            return s or None
        return None
    return None


def write_tailored_artifacts(
    ollama: OllamaSettings, job_title: str, company: str, state: AiContentState
) -> None:
    if not ollama.output_tailored_dir or not state._package:
        return
    out_dir = Path(ollama.output_tailored_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug_for_filename(company, job_title)
    data = {
        "company": company,
        "job_title": job_title,
        "cover_letter": state._package.get("cover_letter", ""),
        "professional_summary": state._package.get("professional_summary", ""),
        "keyword_phrases": state._package.get("keyword_phrases", []),
    }
    p = out_dir / f"tailored_{slug}.json"
    p.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    kw = data["keyword_phrases"]
    if isinstance(kw, str):
        lines = [kw]
    else:
        lines = [f"- {x}" for x in (kw or [])] if isinstance(kw, list) else []
    lines_path = out_dir / f"resume_keywords_{slug}.txt"
    lines_path.write_text(
        (data.get("professional_summary") or "")
        + "\n\n# Suggested resume keyword phrases (paste into your CV or skills block):\n"
        + "\n".join(lines),
        encoding="utf-8",
    )
    print(f"[AI] Wrote {p.name} and {lines_path.name} under {out_dir}")


def prepare_job_application_ai(
    ollama: OllamaSettings,
    user_profile: str,
    job_title: str,
    company: str,
    job_description: str,
    cv_path: str,
) -> tuple[dict[str, Any] | None, str]:
    """
    Call before opening Easy Apply: uses job description + CV text (PDF/txt) + profile
    to build cover_letter, professional_summary, and keyword_phrases.
    """
    ex = extract_text_from_cv_file(cv_path) if (cv_path or "").strip() else ""
    if not ex and (user_profile or "").strip():
        ex = (user_profile or "").strip()[:5000]
    try:
        pkg = build_tailored_package(
            ollama,
            user_profile,
            job_title,
            company,
            job_description,
            cv_excerpt=ex,
        )
        print("[AI] Ready — use generated cover letter and keywords in the form as needed.")
        return pkg, ex
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        print(f"[AI] Pre-apply generation failed: {exc}")
        return None, ex


def load_user_profile_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace").strip()
