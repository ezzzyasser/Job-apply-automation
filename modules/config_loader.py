import json
from dataclasses import dataclass
from pathlib import Path

from modules.ai_job_content import OllamaSettings


@dataclass(frozen=True)
class AppConfig:
    email: str
    password: str
    keywords: list[str]
    location: str
    headless: bool
    storage_state_path: str
    login_url: str
    max_jobs_per_run: int
    memory_path: str
    run_log_path: str
    ai_cv_path: str
    general_cv_path: str
    min_delay_seconds: float
    max_delay_seconds: float
    max_accepted_min_experience_years: int
    user_profile_path: str
    ollama: OllamaSettings | None


def load_config(path: str) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    keywords = raw.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [keywords]

    ollama_raw = raw.get("ollama")
    ollama: OllamaSettings | None = None
    if isinstance(ollama_raw, dict) and bool(ollama_raw.get("enabled")):
        ollama = OllamaSettings(
            base_url=str(ollama_raw.get("base_url", "http://127.0.0.1:11434")).strip()
            or "http://127.0.0.1:11434",
            model=str(ollama_raw.get("model", "llama3.1")).strip() or "llama3.1",
            timeout_seconds=float(ollama_raw.get("timeout_seconds", 120)),
            max_description_chars=int(ollama_raw.get("max_description_chars", 8000)),
            auto_cover_letter=bool(ollama_raw.get("auto_cover_letter", True)),
            auto_tailor_fields=bool(ollama_raw.get("auto_tailor_fields", True)),
            output_tailored_dir=str(ollama_raw.get("output_tailored_dir", "tailored_out")).strip(),
        )

    return AppConfig(
        email=raw.get("email", ""),
        password=raw.get("password", ""),
        keywords=keywords,
        location=raw.get("location", ""),
        headless=bool(raw.get("headless", False)),
        storage_state_path=raw.get("storage_state_path", "auth.json"),
        login_url=raw.get("login_url", "https://www.linkedin.com/login"),
        max_jobs_per_run=int(raw.get("max_jobs_per_run", 10)),
        memory_path=raw.get("memory_path", "memory.json"),
        run_log_path=raw.get("run_log_path", "run_logs.jsonl"),
        ai_cv_path=raw.get("ai_cv_path", ""),
        general_cv_path=raw.get("general_cv_path", ""),
        min_delay_seconds=float(raw.get("min_delay_seconds", 1.0)),
        max_delay_seconds=float(raw.get("max_delay_seconds", 2.2)),
        max_accepted_min_experience_years=int(raw.get("max_accepted_min_experience_years", 1)),
        user_profile_path=str(raw.get("user_profile_path", "profile.txt")).strip(),
        ollama=ollama,
    )
