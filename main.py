import random
import time

from modules.ai_job_content import load_user_profile_file
from modules.apply_flow import ApplyConfig
from modules.browser import launch_browser
from modules.config_loader import load_config
from modules.job_search import scan_jobs_for_keyword
from modules.logger import append_run_log
from modules.memory_store import MemoryStore
from modules.session import open_or_create_session


def main() -> None:
    config = load_config("config.json")
    context_manager = launch_browser(headless=config.headless)
    memory = MemoryStore(config.memory_path)
    profile = load_user_profile_file(config.user_profile_path)

    if config.ollama and not profile:
        print(
            f"[AI] Ollama is enabled: add a short bio/skills to {config.user_profile_path!r} "
            "so cover letters match you."
        )

    apply_cfg = ApplyConfig(
        ai_cv_path=config.ai_cv_path,
        general_cv_path=config.general_cv_path,
        min_delay_seconds=config.min_delay_seconds,
        max_delay_seconds=config.max_delay_seconds,
        ollama=config.ollama,
        user_profile=profile,
    )

    with context_manager as browser:
        page = open_or_create_session(
            browser=browser,
            storage_state_path=config.storage_state_path,
            login_url=config.login_url,
            email=config.email,
            password=config.password,
        )

        page.goto("https://www.linkedin.com/jobs/")

        # 🔥 NEW DELAY (page load stabilization)
        time.sleep(random.uniform(4, 7))

        print("Browser and session are ready. Starting job list scan...")

        scanned_total = 0
        applied_total = 0
        unlimited_run = config.max_jobs_per_run <= 0

        for keyword in config.keywords:

            # 🔥 NEW DELAY (between keywords)
            time.sleep(random.uniform(2, 5))

            if (not unlimited_run) and scanned_total >= config.max_jobs_per_run:
                break

            print(f"\n[SCAN] Searching keyword: {keyword}")

            remaining = 0 if unlimited_run else (config.max_jobs_per_run - scanned_total)

            # 🔥 NEW DELAY (before scanning starts)
            time.sleep(random.uniform(2, 4))

            results = scan_jobs_for_keyword(
                page=page,
                keyword=keyword,
                location=config.location,
                max_jobs=remaining,
                max_accepted_min_experience_years=config.max_accepted_min_experience_years,
                memory=memory,
                apply_cfg=apply_cfg,
            )

            scanned_total += len(results)

            for item in results:

                # 🔥 NEW DELAY (before processing each job)
                time.sleep(random.uniform(1.5, 3.5))

                print(f"[JOB] {item.title} @ {item.company} -> {item.status}")

                if item.status == "applied":
                    applied_total += 1

                print(f"[JOB] Final status: {item.status}")

                append_run_log(
                    config.run_log_path,
                    {
                        "event": "job_processed",
                        "keyword": item.keyword,
                        "job_title": item.title,
                        "company": item.company,
                        "easy_apply": item.easy_apply,
                        "status": item.status,
                    },
                )

                # 🔥 EXISTING DELAY (kept but slightly more human)
                time.sleep(random.uniform(
                    config.min_delay_seconds + 0.5,
                    config.max_delay_seconds + 2
                ))

        print(f"Run complete. Jobs scanned: {scanned_total}, applied: {applied_total}")
        input("Press Enter to close...")

    append_run_log(
        config.run_log_path,
        {
            "event": "run_complete",
            "keywords": config.keywords,
            "location": config.location,
            "max_jobs_per_run": config.max_jobs_per_run,
            "memory_path": config.memory_path,
        },
    )


if __name__ == "__main__":
    main()