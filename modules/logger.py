import json
from datetime import datetime, timezone
from pathlib import Path


def append_run_log(path: str, payload: dict) -> None:
    target = Path(path)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")
