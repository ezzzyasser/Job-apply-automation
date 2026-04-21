import json
from pathlib import Path


class MemoryStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._memory = self._load()

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
            return {}

    def _save(self) -> None:
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(self._memory, file, indent=2, ensure_ascii=True)

    @staticmethod
    def build_key(question: str, field_type: str) -> str:
        normalized = " ".join(question.strip().lower().split())
        return f"{field_type}::{normalized}"

    def get(self, question: str, field_type: str) -> str | None:
        return self._memory.get(self.build_key(question, field_type))

    def set(self, question: str, field_type: str, answer: str) -> None:
        self._memory[self.build_key(question, field_type)] = answer
        self._save()
