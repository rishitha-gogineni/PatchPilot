from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlRunLogger:
    """Append-only, structured run events with no source-content capture."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_repository(cls, repository: Path) -> "JsonlRunLogger":
        return cls(repository.expanduser().resolve() / ".patchpilot" / "runs.jsonl")

    def record(self, event: str, **fields: Any) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def tail(self, limit: int = 20) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines]
