import json
from pathlib import Path

from patchpilot.logging import JsonlRunLogger


def test_logger_writes_structured_events_without_content(tmp_path: Path) -> None:
    logger = JsonlRunLogger.for_repository(tmp_path)
    logger.record("edit_applied", path="src/app.py", changed=True)
    event = logger.tail(1)[0]
    assert event["event"] == "edit_applied"
    assert event["path"] == "src/app.py"
    assert "timestamp" in event
    assert event["run_id"]
    assert event["trace_id"]
    assert json.loads((tmp_path / ".patchpilot" / "runs.jsonl").read_text()) == event
