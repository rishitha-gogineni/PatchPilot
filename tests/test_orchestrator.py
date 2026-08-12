from pathlib import Path

from patchpilot.orchestrator import run_test_loop


def test_successful_test_does_not_recover(tmp_path: Path) -> None:
    called = []
    summary = run_test_loop(tmp_path, "python -m pytest --version", recovery=lambda: called.append(True) or True)
    assert summary.success is True
    assert summary.recovery_attempted is False
    assert called == []


def test_failed_test_allows_one_approved_recovery(tmp_path: Path) -> None:
    calls = []
    (tmp_path / "test_sample.py").write_text("def test_value():\n    assert False\n", encoding="utf-8")

    def recover() -> bool:
        calls.append(True)
        (tmp_path / "test_sample.py").write_text("def test_value():\n    assert True\n", encoding="utf-8")
        return True

    summary = run_test_loop(tmp_path, "python -m pytest -q", recovery=recover)
    assert summary.success is True
    assert summary.recovery_attempted is True
    assert len(calls) == 1


def test_failed_test_without_approved_recovery_stops(tmp_path: Path) -> None:
    calls = []
    events = []
    summary = run_test_loop(
        tmp_path,
        "python -m pytest --definitely-invalid-option",
        recovery=lambda: calls.append(True) or False,
        logger=lambda event, **fields: events.append((event, fields)),
    )
    assert summary.success is False
    assert summary.recovery_attempted is False
    assert len(calls) == 1
    assert any(event == "recovery_skipped" for event, _ in events)
