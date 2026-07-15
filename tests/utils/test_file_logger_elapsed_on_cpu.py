import json

from verl.utils import tracking


def test_file_logger_records_monotonic_elapsed_time(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    monotonic_values = iter([100.0, 105.5])
    monkeypatch.setenv("VERL_FILE_LOGGER_PATH", str(path))
    monkeypatch.setattr(tracking.time, "monotonic", lambda: next(monotonic_values))

    logger = tracking.FileLogger("project", "experiment")
    logger.log({"metric": 1.0}, step=0)
    logger.finish()

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == {"step": 0, "elapsed_seconds": 5.5, "data": {"metric": 1.0}}
