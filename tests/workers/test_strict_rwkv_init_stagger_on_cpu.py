from pathlib import Path

import pytest

from verl.workers.engine_workers import (
    acquire_strict_rwkv_init_slot,
    release_strict_rwkv_init_slot,
    strict_rwkv_init_delay_seconds,
    strict_rwkv_init_slot,
)


def test_strict_rwkv_init_delay_staggers_ranks_without_changing_rank_zero():
    assert strict_rwkv_init_delay_seconds(0, "30") == 0
    assert strict_rwkv_init_delay_seconds(7, "30") == 210
    assert strict_rwkv_init_delay_seconds(3, None) == 0


@pytest.mark.parametrize("value", ["-1", "301", "invalid"])
def test_strict_rwkv_init_delay_rejects_unsafe_values(value):
    with pytest.raises(RuntimeError):
        strict_rwkv_init_delay_seconds(1, value)


def test_strict_rwkv_init_slot_is_released_after_failure(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        with strict_rwkv_init_slot("1", str(tmp_path)):
            raise RuntimeError("boom")

    handle = acquire_strict_rwkv_init_slot("1", str(tmp_path))
    assert handle is not None
    assert Path(handle.name).parent == tmp_path / "rwkv-init-slots"
    release_strict_rwkv_init_slot(handle)


@pytest.mark.parametrize("value", ["0", "9", "invalid"])
def test_strict_rwkv_init_slot_rejects_unsafe_concurrency(value, tmp_path):
    with pytest.raises(RuntimeError):
        acquire_strict_rwkv_init_slot(value, str(tmp_path))


def test_strict_rwkv_init_slot_requires_shared_run_directory():
    with pytest.raises(RuntimeError, match="REMOTE_RUN_LOG_DIR"):
        acquire_strict_rwkv_init_slot("1", None)
