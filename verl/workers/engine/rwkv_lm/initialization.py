# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Process-local coordination for native RWKV model initialization."""

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator


def init_delay_seconds(rank: int, value: str | None) -> int:
    if value is None or value == "":
        return 0
    try:
        stagger = int(value)
    except ValueError as exc:
        raise RuntimeError(f"HELICOPTER_RWKV_INIT_STAGGER_SECONDS must be an integer, got {value!r}") from exc
    if stagger < 0 or stagger > 300:
        raise RuntimeError("HELICOPTER_RWKV_INIT_STAGGER_SECONDS must be between 0 and 300")
    return rank * stagger


def acquire_init_slot(value: str | None, run_log_dir: str | None) -> IO[str] | None:
    """Bound concurrent checkpoint/model materialization across colocated ranks."""
    if value is None or value == "":
        return None
    try:
        concurrency = int(value)
    except ValueError as exc:
        raise RuntimeError(f"HELICOPTER_RWKV_INIT_CONCURRENCY must be an integer, got {value!r}") from exc
    if concurrency < 1 or concurrency > 8:
        raise RuntimeError("HELICOPTER_RWKV_INIT_CONCURRENCY must be between 1 and 8")
    if not run_log_dir:
        raise RuntimeError("REMOTE_RUN_LOG_DIR is required when HELICOPTER_RWKV_INIT_CONCURRENCY is set")

    lock_dir = Path(run_log_dir) / "rwkv-init-slots"
    lock_dir.mkdir(parents=True, exist_ok=True)
    while True:
        for slot in range(concurrency):
            handle = (lock_dir / f"slot-{slot}.lock").open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            return handle
        time.sleep(0.25)


def release_init_slot(handle: IO[str] | None) -> None:
    if handle is None:
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


@contextmanager
def init_slot(value: str | None, run_log_dir: str | None) -> Iterator[None]:
    handle = acquire_init_slot(value, run_log_dir)
    try:
        yield
    finally:
        release_init_slot(handle)


@contextmanager
def coordinated_initialization(rank: int) -> Iterator[None]:
    delay = init_delay_seconds(
        rank,
        os.getenv("HELICOPTER_RWKV_INIT_STAGGER_SECONDS"),
    )
    if delay:
        time.sleep(delay)
    with init_slot(
        os.getenv("HELICOPTER_RWKV_INIT_CONCURRENCY"),
        os.getenv("REMOTE_RUN_LOG_DIR"),
    ):
        yield


__all__ = [
    "acquire_init_slot",
    "coordinated_initialization",
    "init_delay_seconds",
    "init_slot",
    "release_init_slot",
]
