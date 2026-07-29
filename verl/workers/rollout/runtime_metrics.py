# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Round-scoped rollout capacity telemetry for strict MaxRL."""

from __future__ import annotations

import subprocess
import threading
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

RUNNING_REQUESTS_METRIC = "vllm:num_requests_running"
WAITING_REQUESTS_METRIC = "vllm:num_requests_waiting"


def parse_prometheus_request_counts(payload: str) -> tuple[int, int]:
    counts = {RUNNING_REQUESTS_METRIC: 0.0, WAITING_REQUESTS_METRIC: 0.0}
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        metric, separator, value_text = line.rpartition(" ")
        if not separator:
            continue
        metric_name = metric.split("{", 1)[0]
        if metric_name not in counts:
            continue
        try:
            counts[metric_name] += float(value_text)
        except ValueError:
            continue
    return round(counts[RUNNING_REQUESTS_METRIC]), round(counts[WAITING_REQUESTS_METRIC])


def parse_nvidia_smi_rows(rows: Iterable[str]) -> dict[str, dict[str, int]]:
    parsed: dict[str, dict[str, int]] = {}
    for row in rows:
        columns = [column.strip() for column in row.split(",")]
        if len(columns) != 4:
            continue
        gpu_index, utilization, memory_used_mib, memory_total_mib = columns
        try:
            parsed[gpu_index] = {
                "utilization_percent": int(utilization),
                "memory_used_bytes": int(memory_used_mib) * 1024 * 1024,
                "memory_total_bytes": int(memory_total_mib) * 1024 * 1024,
            }
        except ValueError:
            continue
    return parsed


class RolloutRuntimeMetricsSampler:
    """Poll vLLM and NVIDIA counters while one strict MaxRL round is generating."""

    def __init__(self, server_addresses: list[str], *, interval_seconds: float = 1.0) -> None:
        if not server_addresses:
            raise ValueError("server_addresses must be non-empty")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._server_addresses = list(server_addresses)
        self._interval_seconds = interval_seconds
        self._http_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._replica_peaks = {
            address: {"running_sequences": 0, "waiting_sequences": 0} for address in self._server_addresses
        }
        self._gpu_peaks: dict[str, dict[str, int]] = {}
        self._samples = 0
        self._errors = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("rollout runtime metrics sampler is already started")
        self._thread = threading.Thread(target=self._run, name="rollout-runtime-metrics", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(10.0, self._interval_seconds + 2.0))
            if self._thread.is_alive():
                with self._lock:
                    self._errors += 1
        with self._lock:
            return {
                "samples": self._samples,
                "errors": self._errors,
                "replicas": {address: dict(values) for address, values in self._replica_peaks.items()},
                "gpus": {gpu: dict(values) for gpu, values in self._gpu_peaks.items()},
            }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self._interval_seconds)

    def _sample_once(self) -> None:
        replica_samples: dict[str, tuple[int, int]] = {}
        errors = 0
        with ThreadPoolExecutor(max_workers=len(self._server_addresses)) as executor:
            for address, sample in executor.map(self._poll_server, self._server_addresses):
                if sample is None:
                    errors += 1
                else:
                    replica_samples[address] = sample

        gpu_samples: dict[str, dict[str, int]] = {}
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if result.returncode == 0:
                gpu_samples = parse_nvidia_smi_rows(result.stdout.splitlines())
            else:
                errors += 1
        except (OSError, subprocess.TimeoutExpired):
            errors += 1

        with self._lock:
            self._samples += 1
            self._errors += errors
            for address, (running, waiting) in replica_samples.items():
                peaks = self._replica_peaks[address]
                peaks["running_sequences"] = max(peaks["running_sequences"], running)
                peaks["waiting_sequences"] = max(peaks["waiting_sequences"], waiting)
            for gpu, sample in gpu_samples.items():
                peaks = self._gpu_peaks.setdefault(
                    gpu,
                    {
                        "utilization_percent": 0,
                        "memory_used_bytes": 0,
                        "memory_total_bytes": sample["memory_total_bytes"],
                    },
                )
                peaks["utilization_percent"] = max(peaks["utilization_percent"], sample["utilization_percent"])
                peaks["memory_used_bytes"] = max(peaks["memory_used_bytes"], sample["memory_used_bytes"])
                peaks["memory_total_bytes"] = sample["memory_total_bytes"]

    def _poll_server(self, address: str) -> tuple[str, tuple[int, int] | None]:
        try:
            with self._http_opener.open(f"http://{address}/metrics", timeout=0.5) as response:
                payload = response.read().decode("utf-8")
            return address, parse_prometheus_request_counts(payload)
        except (OSError, TimeoutError, UnicodeDecodeError):
            return address, None
