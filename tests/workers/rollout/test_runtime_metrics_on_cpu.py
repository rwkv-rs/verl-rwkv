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

from types import SimpleNamespace

from verl.workers.rollout import runtime_metrics
from verl.workers.rollout.runtime_metrics import (
    RolloutRuntimeMetricsSampler,
    parse_nvidia_smi_rows,
    parse_prometheus_request_counts,
)


def test_parse_prometheus_request_counts_sums_labeled_series():
    payload = """
# HELP vllm:num_requests_running Number of requests in the model execution batches.
vllm:num_requests_running{model_name="rwkv",engine="0"} 120.0
vllm:num_requests_running{model_name="rwkv",engine="1"} 80.0
vllm:num_requests_waiting{model_name="rwkv"} 17.0
process_resident_memory_bytes 1234
"""

    assert parse_prometheus_request_counts(payload) == (200, 17)


def test_parse_nvidia_smi_rows_reports_bytes_and_skips_malformed_rows():
    assert parse_nvidia_smi_rows(
        [
            "0, 97, 16384, 97887",
            "not-a-gpu-row",
            "1, 42, 8192, 97887",
        ]
    ) == {
        "0": {
            "utilization_percent": 97,
            "memory_used_bytes": 16384 * 1024 * 1024,
            "memory_total_bytes": 97887 * 1024 * 1024,
        },
        "1": {
            "utilization_percent": 42,
            "memory_used_bytes": 8192 * 1024 * 1024,
            "memory_total_bytes": 97887 * 1024 * 1024,
        },
    }


def test_sampler_records_replica_and_gpu_peaks_without_failing_on_one_endpoint(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"vllm:num_requests_running 7\nvllm:num_requests_waiting 2\n"

    class Opener:
        def open(self, url, *, timeout):
            assert timeout == 0.5
            if "replica-1" in url:
                raise OSError("endpoint unavailable")
            return Response()

    monkeypatch.setattr(
        runtime_metrics.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="0, 91, 16384, 97887\n",
        ),
    )
    sampler = RolloutRuntimeMetricsSampler(["replica-0", "replica-1"])
    sampler._http_opener = Opener()

    sampler._sample_once()
    snapshot = sampler.stop()

    assert snapshot["samples"] == 1
    assert snapshot["errors"] == 1
    assert snapshot["replicas"]["replica-0"] == {
        "running_sequences": 7,
        "waiting_sequences": 2,
    }
    assert snapshot["replicas"]["replica-1"] == {
        "running_sequences": 0,
        "waiting_sequences": 0,
    }
    assert snapshot["gpus"]["0"]["utilization_percent"] == 91
