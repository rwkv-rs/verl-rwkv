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
