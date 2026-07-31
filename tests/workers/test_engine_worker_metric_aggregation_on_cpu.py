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

from verl.utils.metric.utils import Metric
from verl.workers.engine_workers import _normalize_gathered_metric_values


def test_normalize_gathered_metric_values_keeps_scalar_lists():
    assert _normalize_gathered_metric_values([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_normalize_gathered_metric_values_flattens_nested_lists():
    assert _normalize_gathered_metric_values([[1.0], [2.0, 3.0]]) == [1.0, 2.0, 3.0]


def test_normalize_gathered_metric_values_aggregates_metric_lists():
    assert (
        _normalize_gathered_metric_values(
            [
                Metric(aggregation="mean", value=1.0),
                Metric(aggregation="mean", value=3.0),
            ]
        )
        == 2.0
    )
