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

import hashlib
import json

CAMPAIGN_PROBLEM_ID_FIELD = "__campaign_problem_id__"
CAMPAIGN_ROLLOUT_INDEX_FIELD = "__campaign_rollout_index__"
CAMPAIGN_SAMPLING_SEED_FIELD = "__sampling_seed__"


def deterministic_campaign_seed(
    campaign_id: str,
    problem_id: str,
    rollout_index: int,
    base_seed: int,
) -> int:
    """Derive one stable signed-63-bit seed for an iid campaign response."""

    payload = json.dumps(
        {
            "base_seed": base_seed,
            "campaign_id": campaign_id,
            "problem_id": problem_id,
            "rollout_index": rollout_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


__all__ = [
    "CAMPAIGN_PROBLEM_ID_FIELD",
    "CAMPAIGN_ROLLOUT_INDEX_FIELD",
    "CAMPAIGN_SAMPLING_SEED_FIELD",
    "deterministic_campaign_seed",
]
