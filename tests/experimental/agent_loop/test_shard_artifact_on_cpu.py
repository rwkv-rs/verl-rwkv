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
import os
import stat

import pytest

from verl.experimental.agent_loop.shard_artifact import (
    RolloutCampaignArtifact,
    RolloutResponseIdentity,
    RolloutShardArtifact,
    validate_promoted_shard,
)


def _campaign(root, *, rollouts_per_problem=4):
    return RolloutCampaignArtifact(
        root,
        "dapo-campaign",
        ["problem-c", "problem-a", "problem-b"],
        rollouts_per_problem=rollouts_per_problem,
        dataset_fingerprint="sha256:dapo-17k",
        source_revision="verl-rwkv@abc123",
        parameters={"temperature": 1.0, "rollouts_per_problem": rollouts_per_problem},
    )


def _write_campaign_completion(store, problem_id, rollout_index, *, response=None):
    category = rollout_index % 4
    values = {
        0: {
            "finish_reason": "stop",
            "backend_stop_reason": 0,
            "repetition_truncated": False,
            "response_token_ids": [101, 0],
            "eos_token_ids": [0],
        },
        1: {
            "finish_reason": "length",
            "backend_stop_reason": None,
            "repetition_truncated": False,
            "response_token_ids": [101, 102],
        },
        2: {
            "finish_reason": "stop",
            "backend_stop_reason": "repetition_detected",
            "repetition_truncated": True,
            "response_token_ids": [101, 102, 103, 104] * 3,
        },
        3: {
            "finish_reason": "stop",
            "backend_stop_reason": "stop_token",
            "repetition_truncated": False,
            "response_token_ids": [101, 261],
            "stop_token_ids": [261],
        },
    }[category]
    return store.write_completion(
        problem_id,
        rollout_index,
        response
        if response is not None
        else (
            "invalid"
            if category == 3
            else "<think>work</think><answer>42</answer>"
        ),
        **values,
    )


def _identity(sample_index: int = 0, **overrides) -> RolloutResponseIdentity:
    values = {
        "prompt_id": "dapo-00001",
        "sample_index": sample_index,
        "seed": 20260801 + sample_index,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 4096},
        "model_revision": "rwkv7-g1i@sha256:weights",
        "policy_lineage": "policy-0:lineage-digest",
        "policy_version": 0,
        "source_lineage": "checkpoint:sha256:source",
        "runtime_identity": "vllm-rwkv:sha256:runtime",
    }
    values.update(overrides)
    return RolloutResponseIdentity(**values)


def _write(
    store: RolloutShardArtifact,
    identity: RolloutResponseIdentity,
    response: str = "<think>x</think><answer>1</answer>",
):
    return store.write_response(
        identity,
        response,
        finish_reason="stop",
        backend_stop_reason=0,
        repetition_truncated=False,
        response_token_ids=[101, 102, 0],
        eos_token_ids=[0],
    )


def _make_tree_writable(path):
    if not path.exists():
        return
    for root, dirs, files in os.walk(path):
        for name in dirs:
            (path.__class__(root) / name).chmod(stat.S_IRWXU)
        for name in files:
            (path.__class__(root) / name).chmod(stat.S_IRUSR | stat.S_IWUSR)
    path.chmod(stat.S_IRWXU)


def test_campaign_tiny_3x4_resumes_out_of_order_and_finalizes_deterministically(
    tmp_path,
):
    pairs = [
        (problem_id, rollout_index)
        for problem_id in ("problem-c", "problem-a", "problem-b")
        for rollout_index in range(4)
    ]
    first = _campaign(tmp_path)
    for problem_id, rollout_index in reversed(pairs[:5]):
        assert _write_campaign_completion(first, problem_id, rollout_index)

    resumed = _campaign(tmp_path)
    assert not _write_campaign_completion(resumed, *pairs[0])
    for problem_id, rollout_index in reversed(pairs[5:]):
        assert _write_campaign_completion(resumed, problem_id, rollout_index)

    result = resumed.finalize()
    payload = resumed.final_path.read_bytes()
    assert result["contract"]["dataset_fingerprint"] == "sha256:dapo-17k"
    assert result["contract"]["source_revision"] == "verl-rwkv@abc123"
    assert result["contract"]["parameters"]["rollouts_per_problem"] == 4
    assert result["counts"] == {
        "total": 12,
        "strict_cot_format": 9,
        "ended_by_eos": 3,
        "length_truncated": 3,
        "repetition_truncated": 3,
    }
    assert result["rates"] == {
        "strict_cot_format": 0.75,
        "ended_by_eos": 0.25,
        "length_truncated": 0.25,
        "repetition_truncated": 0.25,
    }
    assert resumed.finalize() == result
    assert resumed.final_path.read_bytes() == payload
    assert _campaign(tmp_path).finalize() == result


def test_campaign_rejects_duplicate_conflict_and_invalid_pair(tmp_path):
    store = _campaign(tmp_path)
    assert _write_campaign_completion(store, "problem-a", 0)
    assert not _write_campaign_completion(store, "problem-a", 0)
    with pytest.raises(RuntimeError, match="conflicting completion for pair"):
        _write_campaign_completion(
            store,
            "problem-a",
            0,
            response="<think>different</think><answer>0</answer>",
        )
    with pytest.raises(ValueError, match="outside the campaign range"):
        _write_campaign_completion(store, "problem-a", 4)
    with pytest.raises(ValueError, match="not part of this campaign"):
        _write_campaign_completion(store, "unknown", 0)


def test_campaign_finalize_rejects_any_missing_pair(tmp_path):
    store = _campaign(tmp_path)
    for problem_id in ("problem-c", "problem-a", "problem-b"):
        for rollout_index in range(4):
            if (problem_id, rollout_index) != ("problem-b", 3):
                _write_campaign_completion(store, problem_id, rollout_index)

    with pytest.raises(RuntimeError, match="campaign is incomplete.*problem-b"):
        store.finalize()


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("prompt_id", "", "prompt_id must be a non-empty string"),
        ("sample_index", -1, "sample_index must be a non-negative integer"),
        ("seed", True, "seed must be an integer"),
        ("sampling", {}, "sampling must be a non-empty mapping"),
        ("model_revision", "", "model_revision must be a non-empty string"),
        ("policy_lineage", "", "policy_lineage must be a non-empty string"),
        ("policy_version", -1, "policy_version must be a non-negative integer"),
        ("source_lineage", "", "source_lineage must be a non-empty string"),
        ("runtime_identity", "", "runtime_identity must be a non-empty string"),
    ],
)
def test_response_identity_rejects_missing_or_invalid_contract_fields(field, value, error):
    with pytest.raises(ValueError, match=error):
        _identity(**{field: value})


def test_response_identity_binds_sampling_model_and_lineage():
    baseline = _identity()

    assert baseline.digest != _identity(sampling={"temperature": 0.5}).digest
    assert baseline.digest != _identity(model_revision="different-model").digest
    assert baseline.digest != _identity(policy_lineage="policy-1:new-lineage").digest
    assert baseline.digest != _identity(policy_version=1).digest
    assert baseline.digest != _identity(source_lineage="checkpoint:sha256:other").digest
    assert baseline.digest != _identity(runtime_identity="vllm-rwkv:sha256:other").digest


def test_finish_metadata_is_recomputed_from_saved_token_ids(tmp_path):
    identity = _identity()
    store = RolloutShardArtifact(tmp_path, "shard-000", [identity])
    token_ids = [1100, 1101, 1102, 1103] * 3

    assert store.write_response(
        identity,
        "<think>x</think><answer>1</answer>",
        finish_reason="stop",
        backend_stop_reason="tail-repeat",
        repetition_truncated=True,
        response_token_ids=token_ids,
        eos_token_ids=[0],
        stop_token_ids=[261],
    )
    promoted = store.promote()
    try:
        record = json.loads((promoted / "records" / f"{identity.digest}.json").read_bytes())
        assert record["finish"]["repetition_truncated"] is True
        assert record["finish_source"]["response_token_ids"] == token_ids
    finally:
        _make_tree_writable(promoted)


def test_response_token_ids_reject_inconsistent_repetition_flag(tmp_path):
    identity = _identity()
    store = RolloutShardArtifact(tmp_path, "shard-000", [identity])

    with pytest.raises(ValueError, match="repetition_truncated conflicts"):
        store.write_response(
            identity,
            "<think>x</think><answer>1</answer>",
            finish_reason="stop",
            backend_stop_reason=0,
            repetition_truncated=True,
            response_token_ids=[101, 102, 103],
            eos_token_ids=[0],
        )


def test_partial_shard_resumes_without_rewriting_completed_records(tmp_path):
    identities = [_identity(0), _identity(1)]
    first = RolloutShardArtifact(tmp_path, "shard-000", identities)
    assert _write(first, identities[0]) is True

    resumed = RolloutShardArtifact(tmp_path, "shard-000", identities)
    assert _write(resumed, identities[0]) is False
    assert _write(resumed, identities[1]) is True
    promoted = resumed.promote()
    try:
        manifest = validate_promoted_shard(promoted)
        assert manifest["record_count"] == 2
    finally:
        _make_tree_writable(promoted)


def test_duplicate_identity_is_idempotent_only_for_identical_record(tmp_path):
    identity = _identity()
    store = RolloutShardArtifact(tmp_path, "shard-000", [identity])

    assert _write(store, identity) is True
    assert _write(store, identity) is False
    with pytest.raises(RuntimeError, match="conflicting response for identity"):
        _write(store, identity, "<think>different</think><answer>2</answer>")


def test_resume_rejects_changed_shard_contract(tmp_path):
    RolloutShardArtifact(tmp_path, "shard-000", [_identity(0)])

    with pytest.raises(RuntimeError, match="partial shard contract conflicts"):
        RolloutShardArtifact(tmp_path, "shard-000", [_identity(1)])


def test_promotion_rejects_missing_and_unexpected_records(tmp_path):
    identities = [_identity(0), _identity(1)]
    store = RolloutShardArtifact(tmp_path, "shard-000", identities)
    _write(store, identities[0])

    with pytest.raises(RuntimeError, match="shard is incomplete"):
        store.promote()

    records = store.partial_path / "records"
    (records / f"{'f' * 64}.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected"):
        store.promote()


def test_promotion_is_read_only_and_idempotent(tmp_path):
    identity = _identity()
    store = RolloutShardArtifact(tmp_path, "shard-000", [identity])
    _write(store, identity)
    promoted = store.promote()
    try:
        assert not store.partial_path.exists()
        assert store.promote() == promoted
        manifest = validate_promoted_shard(promoted)
        assert manifest["expected_identity_digests"] == [identity.digest]
        assert manifest["record_sha256"][identity.digest]
        assert manifest["shard_sha256"]
        record = json.loads((promoted / "records" / f"{identity.digest}.json").read_bytes())
        assert record["finish"]["format_valid"] is True
        assert record["finish"]["format_parser_version"] == "strict-cot-v1"
        assert record["finish"]["ended_by_eos"] is True
        for path in [promoted, promoted / "records", promoted / "contract.json", promoted / "manifest.json"]:
            assert path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0
        with pytest.raises(RuntimeError, match="read-only"):
            _write(store, identity)
    finally:
        _make_tree_writable(promoted)


def test_promotion_recomputes_finish_metadata(tmp_path):
    identity = _identity()
    store = RolloutShardArtifact(tmp_path, "shard-000", [identity])
    _write(store, identity)
    record_path = store.partial_path / "records" / f"{identity.digest}.json"
    record = json.loads(record_path.read_bytes())
    record["finish"]["format_valid"] = False
    record_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    with pytest.raises(RuntimeError, match="finish metadata mismatch"):
        store.promote()


def test_promoted_shard_rejects_record_digest_tampering(tmp_path):
    identity = _identity()
    store = RolloutShardArtifact(tmp_path, "shard-000", [identity])
    _write(store, identity)
    promoted = store.promote()
    record = promoted / "records" / f"{identity.digest}.json"
    try:
        record.chmod(stat.S_IRUSR | stat.S_IWUSR)
        record.write_bytes(record.read_bytes() + b" ")
        record.chmod(stat.S_IRUSR)
        with pytest.raises(RuntimeError, match="not canonical"):
            validate_promoted_shard(promoted)
    finally:
        _make_tree_writable(promoted)


def test_promoted_shard_rejects_manifest_digest_tampering(tmp_path):
    identity = _identity()
    store = RolloutShardArtifact(tmp_path, "shard-000", [identity])
    _write(store, identity)
    promoted = store.promote()
    manifest = promoted / "manifest.json"
    try:
        manifest.chmod(stat.S_IRUSR | stat.S_IWUSR)
        payload = manifest.read_text(encoding="utf-8").replace('"shard_sha256":"', '"shard_sha256":"bad')
        manifest.write_text(payload, encoding="utf-8")
        manifest.chmod(stat.S_IRUSR)
        with pytest.raises(RuntimeError, match="promoted shard digest mismatch"):
            validate_promoted_shard(promoted)
    finally:
        _make_tree_writable(promoted)
