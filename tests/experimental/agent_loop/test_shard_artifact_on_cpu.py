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
    RolloutResponseIdentity,
    RolloutShardArtifact,
    validate_promoted_shard,
)


def _identity(sample_index: int = 0, **overrides) -> RolloutResponseIdentity:
    values = {
        "prompt_id": "dapo-00001",
        "sample_index": sample_index,
        "seed": 20260801 + sample_index,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 4096},
        "model_revision": "rwkv7-g1i@sha256:weights",
        "policy_lineage": "policy-0:lineage-digest",
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


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("prompt_id", "", "prompt_id must be a non-empty string"),
        ("sample_index", -1, "sample_index must be a non-negative integer"),
        ("seed", True, "seed must be an integer"),
        ("sampling", {}, "sampling must be a non-empty mapping"),
        ("model_revision", "", "model_revision must be a non-empty string"),
        ("policy_lineage", "", "policy_lineage must be a non-empty string"),
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
