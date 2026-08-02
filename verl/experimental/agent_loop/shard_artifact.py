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

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from verl.experimental.agent_loop.finish_metadata import ROLLOUT_FINISH_METADATA_KEYS, build_rollout_finish_metadata
from verl.trainer.ppo.v1.policy_identity import canonical_digest

SHARD_ARTIFACT_SCHEMA_VERSION = 1


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class RolloutResponseIdentity:
    prompt_id: str
    sample_index: int
    seed: int
    sampling: Mapping[str, Any]
    model_revision: str
    policy_lineage: str

    def __post_init__(self) -> None:
        _require_nonempty_string("prompt_id", self.prompt_id)
        if isinstance(self.sample_index, bool) or not isinstance(self.sample_index, int) or self.sample_index < 0:
            raise ValueError("sample_index must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.sampling, Mapping) or not self.sampling:
            raise ValueError("sampling must be a non-empty mapping")
        _require_nonempty_string("model_revision", self.model_revision)
        _require_nonempty_string("policy_lineage", self.policy_lineage)
        _canonical_bytes(self.sampling)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "sample_index": self.sample_index,
            "seed": self.seed,
            "sampling": dict(self.sampling),
            "sampling_digest": canonical_digest(self.sampling),
            "model_revision": self.model_revision,
            "policy_lineage": self.policy_lineage,
        }

    @property
    def digest(self) -> str:
        return _sha256(_canonical_bytes(self.as_dict()))


def _atomic_write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class RolloutShardArtifact:
    """Resumable deterministic response records promoted as one immutable shard."""

    def __init__(self, root: str | Path, shard_id: str, expected: Sequence[RolloutResponseIdentity]) -> None:
        self.root = Path(root)
        self.shard_id = _require_nonempty_string("shard_id", shard_id)
        if not expected:
            raise ValueError("expected identities must not be empty")
        self.expected = {identity.digest: identity for identity in expected}
        if len(self.expected) != len(expected):
            raise ValueError("expected identities contain duplicates")
        self.partial_path = self.root / f".{self.shard_id}.partial"
        self.promoted_path = self.root / self.shard_id
        self.contract = {
            "schema_version": SHARD_ARTIFACT_SCHEMA_VERSION,
            "shard_id": self.shard_id,
            "expected_identity_digests": sorted(self.expected),
        }
        self._open_or_create()

    @property
    def is_promoted(self) -> bool:
        return self.promoted_path.is_dir()

    def _open_or_create(self) -> None:
        if self.is_promoted:
            validate_promoted_shard(self.promoted_path, expected_contract=self.contract)
            return
        self.partial_path.mkdir(parents=True, exist_ok=True)
        contract_path = self.partial_path / "contract.json"
        payload = _canonical_bytes(self.contract)
        try:
            _atomic_write_new(contract_path, payload)
        except FileExistsError:
            if contract_path.read_bytes() != payload:
                raise RuntimeError("partial shard contract conflicts with the requested identities") from None

    def write_response(
        self,
        identity: RolloutResponseIdentity,
        response: str,
        *,
        finish_reason: str | None,
        backend_stop_reason: Any,
        repetition_truncated: bool,
    ) -> bool:
        if self.is_promoted:
            raise RuntimeError("promoted shards are read-only")
        if identity.digest not in self.expected:
            raise ValueError("response identity is not part of this shard contract")
        if not isinstance(response, str):
            raise TypeError("response must be a string")
        record = {
            "schema_version": SHARD_ARTIFACT_SCHEMA_VERSION,
            "identity": identity.as_dict(),
            "identity_digest": identity.digest,
            "response": response,
            "finish_source": {
                "finish_reason": finish_reason,
                "backend_stop_reason": backend_stop_reason,
                "repetition_truncated": repetition_truncated,
            },
            "finish": build_rollout_finish_metadata(
                response,
                finish_reason=finish_reason,
                backend_stop_reason=backend_stop_reason,
                repetition_truncated=repetition_truncated,
            ),
        }
        payload = _canonical_bytes(record)
        record_path = self.partial_path / "records" / f"{identity.digest}.json"
        try:
            _atomic_write_new(record_path, payload)
        except FileExistsError:
            if record_path.read_bytes() == payload:
                return False
            raise RuntimeError(f"conflicting response for identity {identity.digest}") from None
        return True

    def promote(self) -> Path:
        if self.is_promoted:
            validate_promoted_shard(self.promoted_path, expected_contract=self.contract)
            return self.promoted_path
        manifest = _validate_partial_shard(self.partial_path, self.contract)
        _atomic_replace(self.partial_path / "manifest.json", _canonical_bytes(manifest))
        for path in (self.partial_path / "records").glob("*.json"):
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        (self.partial_path / "contract.json").chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        (self.partial_path / "manifest.json").chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        readonly_directory_mode = (
            stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
        )
        (self.partial_path / "records").chmod(readonly_directory_mode)
        self.partial_path.chmod(readonly_directory_mode)
        os.replace(self.partial_path, self.promoted_path)
        _fsync_directory(self.root)
        validate_promoted_shard(self.promoted_path, expected_contract=self.contract)
        return self.promoted_path


def _load_record(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    try:
        record = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid response record {path.name}") from error
    if _canonical_bytes(record) != payload:
        raise RuntimeError(f"response record {path.name} is not canonical")
    required = {"schema_version", "identity", "identity_digest", "response", "finish_source", "finish"}
    if set(record) != required:
        raise RuntimeError(f"response record {path.name} has an invalid schema")
    if record["schema_version"] != SHARD_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError(f"response record {path.name} has an unsupported schema version")
    if _sha256(_canonical_bytes(record["identity"])) != record["identity_digest"]:
        raise RuntimeError(f"response record {path.name} identity digest mismatch")
    if set(record["finish"]) != set(ROLLOUT_FINISH_METADATA_KEYS):
        raise RuntimeError(f"response record {path.name} has incomplete finish metadata")
    finish_source = record["finish_source"]
    if set(finish_source) != {"finish_reason", "backend_stop_reason", "repetition_truncated"}:
        raise RuntimeError(f"response record {path.name} has an invalid finish source")
    expected_finish = build_rollout_finish_metadata(record["response"], **finish_source)
    if record["finish"] != expected_finish:
        raise RuntimeError(f"response record {path.name} finish metadata mismatch")
    categories = [record["finish"][key] for key in ROLLOUT_FINISH_METADATA_KEYS[2:]]
    if any(not isinstance(value, bool) for value in categories) or sum(categories) != 1:
        raise RuntimeError(f"response record {path.name} has invalid finish categories")
    return record, payload


def _validate_partial_shard(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_dir():
        raise RuntimeError("partial shard directory is missing")
    if json.loads((path / "contract.json").read_bytes()) != contract:
        raise RuntimeError("partial shard contract mismatch")
    records_dir = path / "records"
    record_paths = sorted(records_dir.glob("*.json")) if records_dir.is_dir() else []
    expected_digests = contract["expected_identity_digests"]
    actual_digests = [record_path.stem for record_path in record_paths]
    if actual_digests != expected_digests:
        missing = sorted(set(expected_digests) - set(actual_digests))
        unexpected = sorted(set(actual_digests) - set(expected_digests))
        raise RuntimeError(f"shard is incomplete: missing={missing}, unexpected={unexpected}")
    record_sha256 = {}
    aggregate = hashlib.sha256()
    for record_path in record_paths:
        record, payload = _load_record(record_path)
        if record["identity_digest"] != record_path.stem:
            raise RuntimeError(f"response record {record_path.name} has the wrong filename")
        record_sha256[record_path.stem] = _sha256(payload)
        aggregate.update(payload)
    return {
        **contract,
        "record_count": len(record_paths),
        "record_sha256": record_sha256,
        "shard_sha256": aggregate.hexdigest(),
    }


def validate_promoted_shard(path: str | Path, *, expected_contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not path.is_dir() or not manifest_path.is_file():
        raise RuntimeError("promoted shard is missing its manifest")
    manifest = json.loads(manifest_path.read_bytes())
    contract = {key: manifest[key] for key in ("schema_version", "shard_id", "expected_identity_digests")}
    if expected_contract is not None and contract != expected_contract:
        raise RuntimeError("promoted shard contract mismatch")
    actual_manifest = _validate_partial_shard(path, contract)
    if actual_manifest != manifest:
        raise RuntimeError("promoted shard digest mismatch")
    readonly_paths = [path, path / "records", path / "contract.json", manifest_path, *(path / "records").glob("*.json")]
    if any(item.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) for item in readonly_paths):
        raise RuntimeError("promoted shard must be read-only")
    return manifest


__all__ = [
    "RolloutResponseIdentity",
    "RolloutShardArtifact",
    "SHARD_ARTIFACT_SCHEMA_VERSION",
    "validate_promoted_shard",
]
