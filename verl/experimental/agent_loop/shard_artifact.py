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

import fcntl
import hashlib
import json
import math
import os
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from verl.experimental.agent_loop.campaign_fields import deterministic_campaign_seed
from verl.experimental.agent_loop.finish_metadata import ROLLOUT_FINISH_METADATA_KEYS, build_rollout_finish_metadata
from verl.trainer.ppo.v1.policy_identity import canonical_digest
from verl.utils.ngram_repetition import ConsecutiveRepetitionDetector

SHARD_ARTIFACT_SCHEMA_VERSION = 2
ROLLOUT_CAMPAIGN_SCHEMA_VERSION = 2
ROLLOUT_CAMPAIGN_PARAMETER_KEYS = (
    "base_seed",
    "sampling",
    "model_revision",
    "policy_lineage",
    "policy_version",
    "source_lineage",
    "runtime_identity",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_path_component(name: str, value: Any) -> str:
    value = _require_nonempty_string(name, value)
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{name} must be a single path component")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_campaign_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(parameters, Mapping):
        raise ValueError("campaign parameters must be a mapping")
    missing = [key for key in ROLLOUT_CAMPAIGN_PARAMETER_KEYS if key not in parameters]
    if missing:
        raise ValueError(f"campaign parameters are missing identity fields: {', '.join(missing)}")
    if isinstance(parameters["base_seed"], bool) or not isinstance(parameters["base_seed"], int):
        raise ValueError("campaign base_seed must be an integer")
    if not isinstance(parameters["sampling"], Mapping) or not parameters["sampling"]:
        raise ValueError("campaign sampling must be a non-empty mapping")
    if (
        isinstance(parameters["policy_version"], bool)
        or not isinstance(parameters["policy_version"], int)
        or parameters["policy_version"] < 0
    ):
        raise ValueError("campaign policy_version must be a non-negative integer")
    for key in ("model_revision", "policy_lineage", "source_lineage", "runtime_identity"):
        _require_nonempty_string(f"campaign {key}", parameters[key])
    try:
        canonical = json.loads(_canonical_bytes(parameters))
    except (TypeError, ValueError) as error:
        raise ValueError("campaign parameters must contain only strict JSON values") from error
    return canonical


@dataclass(frozen=True)
class RolloutResponseIdentity:
    prompt_id: str
    sample_index: int
    seed: int
    sampling: Mapping[str, Any]
    model_revision: str
    policy_lineage: str
    policy_version: int
    source_lineage: str
    runtime_identity: str

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
        if isinstance(self.policy_version, bool) or not isinstance(self.policy_version, int) or self.policy_version < 0:
            raise ValueError("policy_version must be a non-negative integer")
        _require_nonempty_string("source_lineage", self.source_lineage)
        _require_nonempty_string("runtime_identity", self.runtime_identity)
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
            "policy_version": self.policy_version,
            "source_lineage": self.source_lineage,
            "runtime_identity": self.runtime_identity,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RolloutResponseIdentity:
        required = {
            "prompt_id",
            "sample_index",
            "seed",
            "sampling",
            "sampling_digest",
            "model_revision",
            "policy_lineage",
            "policy_version",
            "source_lineage",
            "runtime_identity",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("response identity has an invalid schema")
        identity = cls(**{key: value[key] for key in required if key != "sampling_digest"})
        if value["sampling_digest"] != canonical_digest(identity.sampling):
            raise ValueError("response identity sampling digest mismatch")
        return identity

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
        response_token_ids: Sequence[int],
        eos_token_ids: Sequence[int] = (),
        stop_token_ids: Sequence[int] = (),
    ) -> bool:
        if self.is_promoted:
            raise RuntimeError("promoted shards are read-only")
        if identity.digest not in self.expected:
            raise ValueError("response identity is not part of this shard contract")
        if not isinstance(response, str):
            raise TypeError("response must be a string")
        token_ids = list(response_token_ids)
        if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in token_ids):
            raise TypeError("response_token_ids must contain integers")
        computed_repetition_length = ConsecutiveRepetitionDetector().observe(token_ids)
        computed_repetition_truncated = computed_repetition_length is not None
        if repetition_truncated != computed_repetition_truncated:
            raise ValueError("repetition_truncated conflicts with response token IDs")
        finish_source = {
            "finish_reason": finish_reason,
            "backend_stop_reason": backend_stop_reason,
            "repetition_truncated": computed_repetition_truncated,
            "response_token_ids": token_ids,
            "eos_token_ids": list(eos_token_ids),
            "stop_token_ids": list(stop_token_ids),
        }
        record = {
            "schema_version": SHARD_ARTIFACT_SCHEMA_VERSION,
            "identity": identity.as_dict(),
            "identity_digest": identity.digest,
            "response": response,
            "finish_source": finish_source,
            "finish": build_rollout_finish_metadata(response, **finish_source),
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


class RolloutCampaignArtifact:
    """Atomic completion ledger for a fixed problem-by-rollout campaign."""

    def __init__(
        self,
        root: str | Path,
        campaign_id: str,
        problem_ids: Sequence[str],
        *,
        rollouts_per_problem: int,
        dataset_fingerprint: str,
        source_revision: str,
        parameters: Mapping[str, Any],
    ) -> None:
        self.root = Path(root)
        self.campaign_id = _require_path_component("campaign_id", campaign_id)
        problems = tuple(_require_nonempty_string("problem_id", value) for value in problem_ids)
        if not problems or len(set(problems)) != len(problems):
            raise ValueError("problem_ids must be a non-empty unique sequence")
        if (
            isinstance(rollouts_per_problem, bool)
            or not isinstance(rollouts_per_problem, int)
            or rollouts_per_problem <= 0
        ):
            raise ValueError("rollouts_per_problem must be a positive integer")
        canonical_parameters = _validate_campaign_parameters(parameters)
        self.problem_ids = problems
        self.problem_id_set = frozenset(problems)
        self.rollouts_per_problem = rollouts_per_problem
        self.expected_pair_count = len(problems) * rollouts_per_problem
        self.partial_path = self.root / f".{self.campaign_id}.campaign"
        self.promoted_path = self.root / self.campaign_id
        self.final_path = self.promoted_path / "manifest.json"
        self.lock_path = self.root / f".{self.campaign_id}.lock"
        self.contract = {
            "schema_version": ROLLOUT_CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "problem_ids": list(problems),
            "rollouts_per_problem": rollouts_per_problem,
            "dataset_fingerprint": _require_nonempty_string("dataset_fingerprint", dataset_fingerprint),
            "source_revision": _require_nonempty_string("source_revision", source_revision),
            "parameters": canonical_parameters,
            "expected_pair_count": self.expected_pair_count,
        }
        self._open_or_create()

    @property
    def is_promoted(self) -> bool:
        return self.promoted_path.is_dir()

    @staticmethod
    def _pair_digest(problem_id: str, rollout_index: int) -> str:
        return _sha256(_canonical_bytes({"problem_id": problem_id, "rollout_index": rollout_index}))

    @contextmanager
    def _lock(self, *, exclusive: bool) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as stream:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(stream.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _open_or_create(self) -> None:
        with self._lock(exclusive=True):
            payload = _canonical_bytes(self.contract)
            if self.is_promoted:
                _validate_promoted_campaign(self.promoted_path, self.contract)
                return
            if self.promoted_path.exists():
                raise RuntimeError("promoted campaign path exists but is not a directory")
            self.partial_path.mkdir(parents=True, exist_ok=True)
            try:
                _atomic_write_new(self.partial_path / "contract.json", payload)
            except FileExistsError:
                if (self.partial_path / "contract.json").read_bytes() != payload:
                    raise RuntimeError("partial campaign contract conflicts with the request") from None

    def completed_pairs(self) -> frozenset[tuple[str, int]]:
        """Return a validated snapshot of completed pairs for small-campaign inspection."""

        with self._lock(exclusive=False):
            if self.is_promoted:
                _validate_promoted_campaign(self.promoted_path, self.contract)
                return frozenset(
                    (problem_id, rollout_index)
                    for problem_id in self.problem_ids
                    for rollout_index in range(self.rollouts_per_problem)
                )
            paths = _campaign_record_paths(self.partial_path)
            completed: set[tuple[str, int]] = set()
            for path in paths:
                record, _ = _load_campaign_record(path)
                identity = record["identity"]
                pair = (identity["prompt_id"], identity["sample_index"])
                digest = self._pair_digest(*pair)
                if digest != path.stem or not self.contains_pair(*pair):
                    raise RuntimeError(f"campaign record {path.name} has an invalid pair")
                completed.add(pair)
            return frozenset(completed)

    def pending_pairs(self) -> tuple[tuple[str, int], ...]:
        """Return missing pairs in stable dataset then rollout-index order."""

        completed = self.completed_pairs()
        return tuple(
            (problem_id, rollout_index)
            for problem_id in self.problem_ids
            for rollout_index in range(self.rollouts_per_problem)
            if (problem_id, rollout_index) not in completed
        )

    def contains_pair(self, problem_id: str, rollout_index: int) -> bool:
        return problem_id in self.problem_id_set and 0 <= rollout_index < self.rollouts_per_problem

    def has_completion(
        self,
        problem_id: str,
        rollout_index: int,
        *,
        expected_identity: RolloutResponseIdentity | None = None,
    ) -> bool:
        if not self.contains_pair(problem_id, rollout_index):
            raise ValueError("problem-rollout pair is outside the campaign contract")
        if self.is_promoted:
            return True
        path = _campaign_record_path(
            self.partial_path,
            self._pair_digest(problem_id, rollout_index),
        )
        if not path.is_file():
            return False
        record, _ = _load_campaign_record(path)
        identity = RolloutResponseIdentity.from_dict(record["identity"])
        if (identity.prompt_id, identity.sample_index) != (problem_id, rollout_index):
            raise RuntimeError(f"campaign record {path.name} has an invalid pair")
        if expected_identity is not None and identity.digest != expected_identity.digest:
            raise RuntimeError(f"campaign record {path.name} conflicts with the resumed policy identity")
        return True

    def remaining_count(self) -> int:
        with self._lock(exclusive=False):
            if self.is_promoted:
                _validate_promoted_campaign(self.promoted_path, self.contract)
                return 0
            completed = sum(1 for _ in _campaign_record_paths(self.partial_path))
            if completed > self.expected_pair_count:
                raise RuntimeError(
                    "campaign contains more records than its contract: "
                    f"expected={self.expected_pair_count}, actual={completed}"
                )
            return self.expected_pair_count - completed

    def write_completion(
        self,
        identity: RolloutResponseIdentity,
        response: str,
        *,
        reward: float,
        finish_reason: str | None,
        backend_stop_reason: Any,
        repetition_truncated: bool,
        response_token_ids: Sequence[int],
        eos_token_ids: Sequence[int] = (),
        stop_token_ids: Sequence[int] = (),
    ) -> bool:
        problem_id = identity.prompt_id
        rollout_index = identity.sample_index
        if problem_id not in self.problem_ids:
            raise ValueError("problem_id is not part of this campaign")
        if (
            isinstance(rollout_index, bool)
            or not isinstance(rollout_index, int)
            or not 0 <= rollout_index < self.rollouts_per_problem
        ):
            raise ValueError("rollout_index is outside the campaign range")
        _validate_campaign_identity(identity, self.contract)
        if not isinstance(response, str):
            raise TypeError("response must be a string")
        if isinstance(reward, bool) or not isinstance(reward, int | float) or not math.isfinite(reward):
            raise TypeError("reward must be a finite number")
        token_ids = list(response_token_ids)
        if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in token_ids):
            raise TypeError("response_token_ids must contain only integers")
        computed_repetition = ConsecutiveRepetitionDetector().observe(token_ids) is not None
        if repetition_truncated != computed_repetition:
            raise ValueError("repetition_truncated conflicts with response token IDs")
        finish_source = {
            "finish_reason": finish_reason,
            "backend_stop_reason": backend_stop_reason,
            "repetition_truncated": computed_repetition,
            "response_token_ids": token_ids,
            "eos_token_ids": list(eos_token_ids),
            "stop_token_ids": list(stop_token_ids),
        }
        record = {
            "schema_version": ROLLOUT_CAMPAIGN_SCHEMA_VERSION,
            "pair_digest": self._pair_digest(problem_id, rollout_index),
            "identity": identity.as_dict(),
            "identity_digest": identity.digest,
            "response": response,
            "reward": float(reward),
            "finish_source": finish_source,
            "finish": build_rollout_finish_metadata(response, **finish_source),
        }
        payload = _canonical_bytes(record)
        with self._lock(exclusive=False):
            campaign_path = self.promoted_path if self.is_promoted else self.partial_path
            record_path = _campaign_record_path(campaign_path, record["pair_digest"])
            if self.is_promoted:
                if record_path.read_bytes() == payload:
                    return False
                raise RuntimeError(f"conflicting completion for pair ({problem_id!r}, {rollout_index})")
            try:
                _atomic_write_new(record_path, payload)
            except FileExistsError:
                if record_path.read_bytes() == payload:
                    return False
                raise RuntimeError(f"conflicting completion for pair ({problem_id!r}, {rollout_index})") from None
            return True

    def finalize(self) -> dict[str, Any]:
        with self._lock(exclusive=True):
            if self.is_promoted:
                return _validate_promoted_campaign(self.promoted_path, self.contract)
            result = _build_campaign_manifest(self.partial_path, self.contract)
            manifest_path = self.partial_path / "manifest.json"
            manifest_payload = _canonical_bytes(result)
            if manifest_path.is_file():
                if manifest_path.read_bytes() != manifest_payload:
                    raise RuntimeError("partial campaign manifest conflicts with its records")
            else:
                _atomic_replace(manifest_path, manifest_payload)
            _make_campaign_read_only(self.partial_path)
            os.replace(self.partial_path, self.promoted_path)
            _fsync_directory(self.root)
            return _validate_promoted_campaign(self.promoted_path, self.contract)


def _load_campaign_record(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    try:
        record = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid campaign record {path.name}") from error
    if _canonical_bytes(record) != payload:
        raise RuntimeError(f"campaign record {path.name} is not canonical")
    required = {
        "schema_version",
        "pair_digest",
        "identity",
        "identity_digest",
        "response",
        "reward",
        "finish_source",
        "finish",
    }
    if set(record) != required or record["schema_version"] != ROLLOUT_CAMPAIGN_SCHEMA_VERSION:
        raise RuntimeError(f"campaign record {path.name} has an invalid schema")
    if not isinstance(record["response"], str):
        raise RuntimeError(f"campaign record {path.name} has an invalid response")
    try:
        identity = RolloutResponseIdentity.from_dict(record["identity"])
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"campaign record {path.name} has an invalid identity") from error
    if identity.digest != record["identity_digest"]:
        raise RuntimeError(f"campaign record {path.name} identity digest mismatch")
    pair_digest = RolloutCampaignArtifact._pair_digest(identity.prompt_id, identity.sample_index)
    if record["pair_digest"] != pair_digest or path.stem != pair_digest or path.parent.name != pair_digest[:2]:
        raise RuntimeError(f"campaign record {path.name} pair digest mismatch")
    if (
        isinstance(record["reward"], bool)
        or not isinstance(record["reward"], int | float)
        or not math.isfinite(record["reward"])
    ):
        raise RuntimeError(f"campaign record {path.name} has an invalid reward")
    finish_source = record["finish_source"]
    if not isinstance(finish_source, dict) or set(finish_source) != {
        "finish_reason",
        "backend_stop_reason",
        "repetition_truncated",
        "response_token_ids",
        "eos_token_ids",
        "stop_token_ids",
    }:
        raise RuntimeError(f"campaign record {path.name} has an invalid finish source")
    token_ids = finish_source["response_token_ids"]
    if not isinstance(token_ids, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in token_ids
    ):
        raise RuntimeError(f"campaign record {path.name} has invalid response token IDs")
    recomputed_repetition = ConsecutiveRepetitionDetector().observe(token_ids) is not None
    if finish_source["repetition_truncated"] != recomputed_repetition:
        raise RuntimeError(f"campaign record {path.name} repetition metadata mismatch")
    expected_finish = build_rollout_finish_metadata(record["response"], **finish_source)
    if record["finish"] != expected_finish:
        raise RuntimeError(f"campaign record {path.name} finish metadata mismatch")
    if set(record["finish"]) != set(ROLLOUT_FINISH_METADATA_KEYS):
        raise RuntimeError(f"campaign record {path.name} has incomplete finish metadata")
    categories = [record["finish"][key] for key in ROLLOUT_FINISH_METADATA_KEYS[2:]]
    if any(not isinstance(value, bool) for value in categories) or sum(categories) != 1:
        raise RuntimeError(f"campaign record {path.name} has invalid finish categories")
    return record, payload


def _campaign_record_path(campaign_path: Path, pair_digest: str) -> Path:
    return campaign_path / "records" / pair_digest[:2] / f"{pair_digest}.json"


def _campaign_record_paths(campaign_path: Path) -> Iterator[Path]:
    records_dir = campaign_path / "records"
    if not records_dir.is_dir():
        return
    for shard_path in sorted(records_dir.iterdir()):
        if (
            shard_path.is_symlink()
            or not shard_path.is_dir()
            or len(shard_path.name) != 2
            or any(character not in "0123456789abcdef" for character in shard_path.name)
        ):
            raise RuntimeError(f"campaign records contain an unexpected entry: {shard_path.name}")
        for record_path in sorted(shard_path.iterdir()):
            if record_path.is_symlink() or not record_path.is_file() or record_path.suffix != ".json":
                raise RuntimeError(
                    f"campaign record shard {shard_path.name} contains an unexpected entry: {record_path.name}"
                )
            yield record_path


def _validate_campaign_identity(
    identity: RolloutResponseIdentity,
    contract: Mapping[str, Any],
) -> None:
    parameters = contract["parameters"]
    expected = {
        "sampling": parameters["sampling"],
        "model_revision": parameters["model_revision"],
        "policy_lineage": parameters["policy_lineage"],
        "policy_version": parameters["policy_version"],
        "source_lineage": parameters["source_lineage"],
        "runtime_identity": parameters["runtime_identity"],
        "seed": deterministic_campaign_seed(
            contract["campaign_id"],
            identity.prompt_id,
            identity.sample_index,
            parameters["base_seed"],
        ),
    }
    actual = {
        "sampling": dict(identity.sampling),
        "model_revision": identity.model_revision,
        "policy_lineage": identity.policy_lineage,
        "policy_version": identity.policy_version,
        "source_lineage": identity.source_lineage,
        "runtime_identity": identity.runtime_identity,
        "seed": identity.seed,
    }
    mismatched = [key for key, value in expected.items() if actual[key] != value]
    if mismatched:
        raise ValueError(f"response identity conflicts with campaign fields: {', '.join(mismatched)}")


def _build_campaign_manifest(
    path: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    contract_path = path / "contract.json"
    if not contract_path.is_file() or contract_path.read_bytes() != _canonical_bytes(contract):
        raise RuntimeError("campaign contract is missing or non-canonical")
    expected_count = contract["expected_pair_count"]
    problem_ids = frozenset(contract["problem_ids"])
    rollouts_per_problem = contract["rollouts_per_problem"]

    count_keys = (
        "format_valid",
        "ended_by_eos",
        "ended_by_stop_token",
        "repetition_truncated",
        "context_exhausted",
        "other_failure",
    )
    counts = dict.fromkeys(count_keys, 0)
    records_digest = hashlib.sha256()
    total = 0
    for record_path in _campaign_record_paths(path):
        total += 1
        record, payload = _load_campaign_record(record_path)
        identity = RolloutResponseIdentity.from_dict(record["identity"])
        try:
            _validate_campaign_identity(identity, contract)
        except ValueError as error:
            raise RuntimeError(f"campaign record {record_path.name} conflicts with its contract") from error
        pair = (identity.prompt_id, identity.sample_index)
        digest = RolloutCampaignArtifact._pair_digest(*pair)
        if digest != record_path.stem or pair[0] not in problem_ids or not 0 <= pair[1] < rollouts_per_problem:
            raise RuntimeError(f"campaign record {record_path.name} has an invalid pair")
        finish = record["finish"]
        for key in count_keys:
            counts[key] += int(finish[key])
        records_digest.update(record_path.stem.encode("ascii"))
        records_digest.update(b":")
        records_digest.update(payload)
    if total != expected_count:
        missing_preview = []
        for problem_id in contract["problem_ids"]:
            for rollout_index in range(rollouts_per_problem):
                digest = RolloutCampaignArtifact._pair_digest(problem_id, rollout_index)
                if not _campaign_record_path(path, digest).is_file():
                    missing_preview.append((problem_id, rollout_index))
                    if len(missing_preview) == 8:
                        break
            if len(missing_preview) == 8:
                break
        raise RuntimeError(
            f"campaign is incomplete: expected={expected_count}, actual={total}, missing_preview={missing_preview}"
        )
    return {
        "schema_version": ROLLOUT_CAMPAIGN_SCHEMA_VERSION,
        "contract": dict(contract),
        "counts": {"total": total, **counts},
        "rates": {name: count / total for name, count in counts.items()},
        "records_sha256": records_digest.hexdigest(),
    }


def _make_campaign_read_only(path: Path) -> None:
    file_mode = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    directory_mode = stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    records_path = path / "records"
    for file_path in _campaign_record_paths(path):
        file_path.chmod(file_mode)
    (path / "contract.json").chmod(file_mode)
    (path / "manifest.json").chmod(file_mode)
    for shard_path in records_path.iterdir():
        if shard_path.is_dir():
            shard_path.chmod(directory_mode)
    records_path.chmod(directory_mode)
    path.chmod(directory_mode)


def _validate_promoted_campaign(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not path.is_dir() or not manifest_path.is_file():
        raise RuntimeError("promoted campaign is missing its manifest")
    payload = manifest_path.read_bytes()
    try:
        manifest = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError("promoted campaign manifest is invalid") from error
    if _canonical_bytes(manifest) != payload or manifest.get("contract") != contract:
        raise RuntimeError("promoted campaign manifest is non-canonical or conflicts with the contract")
    if _build_campaign_manifest(path, contract) != manifest:
        raise RuntimeError("promoted campaign digest mismatch")
    readonly_paths = [
        path,
        path / "records",
        path / "contract.json",
        manifest_path,
        *((path / "records").glob("*")),
        *_campaign_record_paths(path),
    ]
    if any(item.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) for item in readonly_paths):
        raise RuntimeError("promoted campaign must be read-only")
    return manifest


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
    if set(finish_source) != {
        "finish_reason",
        "backend_stop_reason",
        "repetition_truncated",
        "response_token_ids",
        "eos_token_ids",
        "stop_token_ids",
    }:
        raise RuntimeError(f"response record {path.name} has an invalid finish source")
    detector = ConsecutiveRepetitionDetector()
    recomputed_repetition = detector.observe(finish_source["response_token_ids"]) is not None
    if finish_source["repetition_truncated"] != recomputed_repetition:
        raise RuntimeError(f"response record {path.name} repetition metadata mismatch")
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
    """Validate an immutable promoted shard and return its verified manifest."""

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
    "RolloutCampaignArtifact",
    "RolloutShardArtifact",
    "ROLLOUT_CAMPAIGN_SCHEMA_VERSION",
    "SHARD_ARTIFACT_SCHEMA_VERSION",
    "validate_promoted_shard",
]
