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

"""Strict on-policy behavior-policy identity and round state machine.

``global_steps`` is an execution-loop counter in V1.  It is deliberately not
used directly as the external policy lineage: while training step ``s`` runs,
the actor and every consumed rollout must still be policy ``s - 1``.  This
module is the single adapter between those two meanings.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

IDENTITY_TAG_KEYS = (
    "policy_version",
    "weight_digest",
    "sampling_config_digest",
    "runtime_identity",
)


class PolicyIdentityError(RuntimeError):
    """Raised before policy loss when strict on-policy lineage is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def canonical_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible identity material."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def checkpoint_policy_version(global_steps: int) -> int:
    """Map a fresh/restored V1 checkpoint counter to its published policy."""

    if not isinstance(global_steps, int) or global_steps < 0:
        raise PolicyIdentityError(f"checkpoint global_steps must be a non-negative integer, got {global_steps!r}")
    return global_steps


def training_policy_version(global_steps: int) -> int:
    """Return the actor version consumed by V1 training step ``global_steps``."""

    if not isinstance(global_steps, int) or global_steps <= 0:
        raise PolicyIdentityError(f"training global_steps must be a positive integer, got {global_steps!r}")
    return global_steps - 1


@dataclass(frozen=True)
class BehaviorPolicyIdentity:
    """Immutable lineage shared by rollout requests, responses and training."""

    policy_version: int
    weight_digest: str
    sampling_config_digest: str
    runtime_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.policy_version, int) or self.policy_version < 0:
            raise PolicyIdentityError(f"invalid policy_version: {self.policy_version!r}")
        for key in IDENTITY_TAG_KEYS[1:]:
            value = getattr(self, key)
            if not isinstance(value, str) or not value:
                raise PolicyIdentityError(f"{key} must be a non-empty string")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BehaviorPolicyIdentity:
        missing = [key for key in IDENTITY_TAG_KEYS if value.get(key) is None]
        if missing:
            raise PolicyIdentityError(f"missing behavior-policy identity fields: {', '.join(missing)}")
        return cls(**{key: value[key] for key in IDENTITY_TAG_KEYS})

    @classmethod
    def for_publication(
        cls,
        *,
        policy_version: int,
        previous_weight_digest: str,
        export_contract: Mapping[str, Any],
        sampling_config: Mapping[str, Any],
        runtime_identity: str,
    ) -> BehaviorPolicyIdentity:
        """Build the deterministic digest of one completed actor export.

        The digest binds the prior lineage, publication version and resolved
        export contract.  Every colocated worker receives this value only after
        its local tensor transfer completes; matching acknowledgements therefore
        form the all-replica publication barrier without copying model bytes back
        to the controller.
        """

        sampling_config_digest = canonical_digest(sampling_config)
        weight_digest = canonical_digest(
            {
                "previous_weight_digest": previous_weight_digest,
                "policy_version": policy_version,
                "export_contract": export_contract,
            }
        )
        return cls(
            policy_version=policy_version,
            weight_digest=weight_digest,
            sampling_config_digest=sampling_config_digest,
            runtime_identity=runtime_identity,
        )


class RoundPhase(str, Enum):
    UNINITIALIZED = "uninitialized"
    PUBLISHED = "published"
    ROLLOUT = "rollout"
    PREPARED = "prepared"
    TRAINING = "training"
    PUBLISHING = "publishing"
    FAILED = "failed"


class StrictOnPolicyRound:
    """Fail-closed controller for one-policy-per-round call ordering."""

    def __init__(self, expected_replica_ids: Iterable[int] = range(8)) -> None:
        self.phase = RoundPhase.UNINITIALIZED
        self.published: BehaviorPolicyIdentity | None = None
        self.training_identity: BehaviorPolicyIdentity | None = None
        self.expected_replica_ids = frozenset(expected_replica_ids)
        if not self.expected_replica_ids:
            raise PolicyIdentityError("strict publication requires at least one rollout replica")

    def _validate_acknowledgements(
        self, identity: BehaviorPolicyIdentity, acknowledgements: Iterable[Mapping[str, Any]]
    ) -> None:
        acknowledgements = list(acknowledgements)
        if not acknowledgements:
            raise PolicyIdentityError("weight publication returned no replica acknowledgements")
        replica_ids = [acknowledgement.get("replica_id") for acknowledgement in acknowledgements]
        if len(replica_ids) != len(set(replica_ids)):
            raise PolicyIdentityError(f"weight publication returned duplicate replica ids: {replica_ids}")
        if set(replica_ids) != self.expected_replica_ids:
            raise PolicyIdentityError(
                "weight publication did not cover the expected replica set: "
                f"expected={sorted(self.expected_replica_ids)} actual={sorted(replica_ids, key=str)}"
            )
        for index, acknowledgement in enumerate(acknowledgements):
            try:
                acknowledged = BehaviorPolicyIdentity.from_dict(acknowledgement)
            except (PolicyIdentityError, TypeError) as exc:
                raise PolicyIdentityError(f"invalid rollout replica acknowledgement {index}: {exc}") from exc
            if acknowledged != identity:
                raise PolicyIdentityError(
                    f"rollout replica acknowledgement {index} does not match publication: "
                    f"expected={identity.as_dict()} actual={acknowledged.as_dict()}"
                )

    def publish_initial(self, identity: BehaviorPolicyIdentity, acknowledgements: Iterable[Mapping[str, Any]]) -> None:
        if self.phase is not RoundPhase.UNINITIALIZED:
            raise PolicyIdentityError(f"initial publication is invalid in phase {self.phase.value}")
        self._validate_acknowledgements(identity, acknowledgements)
        self.published = identity
        self.phase = RoundPhase.PUBLISHED

    def begin_rollout(self, expected_policy_version: int) -> BehaviorPolicyIdentity:
        if self.phase is not RoundPhase.PUBLISHED or self.published is None:
            raise PolicyIdentityError(f"rollout is invalid in phase {self.phase.value}")
        if self.published.policy_version != expected_policy_version:
            raise PolicyIdentityError(
                f"rollout expects policy {expected_policy_version}, published policy is {self.published.policy_version}"
            )
        self.training_identity = self.published
        self.phase = RoundPhase.ROLLOUT
        return self.published

    def prepare(self, tags: Iterable[Mapping[str, Any]]) -> None:
        if self.phase is not RoundPhase.ROLLOUT or self.training_identity is None:
            raise PolicyIdentityError(f"prepare is invalid in phase {self.phase.value}")
        validate_behavior_policy_batch(tags, expected=self.training_identity)
        self.phase = RoundPhase.PREPARED

    def begin_training(self) -> None:
        if self.phase is not RoundPhase.PREPARED:
            raise PolicyIdentityError(f"training is invalid in phase {self.phase.value}")
        self.phase = RoundPhase.TRAINING

    def begin_publication(self, next_identity: BehaviorPolicyIdentity) -> None:
        if self.phase is not RoundPhase.TRAINING or self.training_identity is None:
            raise PolicyIdentityError(f"publication is invalid in phase {self.phase.value}")
        expected = self.training_identity.policy_version + 1
        if next_identity.policy_version != expected:
            raise PolicyIdentityError(
                f"publication must advance exactly once from {self.training_identity.policy_version} to {expected}, "
                f"got {next_identity.policy_version}"
            )
        self.phase = RoundPhase.PUBLISHING

    def commit_publication(
        self,
        identity: BehaviorPolicyIdentity,
        acknowledgements: Iterable[Mapping[str, Any]],
    ) -> None:
        if self.phase is not RoundPhase.PUBLISHING:
            raise PolicyIdentityError(f"publication commit is invalid in phase {self.phase.value}")
        try:
            self._validate_acknowledgements(identity, acknowledgements)
        except PolicyIdentityError as exc:
            self.fail(str(exc))
        self.published = identity
        self.training_identity = None
        self.phase = RoundPhase.PUBLISHED

    def fail(self, reason: str) -> None:
        self.phase = RoundPhase.FAILED
        raise PolicyIdentityError(reason)


def validate_behavior_policy_batch(tags: Iterable[Mapping[str, Any]], *, expected: BehaviorPolicyIdentity) -> None:
    """Validate complete prompt groups and one immutable behavior identity."""

    tags = list(tags)
    if not tags:
        raise PolicyIdentityError("strict on-policy batch is empty")

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for index, tag in enumerate(tags):
        required_keys = (*IDENTITY_TAG_KEYS, "request_sampling_digest", "group_id", "group_size", "response_index")
        missing = [key for key in required_keys if tag.get(key) is None]
        if missing:
            raise PolicyIdentityError(f"sample {index} is missing strict on-policy metadata: {', '.join(missing)}")
        actual = BehaviorPolicyIdentity.from_dict(tag)
        if actual != expected:
            raise PolicyIdentityError(
                f"sample {index} behavior-policy identity mismatch: "
                f"expected={expected.as_dict()} actual={actual.as_dict()}"
            )
        wrong_min = tag.get("min_global_steps") != expected.policy_version
        wrong_max = tag.get("max_global_steps") != expected.policy_version
        if wrong_min or wrong_max:
            raise PolicyIdentityError(
                f"sample {index} trajectory version span must equal policy {expected.policy_version}: "
                f"min={tag.get('min_global_steps')} max={tag.get('max_global_steps')}"
            )
        groups.setdefault(str(tag["group_id"]), []).append(tag)

    request_sampling_digests = {tag["request_sampling_digest"] for tag in tags}
    if len(request_sampling_digests) != 1:
        raise PolicyIdentityError(
            f"strict on-policy batch contains mixed effective request sampling: {sorted(request_sampling_digests)}"
        )

    for group_id, group_tags in groups.items():
        sizes = {tag["group_size"] for tag in group_tags}
        if len(sizes) != 1:
            raise PolicyIdentityError(f"prompt group {group_id} has mixed group_size metadata: {sorted(sizes)}")
        group_size = sizes.pop()
        indices = {tag["response_index"] for tag in group_tags}
        if len(group_tags) != group_size or indices != set(range(group_size)):
            raise PolicyIdentityError(
                f"prompt group {group_id} is incomplete: expected {group_size} responses, got indices {sorted(indices)}"
            )
