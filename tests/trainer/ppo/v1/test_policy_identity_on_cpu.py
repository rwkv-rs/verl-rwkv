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

from copy import deepcopy

import pytest

from verl.trainer.ppo.v1.policy_identity import (
    BehaviorPolicyIdentity,
    PolicyIdentityError,
    RoundPhase,
    StrictOnPolicyRound,
    checkpoint_policy_version,
    training_policy_version,
    validate_behavior_policy_batch,
)
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync, record_policy_identity
from verl.workers.engine_workers import validate_strict_on_policy_optimizer_iterations


def _identity(version: int, previous: str = "checkpoint") -> BehaviorPolicyIdentity:
    return BehaviorPolicyIdentity.for_publication(
        policy_version=version,
        previous_weight_digest=previous,
        export_contract={"engine": "rwkv_lm", "dtype": "bfloat16"},
        sampling_config={"temperature": 1.0, "top_p": 1.0},
        runtime_identity="vllm:async:rwkv7",
    )


def _tags(identity: BehaviorPolicyIdentity, group_count: int = 2, group_size: int = 3):
    tags = []
    for group_index in range(group_count):
        for response_index in range(group_size):
            tags.append(
                {
                    **identity.as_dict(),
                    "group_id": f"prompt-{group_index}",
                    "group_size": group_size,
                    "response_index": response_index,
                    "min_global_steps": identity.policy_version,
                    "max_global_steps": identity.policy_version,
                    "request_sampling_digest": "effective-sampling",
                }
            )
    return tags


def _acks(identity: BehaviorPolicyIdentity):
    return [{**identity.as_dict(), "replica_id": replica_id} for replica_id in range(8)]


def _publish_initial(state: StrictOnPolicyRound, identity: BehaviorPolicyIdentity, acknowledgements=None):
    state.publish_initial(
        identity,
        _acks(identity) if acknowledgements is None else acknowledgements,
        source_policy_version=identity.policy_version,
        source_policy_lineage=f"checkpoint:{identity.weight_digest}",
    )


def _begin_publication(state: StrictOnPolicyRound, target: BehaviorPolicyIdentity):
    source = state.training_identity
    assert source is not None
    return state.begin_publication(
        target,
        source_policy_version=source.policy_version,
        source_policy_lineage=source.weight_digest,
    )


def test_global_step_adapter_preserves_fresh_and_restored_lineage():
    assert checkpoint_policy_version(0) == 0
    assert checkpoint_policy_version(17) == 17
    assert training_policy_version(1) == 0
    assert training_policy_version(18) == 17

    with pytest.raises(PolicyIdentityError):
        checkpoint_policy_version(-1)
    with pytest.raises(PolicyIdentityError):
        training_policy_version(0)


def test_identity_is_canonical_and_changes_with_publication_lineage():
    left = BehaviorPolicyIdentity.for_publication(
        policy_version=4,
        previous_weight_digest="old",
        export_contract={"b": 2, "a": 1},
        sampling_config={"top_p": 1.0, "temperature": 0.8},
        runtime_identity="runtime",
    )
    reordered = BehaviorPolicyIdentity.for_publication(
        policy_version=4,
        previous_weight_digest="old",
        export_contract={"a": 1, "b": 2},
        sampling_config={"temperature": 0.8, "top_p": 1.0},
        runtime_identity="runtime",
    )
    next_version = BehaviorPolicyIdentity.for_publication(
        policy_version=5,
        previous_weight_digest=left.weight_digest,
        export_contract={"a": 1, "b": 2},
        sampling_config={"temperature": 0.8, "top_p": 1.0},
        runtime_identity="runtime",
    )

    assert left == reordered
    assert next_version.weight_digest != left.weight_digest


def test_policy_identity_artifact_records_exact_lineage(tmp_path, monkeypatch):
    output = tmp_path / "policy_identity.jsonl"
    monkeypatch.setenv("VERL_POLICY_IDENTITY_LOG_PATH", str(output))
    identity = _identity(4)

    record_policy_identity("train_begin", 5, identity)

    assert output.read_text().strip()
    assert '"event": "train_begin"' in output.read_text()
    assert f'"weight_digest": "{identity.weight_digest}"' in output.read_text()


def test_complete_same_identity_groups_are_accepted():
    identity = _identity(7)
    validate_behavior_policy_batch(_tags(identity), expected=identity)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda tags: tags[0].pop("policy_version"), "missing"),
        (lambda tags: tags[0].__setitem__("policy_version", 6), "mismatch"),
        (lambda tags: tags[0].__setitem__("weight_digest", "stale"), "mismatch"),
        (lambda tags: tags[0].__setitem__("sampling_config_digest", "mixed"), "mismatch"),
        (lambda tags: tags[0].__setitem__("request_sampling_digest", "mixed"), "mixed effective"),
        (lambda tags: tags[0].__setitem__("runtime_identity", "other"), "mismatch"),
        (lambda tags: tags[0].__setitem__("min_global_steps", 6), "version span"),
        (lambda tags: tags.pop(), "incomplete"),
        (lambda tags: tags[0].__setitem__("group_id", "other-group"), "incomplete"),
    ],
)
def test_batch_gate_rejects_missing_stale_mixed_cross_identity_and_incomplete_groups(mutation, message):
    identity = _identity(7)
    tags = _tags(identity)
    mutation(tags)
    with pytest.raises(PolicyIdentityError, match=message):
        validate_behavior_policy_batch(tags, expected=identity)


def test_round_state_machine_bootstrap_train_publish_two_consecutive_rounds():
    state = StrictOnPolicyRound()
    identity0 = _identity(0)
    _publish_initial(state, identity0)

    for version in (0, 1):
        current = state.begin_rollout(version)
        state.prepare(_tags(current))
        state.begin_training()
        next_identity = _identity(version + 1, previous=current.weight_digest)
        _begin_publication(state, next_identity)
        state.commit_publication(next_identity, _acks(next_identity))

    assert state.phase is RoundPhase.PUBLISHED
    assert state.published.policy_version == 2


def test_restore_does_not_increment_or_reset_policy_lineage():
    restored = _identity(23)
    state = StrictOnPolicyRound()
    _publish_initial(state, restored)
    assert state.begin_rollout(23) == restored


@pytest.mark.parametrize(
    ("source_version", "source_lineage", "message"),
    [(-1, "checkpoint", "source policy version"), (0, "", "source policy lineage")],
)
def test_initial_publication_requires_explicit_valid_source_lineage(source_version, source_lineage, message):
    identity = _identity(0)
    state = StrictOnPolicyRound()

    with pytest.raises(PolicyIdentityError, match=message):
        state.publish_initial(
            identity,
            _acks(identity),
            source_policy_version=source_version,
            source_policy_lineage=source_lineage,
        )


def test_publication_acknowledgements_are_incremental_idempotent_and_atomic():
    current = _identity(5)
    target = _identity(6, previous=current.weight_digest)
    state = StrictOnPolicyRound(expected_replica_ids=range(2))
    _publish_initial(state, current, _acks(current)[:2])
    state.begin_rollout(5)
    state.prepare(_tags(current))
    state.begin_training()
    assert _begin_publication(state, target) is True
    assert (
        state.begin_publication(
            target,
            source_policy_version=current.policy_version,
            source_policy_lineage=current.weight_digest,
        )
        is False
    )

    assert state.acknowledge_publication(_acks(target)[0]) is True
    assert state.acknowledge_publication(_acks(target)[0]) is False
    assert state.published == current
    state.commit_publication(target, [_acks(target)[1]])

    assert state.published == target
    assert state.begin_rollout(6) == target


@pytest.mark.parametrize("failure", ["stale", "conflicting"])
def test_stale_or_conflicting_ack_rolls_back_without_partial_visibility(failure):
    current = _identity(5)
    target = _identity(6, previous=current.weight_digest)
    state = StrictOnPolicyRound(expected_replica_ids=range(2))
    _publish_initial(state, current, _acks(current)[:2])
    state.begin_rollout(5)
    state.prepare(_tags(current))
    state.begin_training()
    _begin_publication(state, target)
    state.acknowledge_publication(_acks(target)[0])
    bad = {**(_identity(5) if failure == "stale" else _identity(7)).as_dict(), "replica_id": 1}

    with pytest.raises(PolicyIdentityError, match="does not match"):
        state.acknowledge_publication(bad)

    assert state.phase is RoundPhase.PUBLISHED
    assert state.published == current
    assert state.publication is None


def test_publication_is_refused_while_rollout_is_active():
    current = _identity(2)
    state = StrictOnPolicyRound()
    _publish_initial(state, current)
    state.begin_rollout(2)

    with pytest.raises(PolicyIdentityError, match="publication is invalid in phase rollout"):
        state.begin_publication(
            _identity(3, previous=current.weight_digest),
            source_policy_version=2,
            source_policy_lineage=current.weight_digest,
        )

    assert state.published == current


def test_publication_source_must_match_active_committed_lineage():
    current = _identity(2)
    state = StrictOnPolicyRound()
    _publish_initial(state, current)
    state.begin_rollout(2)
    state.prepare(_tags(current))
    state.begin_training()

    with pytest.raises(PolicyIdentityError, match="source lineage"):
        state.begin_publication(
            _identity(3, previous=current.weight_digest),
            source_policy_version=2,
            source_policy_lineage="different-lineage",
        )


def test_conflicting_repeated_publication_update_rolls_back():
    current = _identity(2)
    state = StrictOnPolicyRound()
    _publish_initial(state, current)
    state.begin_rollout(2)
    state.prepare(_tags(current))
    state.begin_training()
    _begin_publication(state, _identity(3, previous=current.weight_digest))

    with pytest.raises(PolicyIdentityError, match="conflicting weight publication update"):
        state.begin_publication(
            _identity(4, previous=current.weight_digest),
            source_policy_version=2,
            source_policy_lineage=current.weight_digest,
        )

    assert state.phase is RoundPhase.PUBLISHED
    assert state.published == current


def test_failed_or_mixed_replica_ack_never_commits_publication():
    current = _identity(3)
    state = StrictOnPolicyRound()
    _publish_initial(state, current)
    state.begin_rollout(3)
    state.prepare(_tags(current))
    state.begin_training()
    candidate = _identity(4, previous=current.weight_digest)
    _begin_publication(state, candidate)

    bad_ack = deepcopy(candidate.as_dict())
    bad_ack["weight_digest"] = "partial-update"
    with pytest.raises(PolicyIdentityError, match="does not match"):
        state.commit_publication(candidate, _acks(candidate)[:-1] + [{**bad_ack, "replica_id": 7}])

    assert state.phase is RoundPhase.PUBLISHED
    assert state.published == current


@pytest.mark.parametrize(
    "acks",
    [
        lambda identity: _acks(identity)[:-1],
        lambda identity: _acks(identity)[:-1] + [{**identity.as_dict(), "replica_id": 6}],
    ],
)
def test_publication_rejects_missing_or_duplicate_replica_acknowledgements(acks):
    identity = _identity(0)
    state = StrictOnPolicyRound()
    with pytest.raises(PolicyIdentityError, match="replica"):
        _publish_initial(state, identity, acks(identity))


def test_optimizer_failure_path_cannot_publish_or_advance_version():
    current = _identity(8)
    state = StrictOnPolicyRound()
    _publish_initial(state, current)
    state.begin_rollout(8)
    state.prepare(_tags(current))
    state.begin_training()

    # on_step_end is never entered after the optimizer raises, so no publication
    # transition is legal and the last acknowledged identity remains current.
    assert state.phase is RoundPhase.TRAINING
    assert state.published == current
    with pytest.raises(PolicyIdentityError, match="commit is invalid"):
        state.commit_publication(_identity(9, previous=current.weight_digest), [])


def test_next_round_cannot_roll_out_before_publication_commits():
    current = _identity(1)
    state = StrictOnPolicyRound()
    _publish_initial(state, current)
    state.begin_rollout(1)
    state.prepare(_tags(current))
    state.begin_training()

    with pytest.raises(PolicyIdentityError, match="rollout is invalid"):
        state.begin_rollout(1)


def test_strict_actor_update_allows_exactly_one_optimizer_iteration():
    validate_strict_on_policy_optimizer_iterations(enabled=True, epochs=1, iterations=1)
    for epochs, iterations in ((2, 2), (1, 2), (2, 1)):
        with pytest.raises(RuntimeError, match="exactly one optimizer step"):
            validate_strict_on_policy_optimizer_iterations(
                enabled=True,
                epochs=epochs,
                iterations=iterations,
            )

    # Non-strict trainers retain their existing multi-epoch/multi-mini-batch behavior.
    validate_strict_on_policy_optimizer_iterations(enabled=False, epochs=4, iterations=8)


class _CheckpointManager:
    def __init__(self, events, error=None):
        self.events = events
        self.error = error

    def sleep_replicas(self):
        self.events.append("sleep")

    def update_weights(self, global_steps, policy_identity):
        self.events.append(("update", global_steps, policy_identity))
        if self.error is not None:
            raise self.error
        return [{**policy_identity, "replica_id": replica_id} for replica_id in range(8)]


def _training_sync(checkpoint_manager):
    trainer = PPOTrainerSync.__new__(PPOTrainerSync)
    trainer.checkpoint_manager = checkpoint_manager
    trainer.timing_raw = {}
    trainer.global_steps = 1
    trainer._sampling_config = {"temperature": 1.0}
    trainer._runtime_identity = "vllm:async:rwkv7"
    trainer._export_contract = {"engine": "rwkv_lm", "dtype": "bfloat16"}
    current = _identity(0)
    trainer.policy_round = StrictOnPolicyRound()
    _publish_initial(trainer.policy_round, current)
    trainer.policy_round.begin_rollout(0)
    trainer.policy_round.prepare(_tags(current))
    trainer.policy_round.begin_training()
    return trainer


def test_sync_lifecycle_sleeps_then_updates_all_replicas_and_commits_once():
    events = []
    trainer = _training_sync(_CheckpointManager(events))

    trainer.on_sample_end()
    trainer.on_step_end()

    assert events[0] == "sleep"
    assert events[1][0:2] == ("update", 1)
    assert trainer.policy_round.phase is RoundPhase.PUBLISHED
    assert trainer.policy_round.published.policy_version == 1


@pytest.mark.parametrize("error", [TimeoutError("timed out"), RuntimeError("replica update failed")])
def test_sync_lifecycle_update_failure_or_timeout_never_commits(error):
    events = []
    trainer = _training_sync(_CheckpointManager(events, error=error))
    previous = trainer.policy_round.published

    trainer.on_sample_end()
    with pytest.raises(type(error), match=str(error)):
        trainer.on_step_end()

    assert trainer.policy_round.phase is RoundPhase.PUBLISHED
    assert trainer.policy_round.published == previous
