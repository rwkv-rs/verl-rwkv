from collections import defaultdict

import pytest
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.utils import create_rl_sampler
from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.effective_groups import (
    CandidateDatasetPosition,
    CandidateWavePlanner,
    EffectiveRoundState,
    EffectiveSamplingCheckpoint,
    GracefulRoundCancellation,
    GroupOutcome,
    binary_success_from_rewards,
    classify_complete_groups,
    effective_training_should_stop,
)
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


def test_binary_success_matches_maxrl_reward_threshold():
    rewards = torch.tensor([[0.0, -1.0], [0.0, 0.1], [2.0, -1.0]])
    assert binary_success_from_rewards(rewards).tolist() == [False, True, True]


@pytest.mark.parametrize(
    ("step_limit", "global_step", "passes", "target_passes", "signal", "expected"),
    [
        (2, 1, 10, 10, False, False),
        (2, 2, 0, 10, False, True),
        (None, 200, 9, 10, False, False),
        (None, 1, 10, 10, False, True),
        (None, 1, 0, 10, True, True),
    ],
)
def test_effective_training_stop_boundary(
    step_limit, global_step, passes, target_passes, signal, expected
):
    assert (
        effective_training_should_stop(
            global_step=global_step,
            configured_total_training_steps=step_limit,
            candidate_dataset_passes_completed=passes,
            target_candidate_dataset_passes=target_passes,
            graceful_stop_requested=signal,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("successes", "outcome"),
    [
        (0, GroupOutcome.ALL_WRONG),
        (1, GroupOutcome.EFFECTIVE),
        (15, GroupOutcome.EFFECTIVE),
        (16, GroupOutcome.ALL_CORRECT),
    ],
)
def test_classifies_binary_group_boundaries(successes, outcome):
    selection = classify_complete_groups(
        ["prompt"] * 16,
        torch.tensor([True] * successes + [False] * (16 - successes)),
        responses_per_prompt=16,
        target_groups=32,
    )
    groups = (*selection.accepted, *selection.rejected)
    assert len(groups) == 1
    assert groups[0].outcome is outcome


def test_rejects_incomplete_group():
    with pytest.raises(ValueError, match="incomplete MaxRL group"):
        classify_complete_groups(
            ["prompt"] * 15,
            torch.zeros(15, dtype=torch.bool),
            responses_per_prompt=16,
            target_groups=32,
        )


def test_caps_effective_groups_in_first_seen_order():
    uids = [uid for uid in ("a", "b", "c") for _ in range(2)]
    success = torch.tensor([True, False] * 3)
    selection = classify_complete_groups(
        uids,
        success,
        responses_per_prompt=2,
        target_groups=2,
    )
    assert [group.uid for group in selection.accepted] == ["a", "b"]
    assert [group.uid for group in selection.surplus] == ["c"]


def test_candidate_wave_planner_uses_rate_quantum_and_limit():
    planner = CandidateWavePlanner(
        target_groups=32,
        group_quantum=32,
        max_candidate_groups=1024,
        max_wave_groups=256,
    )
    assert planner.plan(accepted_groups=0) == 96
    planner.observe(candidate_groups=96, effective_groups=48)
    assert planner.plan(accepted_groups=24) == 32


def test_candidate_wave_planner_stops_at_safety_limit():
    planner = CandidateWavePlanner(
        target_groups=32,
        group_quantum=32,
        max_candidate_groups=64,
        max_wave_groups=64,
    )
    assert planner.plan(0) == 64
    planner.observe(candidate_groups=64, effective_groups=0)
    assert planner.plan(0) == 0


def test_round_state_tracks_inflight_order_rejections_and_surplus():
    state = EffectiveRoundState.create(target_groups=2, responses_per_prompt=2)
    state.submit_wave(4)
    selection = classify_complete_groups(
        [uid for uid in ("effective-a", "wrong", "effective-b", "surplus") for _ in range(2)],
        torch.tensor(
            [True, False, False, False, True, False, False, True],
            dtype=torch.bool,
        ),
        responses_per_prompt=2,
        target_groups=2,
    )

    state.complete_wave(selection)

    assert state.in_flight_groups == 0
    assert state.candidate_order == ["effective-a", "wrong", "effective-b", "surplus"]
    assert [group.uid for group in state.accepted] == ["effective-a", "effective-b"]
    assert [group.uid for group in state.rejected] == ["wrong"]
    assert [group.uid for group in state.surplus] == ["surplus"]


def test_candidate_dataset_position_crosses_passes_without_losing_cursor():
    position = CandidateDatasetPosition(completed_passes=0, cursor=17_397)
    assert position.advance(1, dataset_size=17_398) == CandidateDatasetPosition(1, 0)
    assert position.advance(3, dataset_size=17_398) == CandidateDatasetPosition(1, 2)
    assert CandidateDatasetPosition(9, 17_390).advance(20, dataset_size=17_398) == CandidateDatasetPosition(10, 12)


def test_effective_sampling_checkpoint_restores_committed_progress():
    state = EffectiveSamplingCheckpoint.from_dict(
        {
            "schema_version": 1,
            "candidate_dataset_passes_completed": 3,
            "candidate_prompts_in_current_pass": 41,
            "acceptance_rate": 0.25,
            "policy_version": 17,
            "optimizer_step": 17,
            "metric_totals": {"candidate_groups": 96, "accepted_groups": 32},
        },
        dataset_size=17_398,
        expected_optimizer_step=17,
    )

    assert state.position == CandidateDatasetPosition(3, 41)
    assert state.acceptance_rate == 0.25
    assert state.metric_totals == {"candidate_groups": 96.0, "accepted_groups": 32.0}


@pytest.mark.parametrize(
    "override",
    [
        {"schema_version": 0},
        {"candidate_prompts_in_current_pass": 17_398},
        {"acceptance_rate": 0.0},
        {"policy_version": 16},
        {"optimizer_step": 16},
        {"metric_totals": {"candidate_groups": -1}},
    ],
)
def test_effective_sampling_checkpoint_fails_closed(override):
    payload = {
        "schema_version": 1,
        "candidate_dataset_passes_completed": 3,
        "candidate_prompts_in_current_pass": 41,
        "acceptance_rate": 0.25,
        "policy_version": 17,
        "optimizer_step": 17,
        "metric_totals": {},
    }
    payload.update(override)

    with pytest.raises(ValueError):
        EffectiveSamplingCheckpoint.from_dict(
            payload,
            dataset_size=17_398,
            expected_optimizer_step=17,
        )


def test_candidate_pass_shuffle_is_seeded_distinct_and_resumeable():
    dataset = list(range(19))
    config = OmegaConf.create({"shuffle": True, "seed": 1234})

    sampler_a = create_rl_sampler(config, dataset)
    first_pass = list(iter(sampler_a))
    second_pass = list(iter(sampler_a))
    sampler_b = create_rl_sampler(config, dataset)
    assert first_pass == list(iter(sampler_b))
    assert first_pass != second_pass
    assert sorted(first_pass) == sorted(second_pass) == list(range(19))

    loader = StatefulDataLoader(
        dataset,
        batch_size=1,
        sampler=create_rl_sampler(config, dataset),
        num_workers=0,
    )
    iterator = iter(loader)
    consumed = [next(iterator).item() for _ in range(7)]
    state = loader.state_dict()
    remainder = [item.item() for item in iterator]

    resumed = StatefulDataLoader(
        dataset,
        batch_size=1,
        sampler=create_rl_sampler(config, dataset),
        num_workers=0,
    )
    resumed.load_state_dict(state)
    assert [item.item() for item in resumed] == remainder
    assert len(consumed) + len(remainder) == len(dataset)


class _ListField:
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter(self.values)


class _RewardField:
    def __init__(self, values):
        self.values = torch.tensor(values, dtype=torch.float32)

    def to_padded_tensor(self, _padding):
        return self.values


def _collector_fixture(monkeypatch, waves, *, graceful=False):
    batches = []
    reward_rows = {}
    for wave_number, groups in enumerate(waves):
        keys, tags, uids, indices, rewards = [], [], [], [], []
        for group_number, successes in enumerate(groups):
            uid = f"wave-{wave_number}-group-{group_number}"
            for response in range(2):
                key = f"{uid}-{response}"
                keys.append(key)
                tags.append({})
                uids.append(uid)
                indices.append(wave_number * 10 + group_number)
                rewards.append([float(response < successes)])
        batches.append(KVBatchMeta(partition_id="train", keys=keys, tags=tags))
        reward_rows[tuple(keys)] = {
            "uid": _ListField(uids),
            "index": _ListField(indices),
            "rm_scores": _RewardField(rewards),
        }

    cleared = []
    monkeypatch.setattr(
        trainer_base.tq,
        "kv_batch_get",
        lambda *, keys, **_kwargs: reward_rows[tuple(keys)],
    )
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", lambda **_kwargs: None)
    monkeypatch.setattr(
        trainer_base.tq,
        "kv_clear",
        lambda *, keys, **_kwargs: cleared.extend(keys),
    )

    class Replay:
        def sample(self, **_kwargs):
            return batches.pop(0), {}

    class Loader:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, state):
            self.loaded = state

    trainer = type("CollectorTrainer", (), {})()
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 2}}})
    trainer._effective_group_planner = CandidateWavePlanner(
        target_groups=2,
        group_quantum=1,
        max_candidate_groups=8,
        max_wave_groups=2,
        acceptance_rate=1.0,
        headroom=1.0,
    )
    trainer._effective_group_acceptance_rate = 1.0
    trainer._effective_sampling_totals = defaultdict(float)
    trainer.replay_buffer = Replay()
    trainer.reward_loop_manager = type(
        "RewardManager", (), {"reward_loop_worker_handles": [object()]}
    )()
    trainer.global_steps = 1
    trainer._graceful_stop_requested = graceful
    trainer._add_prompts_to_generate = lambda count: count
    trainer.train_dataloader = Loader()
    trainer.train_dataloader_it = object()
    trainer._candidate_dataset_passes_completed = 4
    trainer._candidate_prompts_in_current_pass = 20
    trainer._effective_round_snapshot = {
        "dataloader": {"snapshot": True},
        "candidate_dataset_passes_completed": 3,
        "candidate_prompts_in_current_pass": 10,
        "acceptance_rate": 0.5,
        "metric_totals": {"candidate_groups": 10},
    }
    return trainer, cleared


def test_collector_refills_to_exact_effective_batch_and_clears_rejections(
    monkeypatch,
):
    trainer, cleared = _collector_fixture(monkeypatch, [[1, 0], [1]])
    metrics = {}

    batch, _ = PPOTrainer._sample_effective_maxrl_batch(
        trainer,
        target_groups=2,
        metrics=metrics,
    )

    assert len(batch.keys) == 4
    assert len(cleared) == 2
    assert metrics["training/effective_sampling/candidate_groups"] == 3
    assert metrics["training/effective_sampling/accepted_groups"] == 2
    assert metrics["training/effective_sampling/refill_waves"] == 2
    assert metrics["training/effective_sampling/effective_batch_trajectories"] == 4
    assert metrics["training/effective_sampling/candidate_data_id_unique"] == 3
    assert metrics["training/effective_sampling/accepted_data_id_unique"] == 2


def test_graceful_stop_cancels_incomplete_round_and_restores_cursor(monkeypatch):
    trainer, cleared = _collector_fixture(monkeypatch, [[1, 0]], graceful=True)

    with pytest.raises(GracefulRoundCancellation):
        PPOTrainer._sample_effective_maxrl_batch(
            trainer,
            target_groups=2,
            metrics={},
        )

    assert len(cleared) == 4
    assert trainer.train_dataloader.loaded == {"snapshot": True}
    assert trainer.train_dataloader_it is None
    assert trainer._candidate_dataset_passes_completed == 3
    assert trainer._candidate_prompts_in_current_pass == 10
    assert trainer._effective_group_acceptance_rate == 0.5
