# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
from pathlib import Path
from uuid import uuid4

from omegaconf import OmegaConf

from verl.trainer.ppo.v1.policy_identity import (
    BehaviorPolicyIdentity,
    StrictOnPolicyRound,
    checkpoint_policy_version,
    training_policy_version,
)
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

POLICY_PUBLICATION_STATE_ENV = "VERL_POLICY_PUBLICATION_STATE_PATH"


def load_committed_policy_round(path: str | Path) -> StrictOnPolicyRound:
    """Load the last atomic publication; partial transfers are never serialized."""

    state_path = Path(path)
    try:
        payload = json.loads(state_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to load committed policy publication state {state_path}: {exc}") from exc
    return StrictOnPolicyRound.from_committed_state_dict(payload)


def save_committed_policy_round(path: str | Path, state: StrictOnPolicyRound) -> None:
    """Atomically persist the all-replica committed policy identity."""

    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state.committed_state_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    temporary = state_path.parent / f".{state_path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, state_path)
        directory = os.open(state_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def record_policy_identity(
    event: str,
    global_steps: int,
    identity: BehaviorPolicyIdentity,
    *,
    effective_sampling_digest: str | None = None,
) -> None:
    """Append the strict policy lineage to the run artifact, when configured."""

    output_path = os.getenv("VERL_POLICY_IDENTITY_LOG_PATH")
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "global_steps": global_steps, **identity.as_dict()}
    if effective_sampling_digest is not None:
        record["effective_sampling_digest"] = effective_sampling_digest
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    """Synchronous PPO trainer
    1. Trainer and rollout are colocated
    2. Partial rollout is disabled
    """

    def on_init_end(self):
        self.policy_round = StrictOnPolicyRound()
        self._sampling_config = OmegaConf.to_container(
            self.config.actor_rollout_ref.rollout,
            resolve=True,
        )
        model_path = str(self.config.actor_rollout_ref.model.path)
        run_id = os.getenv("HELICOPTER_RUN_ID", "untracked-local-run")
        checkpoint_origin = os.getenv("HELICOPTER_CHECKPOINT_SHA256", model_path)
        self._runtime_identity = f"vllm:async:{run_id}:{model_path}"
        self._export_contract = {
            "actor_engine": str(self.config.actor_rollout_ref.actor.strategy),
            "checkpoint_origin": checkpoint_origin,
            "model_repository": self.config.actor_rollout_ref.model.get("repository")
            or os.getenv("HELICOPTER_MODEL_REPOSITORY", "untracked"),
            "model_revision": self.config.actor_rollout_ref.model.get("revision")
            or os.getenv("HELICOPTER_MODEL_REVISION", "untracked"),
            "model_filename": self.config.actor_rollout_ref.model.get("filename")
            or os.getenv("HELICOPTER_MODEL_FILENAME", Path(model_path).name),
            "model_path": model_path,
            "rollout_engine": str(self.config.actor_rollout_ref.rollout.name),
            "floating_dtype": "bfloat16",
            "state_dict_keys": "native",
        }
        initial_version = checkpoint_policy_version(self.global_steps)
        initial_identity = BehaviorPolicyIdentity.for_publication(
            policy_version=initial_version,
            previous_weight_digest=f"checkpoint:{checkpoint_origin}:{initial_version}",
            export_contract=self._export_contract,
            sampling_config=self._sampling_config,
            runtime_identity=self._runtime_identity,
        )
        publication_state_path = os.getenv(POLICY_PUBLICATION_STATE_ENV)
        restored_round = None
        if publication_state_path and Path(publication_state_path).is_file():
            restored_round = load_committed_policy_round(publication_state_path)
            restored = restored_round.published
            if restored is None or restored.policy_version != initial_version:
                restored_version = None if restored is None else restored.policy_version
                raise RuntimeError(
                    "committed policy publication version does not match the resumed checkpoint: "
                    f"state={restored_version} checkpoint={initial_version}"
                )
            if restored.sampling_config_digest != initial_identity.sampling_config_digest:
                raise RuntimeError("committed policy publication sampling contract changed across resume")
            initial_identity = BehaviorPolicyIdentity(
                policy_version=restored.policy_version,
                weight_digest=restored.weight_digest,
                sampling_config_digest=restored.sampling_config_digest,
                runtime_identity=self._runtime_identity,
            )
        acknowledgements = self.checkpoint_manager.update_weights(
            self.global_steps,
            policy_identity=initial_identity.as_dict(),
        )
        self.policy_round.publish_initial(
            initial_identity,
            acknowledgements,
            source_policy_version=initial_version,
            source_policy_lineage=(
                initial_identity.weight_digest
                if restored_round is not None
                else f"checkpoint:{checkpoint_origin}:{initial_version}"
            ),
        )
        event = "publish_resume" if restored_round is not None else "publish_initial"
        record_policy_identity(event, self.global_steps, initial_identity)
        if publication_state_path:
            save_committed_policy_round(publication_state_path, self.policy_round)

    def on_step_begin(self):
        self.policy_round.begin_rollout(training_policy_version(self.global_steps))

    def get_rollout_metadata(self):
        if self.policy_round.published is None:
            return {}
        return self.policy_round.published.as_dict()

    def on_batch_prepared(self, batch):
        self.policy_round.prepare(batch.tags)
        self.policy_round.begin_training()
        effective_sampling_digests = {tag["request_sampling_digest"] for tag in batch.tags}
        if len(effective_sampling_digests) != 1:
            self.policy_round.fail("training batch has mixed effective sampling digests")
        record_policy_identity(
            "train_begin",
            self.global_steps,
            self.policy_round.training_identity,
            effective_sampling_digest=effective_sampling_digests.pop(),
        )

    def _update_actor(self, batch, metrics):
        batch.extra_info["strict_on_policy"] = True
        return super()._update_actor(batch, metrics)

    def on_step_end(self):
        with marked_timer("weight_publish", self.timing_raw, color="red"):
            current_identity = self.policy_round.training_identity
            if current_identity is None:
                self.policy_round.fail("training completed without an active behavior-policy identity")
            next_identity = BehaviorPolicyIdentity.for_publication(
                policy_version=self.global_steps,
                previous_weight_digest=current_identity.weight_digest,
                export_contract=self._export_contract,
                sampling_config=self._sampling_config,
                runtime_identity=self._runtime_identity,
            )
            self.policy_round.begin_publication(
                next_identity,
                source_policy_version=current_identity.policy_version,
                source_policy_lineage=current_identity.weight_digest,
            )
            try:
                acknowledgements = self.checkpoint_manager.update_weights(
                    self.global_steps,
                    policy_identity=next_identity.as_dict(),
                )
            except BaseException as exc:
                self.policy_round.rollback_publication(f"weight publication failed: {exc}")
                raise
            for stage in ("rollout_weights_resume", "weight_transfer", "rollout_kv_wake"):
                stage_values = [
                    acknowledgement.get("publication_timing", {}).get(stage)
                    for acknowledgement in acknowledgements
                    if acknowledgement is not None
                ]
                stage_values = [value for value in stage_values if value is not None]
                if stage_values:
                    self.timing_raw[stage] = max(stage_values)
            self.policy_round.commit_publication(next_identity, acknowledgements)
            record_policy_identity("publish", self.global_steps, next_identity)
            publication_state_path = os.getenv(POLICY_PUBLICATION_STATE_ENV)
            if publication_state_path:
                save_committed_policy_round(publication_state_path, self.policy_round)

    def on_sample_end(self):
        # sleep all replicas to discard weights and kv cache
        with marked_timer("rollout_sleep", self.timing_raw, color="purple"):
            self.checkpoint_manager.sleep_replicas()
