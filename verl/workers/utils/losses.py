# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def antidoom_actor_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute Antidoom FTPO from the final-token logits produced by the FSDP actor."""

    del dp_group
    required = ("chosen_token_ids", "rejected_token_ids")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("Antidoom FTPO batch is missing: " + ",".join(missing))
    logits = model_output.get("antidoom_logits")
    if logits is None:
        raise ValueError("Antidoom FTPO requires dense final-token logits from the actor engine")
    reference_logits = data.get("reference_logits")
    if reference_logits is None and (
        config.policy_loss.ftpo_lambda_mse > 0 or config.policy_loss.ftpo_lambda_mse_target > 0
    ):
        raise ValueError("Antidoom FTPO reference_logits are required when an MSE coefficient is positive")
    loss, values = _antidoom_ftpo_objective(
        logits,
        chosen_token_ids=data["chosen_token_ids"],
        rejected_token_ids=data["rejected_token_ids"],
        chosen_mask=data.get("chosen_mask"),
        reference_logits=reference_logits,
        clip_epsilon=config.policy_loss.ftpo_clip_epsilon,
        lambda_mse=config.policy_loss.ftpo_lambda_mse,
        lambda_mse_target=config.policy_loss.ftpo_lambda_mse_target,
        target_tolerance=config.policy_loss.ftpo_target_tolerance,
    )
    metrics = Metric.from_dict(
        {f"actor/ftpo_{name}": value for name, value in values.items()},
        aggregation=AggregationType.MEAN,
    )
    return loss, metrics


def _antidoom_ftpo_objective(
    logits: torch.Tensor,
    *,
    chosen_token_ids: torch.Tensor,
    rejected_token_ids: torch.Tensor,
    chosen_mask: torch.Tensor | None,
    reference_logits: torch.Tensor | None,
    clip_epsilon: float,
    lambda_mse: float,
    lambda_mse_target: float,
    target_tolerance: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.ndim != 2:
        raise ValueError("Antidoom FTPO logits must have shape [batch, vocabulary]")
    if chosen_token_ids.ndim == 1:
        chosen_token_ids = chosen_token_ids.unsqueeze(-1)
    if chosen_mask is None:
        chosen_mask = torch.ones_like(chosen_token_ids, dtype=torch.bool)
    else:
        chosen_mask = chosen_mask.to(dtype=torch.bool)
    if chosen_token_ids.shape != chosen_mask.shape or rejected_token_ids.shape != logits.shape[:1]:
        raise ValueError("Antidoom FTPO token ids and masks do not match the logits batch")
    if not torch.any(chosen_mask):
        raise ValueError("Antidoom FTPO requires at least one active chosen token")
    if clip_epsilon <= 0:
        raise ValueError("Antidoom FTPO clip_epsilon must be positive")

    row_ids = torch.arange(logits.size(0), device=logits.device).unsqueeze(1)
    rejected_logits = logits.gather(-1, rejected_token_ids.unsqueeze(-1))
    deltas = logits[row_ids, chosen_token_ids] - rejected_logits
    active_weights = torch.clamp((clip_epsilon - deltas) / clip_epsilon, min=0.0, max=1.0) * chosen_mask
    chosen_counts = chosen_mask.sum(dim=-1).clamp(min=1)
    preference_loss = ((F.softplus(clip_epsilon - deltas) * active_weights).sum(dim=-1) / chosen_counts).mean()
    loss = preference_loss

    mse_other = logits.new_tensor(0.0)
    mse_target = logits.new_tensor(0.0)
    if reference_logits is not None:
        if reference_logits.shape != logits.shape:
            raise ValueError("Antidoom FTPO reference_logits must match logits")
        difference = logits - reference_logits.detach()
        target_mask = torch.zeros_like(logits, dtype=torch.bool)
        target_mask[row_ids.expand_as(chosen_token_ids)[chosen_mask], chosen_token_ids[chosen_mask]] = True
        target_mask.scatter_(1, rejected_token_ids.unsqueeze(-1), True)
        other_mask = ~target_mask
        mse_other = (difference.square() * other_mask).sum() / other_mask.sum().clamp(min=1)
        excess = torch.clamp(difference.abs() - target_tolerance, min=0.0)
        mse_target = (excess.square() * target_mask).sum() / target_mask.sum().clamp(min=1)
        loss = loss + lambda_mse * mse_other + lambda_mse_target * mse_target

    active_deltas = deltas[chosen_mask]
    log_probabilities = F.log_softmax(logits, dim=-1)
    chosen_log_probabilities = log_probabilities[row_ids, chosen_token_ids]
    rejected_log_probabilities = log_probabilities.gather(-1, rejected_token_ids.unsqueeze(-1))
    wins = (chosen_log_probabilities > rejected_log_probabilities) & chosen_mask
    return loss, {
        "pref_loss": preference_loss.detach(),
        "chosen_win": (wins.sum(dim=-1) / chosen_counts).float().mean().detach(),
        "margin_win": ((deltas >= clip_epsilon) & chosen_mask).sum().float().div(chosen_mask.sum()).detach(),
        "mean_delta": active_deltas.mean().detach(),
        "median_delta": active_deltas.median().detach(),
        "active_weight": active_weights[chosen_mask].mean().detach(),
        "mse_elem": mse_other.detach(),
        "mse_tgt_tokenwise": mse_target.detach(),
    }


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # Normalize the value loss over the global mini-batch (dp_size / batch_num_tokens /
    # global_batch_size) instead of the local micro-batch, so the accumulated critic gradient is
    # invariant to how the mini-batch is split into micro-batches (as the actor's ppo_loss does).
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]
    global_batch_size = data["global_batch_size"]

    # When the loss is normalized over the global batch, each micro-batch contributes a partial sum,
    # so the loss metric must be aggregated with SUM to reflect the global-batch mean.
    if (
        dp_size > 1
        or batch_num_tokens is not None
        or global_batch_size is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
        loss_scale_factor=config.loss_scale_factor,
    )

    metrics = {
        "critic/vf_loss": Metric(value=vf_loss, aggregation=metric_aggregation),
        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
        "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
    }

    return vf_loss, metrics
