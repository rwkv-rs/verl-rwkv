#!/usr/bin/env bash
set -xeuo pipefail

RWKV_MODEL_PATH=${RWKV_MODEL_PATH:-/workspace/Weights/RWKV/rwkv7-g1f-1.5b-20260419-ctx8192.pth}
RWKV_LM_PATH=${RWKV_LM_PATH:-/workspace/Projects/MachineLearning/rwkv-lm}
VLLM_RWKV_PATH=${VLLM_RWKV_PATH:-}
PYTHON=${PYTHON:-.venv/bin/python}

DATA_ROOT=${DATA_ROOT:-/workspace/Datasets/gsm8k}
TRAIN_FILES=${TRAIN_FILES:-"['${DATA_ROOT}/train.parquet']"}
VAL_FILES=${VAL_FILES:-"['${DATA_ROOT}/test.parquet']"}

[[ -x "${PYTHON}" ]] || { echo "Python executable not found: ${PYTHON}. Run uv sync or set PYTHON."; exit 1; }
[[ -f "${RWKV_MODEL_PATH}" ]] || { echo "RWKV checkpoint not found: ${RWKV_MODEL_PATH}"; exit 1; }
[[ -d "${RWKV_LM_PATH}" ]] || { echo "rwkv-lm repository not found: ${RWKV_LM_PATH}"; exit 1; }

if [[ -n "${VLLM_RWKV_PATH}" ]]; then
    [[ -d "${VLLM_RWKV_PATH}" ]] || { echo "vLLM-RWKV repository not found: ${VLLM_RWKV_PATH}"; exit 1; }
    export PYTHONPATH="${VLLM_RWKV_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
fi

export VLLM_RWKV7_WKV_MODE=${VLLM_RWKV7_WKV_MODE:-fp32io16}
export VLLM_RWKV7_EMB_DEVICE=${VLLM_RWKV7_EMB_DEVICE:-cpu}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-56}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-56}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
RWKV_USE_DYNAMIC_BSZ=${RWKV_USE_DYNAMIC_BSZ:-False}

ACTOR_LR=${ACTOR_LR:-1e-5}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.85}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-2048}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:--1}

PROJECT_NAME=${PROJECT_NAME:-verl_rwkv_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-rwkv7_grpo_vllm}

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    reward.custom_reward_function.path="${PWD}/examples/rwkv_trainer/math_verify_reward.py"
    reward.custom_reward_function.name=compute_score
)

MODEL=(
    model@actor_rollout_ref.model=rwkv_native
    actor_rollout_ref.model.path="${RWKV_MODEL_PATH}"
    actor_rollout_ref.model.rwkv_lm_path="${RWKV_LM_PATH}"
)

ACTOR=(
    actor@actor_rollout_ref.actor=rwkv_lm
    actor_rollout_ref.actor.engine.rwkv_lm_path="${RWKV_LM_PATH}"
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=${RWKV_USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
)

REF=(
    ref@actor_rollout_ref.ref=rwkv_lm
    actor_rollout_ref.ref.engine.rwkv_lm_path="${RWKV_LM_PATH}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${RWKV_USE_DYNAMIC_BSZ}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.load_format=auto
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${RWKV_USE_DYNAMIC_BSZ}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tokenizer_mode=rwkv
)

TRAINER=(
    critic.enable=False
    trainer.logger='["console"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.val_before_train=False
    trainer.total_epochs=${TOTAL_EPOCHS}
)

"${PYTHON}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@"
