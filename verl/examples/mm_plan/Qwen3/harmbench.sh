#!/usr/bin/env bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate mm_plan

# ------------------------------------------------------------
# MM-Plan Training on HarmBench Multimodal Subset
# ------------------------------------------------------------

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)
ENGINE="vllm"

TASK="HarmBench"
ADVANTAGE="grpo"
TARGET_MODEL="Qwen3-VL-8B"

# Hyperparameters matching the paper
LR=1e-5
N_ROLLOUT=4
BATCH_SIZE=16

export TARGET_MODEL="${TARGET_MODEL}"
export JUDGE_MODEL="Claude-Sonnet-4.5"

export PROJECT_NAME="MM-Plan-verl"
export EXP_NAME="${TASK}-${TARGET_MODEL}-${ADVANTAGE}"
export VERL_FILE_LOGGER_ROOT="./logs"

OUTPUT_DIR="checkpoints/${PROJECT_NAME}/${EXP_NAME}/${DATE}/N${N_ROLLOUT}-BS${BATCH_SIZE}-LR${LR}-${TIME_TAG}"

# ------------------------------------------------------------

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=$ADVANTAGE \
  data.train_files=data/HarmBench/train.parquet \
  data.val_files=data/HarmBench/val.parquet \
  data.train_batch_size="${BATCH_SIZE}" \
  data.max_prompt_length=4096 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.image_key=images \
  actor_rollout_ref.model.path=Qwen/Qwen3-VL-4B-Instruct \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.freeze_vision_tower=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.n="${N_ROLLOUT}" \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb","file"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=1 \
  trainer.test_freq=4 \
  trainer.total_epochs=10 \
  trainer.rollout_data_dir="logs_rollout/${PROJECT_NAME}/${EXP_NAME}" \
  trainer.default_local_dir=$OUTPUT_DIR \
  custom_reward_function.path="./verl/utils/reward_score/mm_plan/__init__.py" \
  custom_reward_function.name=compute_score

echo ""
echo "========================================================================"
echo "Training completed for ${TARGET_MODEL} on HarmBench"
echo "========================================================================"
