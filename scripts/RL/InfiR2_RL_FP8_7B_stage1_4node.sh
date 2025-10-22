#!/bin/bash

export PYTHONBUFFERED=16

# ================== model config =============================
SCRIPT_DIR=/path/to/slime/scripts
source "${SCRIPT_DIR}/models/qwen2.5-7B.sh"
# =============================================================

# ================= user config ===============================
HOME_DIR=/path/to/slime
LOG_DIR=/path/to/wandb_log_dir/
LOAD_DIR=/path/to/load_dir_stg1_RL8k/
SAVE_DIR=/path/to/save_dir_stg1_RL8k/
DATA_DIR=/path/to/data/dapo-math-17k.jsonl
HF_CHECKPOINT=/path/to/models_FP8_config_hf/your_model/
REF_LOAD=/path/to/models_torch_dist/your_model/
# ==============================================================

# ================ paralle config ==============================
TP=4
PP=1
CP=2
EP_MP=1
EP_TP=1
MAX_TOKENS_PER_GPU=8192
# ==============================================================

# ================ RL specific config =========================
train_prompt_bsz=128
gen_prompt_bsz=$((train_prompt_bsz * 3))

NUM_ROLLOUT=10240
N_SAMPLES_PER_PROMPT=16
GLOBAL_BATCH_SIZE=$((train_prompt_bsz * N_SAMPLES_PER_PROMPT))
ROLLOUT_MAX_RESPONSE_LEN=8192
ROLLOUT_TEMPERATURE=1.0
OVER_SAMPLING_BATCH_SIZE=${gen_prompt_bsz}
# ==============================================================

CKPT_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
   --ref-load ${REF_LOAD}
   --load ${LOAD_DIR}
   --save ${SAVE_DIR}
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle  # data shufffle
   --rm-type deepscaler
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${train_prompt_bsz}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
   --rollout-temperature ${ROLLOUT_TEMPERATURE}
   --over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE}  # ${gen_prompt_bsz}
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data

   # Response length penalty configuration
#    --enable-length-penalty
#    --max-response-length 8192
#    --length-penalty-buffer 1024
#    --length-penalty-factor 1.0
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /path/to/data/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 30000
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP}
   --context-parallel-size ${CP}
   --expert-model-parallel-size ${EP_MP}
   --expert-tensor-parallel-size ${EP_TP}

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group InfiR2-RL-fp8-stg1-8k
   --wandb-mode offline
   --wandb-dir ${LOG_DIR}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.5
   --sglang-max-running-requests 128
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

PRECISE_ARGS=(
   --transformer-impl transformer_engine
   --bf16
   # for fp8 training
   --fp8-format e4m3
   --fp8-recipe blockwise
   --fp8-param-gather
   # --direct-update-fp8-weight
)

TENSORBOARD_ARGS=(
   --profile-step-start 10
   --profile-step-end 12
   --tensorboard-dir ${LOG_DIR}/tensorboard
   --record-memory-history
)

# launch the master node of ray in container
export http_proxy=""
export https_proxy=""

# Build the runtime environment JSON with proper variable substitution
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
        "working_dir": "/root/slime",
        "env_vars": {
            "PYTHONPATH": "/root/Megatron-LM/:/root/slime",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:2048",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1",
            "NVTE_DEBUG": "0",
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
            "HOME": "/root/slime",
            "http_proxy": "",
            "https_proxy": "",
            "NCCL_SOCKET_IFNAME": "bond0",
            "WANDB_MODE": "offline",
            "WANDB_DIR": "/path/to/wandb_log/",
            "RAY_DEDUP_LOGS_ALLOW_REGEX": "Username",
            "NO_PROXY": "localhost,127.0.0.1,klb-dgx-*",
            "no_proxy": "localhost,127.0.0.1,klb-dgx-*"
        }
    }' \
   -- python3 train.py \
   --actor-num-nodes 4 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${PRECISE_ARGS[@]} \
   ${TENSORBOARD_ARGS[@]} \
   $@