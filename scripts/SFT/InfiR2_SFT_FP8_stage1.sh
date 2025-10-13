#!/bin/bash

export PYTHONBUFFERED=16

# ================== model config =============================
SCRIPT_DIR=/path/to/slime/scripts
source "${SCRIPT_DIR}/models/qwen2.5-7B.sh"
# =============================================================

# ================= user config ===============================
HOME_DIR=/path/to/slime
LOG_DIR=/path/to/wandb_log_dir/
LOAD_DIR=/path/to/load_dir_InfiR2_SFT_FP8_stg1/
SAVE_DIR=/path/to/save_dir_InfiR2_SFT_FP8_stg1/
DATA_DIR=/path/to/stage1_data
HF_CHECKPOINT=/path/to/base_models_hf/qwen2.5-7B/
REF_LOAD=/path/to/base_models_/qwen2.5-7B_torch_dist/
# ==============================================================

# ================ paralle config ==============================
TP=4
PP=1
CP=1
EP_MP=1
EP_TP=1
MAX_TOKENS_PER_GPU=32768
# ==============================================================

CKPT_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
   --ref-load ${REF_LOAD}
   --load ${LOAD_DIR}
   --save ${SAVE_DIR}
   --save-interval 1000
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data ${DATA_DIR}
   --input-key messages
   --rollout-shuffle
   --num-epoch 5
   --rollout-batch-size 64
   --global-batch-size 64

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
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

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-5
   --lr-decay-style cosine
   --min-lr 2e-6
   --lr-warmup-fraction 0.05
   --weight-decay 0
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=(
   --use-wandb
   --wandb-mode offline
   --wandb-project slime-dev
   --wandb-group InfiR2-sft-fp8-stg1
   --wandb-dir ${LOG_DIR}/InfiR2-sft-fp8-stg1__wandb
   # --wandb-key ${WANDB_KEY}
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

PERCISE_ARGS=(
   --transformer-impl transformer_engine
   --bf16
   --fp8-format e4m3
   --fp8-recipe blockwise
   --fp8-param-gather

)

TENSORBOARD_ARGS=(
   --use-pytorch-profiler
   --profile-step-start 16
   --profile-step-end 18
   --tensorboard-dir ${LOG_DIR}/tensorboard/InfiR2-sft-fp8-stg1
   --record-memory-history
)

# launch the master node of ray in container
# export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# export no_proxy="127.0.0.1,${MASTER_ADDR}"
# ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
export http_proxy=""
export https_proxy=""

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
        "working_dir": "/root/slime",
        "env_vars": {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1",
            "NVTE_DEBUG": "0",
            "HOME": "/root/slime",
            "http_proxy": "",
            "https_proxy": "",
            "NCCL_SOCKET_IFNAME": "bond0"
        }
    }' \
   -- python3 train_async.py \
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${PERCISE_ARGS[@]} \
   ${TENSORBOARD_ARGS[@]} \
   ${MISC_ARGS[@]} \
   $@