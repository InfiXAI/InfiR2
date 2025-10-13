#!/bin/bash

NNODES=1
RDZV_ENDPOINT="localhost:29400"

# 创建一个数组存储其他参数
ARGS=()

# 解析所有参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --rdzv_endpoint)
            RDZV_ENDPOINT="$2"
            shift 2
            ;;
        *)
            # 将其他参数添加到数组中
            ARGS+=("$1")
            shift
            ;;
    esac
done

CODE_DIR=/your_megatron_path
export PYTHONPATH=$PYTHONPATH:$CODE_DIR
export WANDB_MODE=offline


# 1.Select the megatron-lm code base and set the log name
EXP="InfiR2_CPT_FP8_7B"
DIR=`pwd`
OUT_DIR=${DIR}/exp/$EXP
TENSORBOARD_DIR=${DIR}/exp/$EXP/tensorboard
LOG_DIR=${DIR}/exp/$EXP/logs
mkdir -p $OUT_DIR
mkdir -p $TENSORBOARD_DIR
mkdir -p $LOG_DIR
echo "Current megatron code base is: $CODE_DIR"
echo "Output dir: $OUT_DIR"

# 2.Set quantization config
FP8_RECIPE=blockwise

# 3.Set the training env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export NVTE_DEBUG=0


# 4.Set the dataset, tokenizer and checkpoint path and config
DATA_PATH="/your_data_path"
DATA_CACHE_PATH="/your_cache_path"
TOKENIZER_MODEL="/your_model_path/Qwen2.5-7B"
CHECKPOINT_PATH=$OUT_DIR/checkpoints

DATA_ARGS=" \
    --data-path ${DATA_PATH} \
    --data-cache-path ${DATA_CACHE_PATH} \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --split 90,5,5 \
    --num-workers 6 \
    --no-create-attention-mask-in-dataloader
"

# 5.Set the distributed arguments env and config
GPUS_PER_NODE=8
DISTRIBUTED_ARGS="\
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_endpoint=$RDZV_ENDPOINT \
    --rdzv_backend=c10d \
"

# 6.Set the parallel startegy env and config
GBS=128
MBS=1
TP_SIZE=4
PP_SIZE=1
CP_SIZE=2


MODEL_PARALLEL_ARGS=" \
    --distributed-timeout-minutes 120 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --context-parallel-size ${CP_SIZE} \
    --use-distributed-optimizer \
    --sequence-parallel \
    --overlap-grad-reduce \
    --overlap-param-gather
"

# 7.Set the model env and config
SEQ_LENGTH=32768

MODEL_ARGS=" \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --use-flash-attn \
    --use-dist-ckpt \
    --no-load-optim \
    --no-load-rng \
    --disable-bias-linear \
    --add-qkv-bias \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --use-rotary-position-embeddings \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --rotary-seq-len-interpolation-factor 1 \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --seq-length ${SEQ_LENGTH} \
    --num-layers 28 \
    --hidden-size 3584 \
    --ffn-hidden-size 18944 \
    --num-attention-heads 28 \
    --group-query-attention \
    --num-query-groups 4 \
    --max-position-embeddings ${SEQ_LENGTH} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --initial-loss-scale 262144 \
    --vocab-size 152064 \
    --untie-embeddings-and-output-weights
"

# 8. Set the training time and log info
MAX_STEPS=40000 #96875 #1000000
WARMUP_STEPS=5000
WSD_DECAY_STEPS=5000


EVAL_AND_LOGGING_ARGS=" \
    --train-iters ${MAX_STEPS} \
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 100 \
    --eval-iters 10 \
    --log-throughput \
    --log-timers-to-tensorboard \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --tensorboard-dir $TENSORBOARD_DIR \
    --wandb-project "qwen2.5_7b" \
    --wandb-exp-name $EXP \
    --wandb-save-dir $LOG_DIR
"

# 9. Set the optimizer
SEED=1236
TRAINING_ARGS=" \
    --seed ${SEED} \
    --lr 1e-4 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0 \
    --lr-decay-style WSD \
    --lr-warmup-iters $WARMUP_STEPS \
    --lr-wsd-decay-style cosine \
    --lr-wsd-decay-iters $WSD_DECAY_STEPS \
    --min-lr 1e-5 \
    --init-method-std 0.008
"
# 10. Set the training precision
PRECISION_ARGS=" \
    --bf16 \
    --fp8-format e4m3 \
    --fp8-recipe ${FP8_RECIPE} \
    --fp8-param-gather \
    --use-precision-aware-optimizer
"
    # --main-params-dtype fp32 \
    # --exp-avg-dtype bf16 \
    # --exp-avg-sq-dtype bf16

# 11. Set the training profile
PROFILER_ARGS=" \
    --profile \
    --use-pytorch-profiler \
    --record-memory-history
"

# 12. launch the training task with torchrun
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
EXE=${CODE_DIR}/pretrain_gpt.py

torchrun $DISTRIBUTED_ARGS ${EXE} \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MODEL_PARALLEL_ARGS \
    $MODEL_ARGS \
    $EVAL_AND_LOGGING_ARGS \
    $TRAINING_ARGS \
    $PRECISION_ARGS \
    $PROFILER_ARGS \
    --save $CHECKPOINT_PATH \
    --load ${CHECKPOINT_PATH} \
    "${ARGS[@]}"
# set +x

