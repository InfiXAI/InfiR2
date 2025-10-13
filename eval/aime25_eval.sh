#!/bin/bash
#SBATCH --job-name=evalscope_aime25
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail

CONTAINER_SQSH="/path/to/your/.sqsh"
ENROOT_NAME="evalscopesg_${SLURM_JOB_ID:-$$}"

# 挂载点
MOUNT1="/path/to/your/workspace:/path/to/your/workspace"
#MOUNT2=""
#MOUNT3=""
#MOUNT4=""

# Python 虚拟环境与工作目录
VENV="/path/to/miniconda3/etc/profile.d/"
# ！需要修改成自己的
WORKDIR="/path/to/your/workspace"

# SGLang 服务参数
# ！需要修改成自己的
MODEL_PATH="/path/to/eval_model_path"
SERVED_NAME="eval_model_name"
# 动态分配端口，避免冲突
BASE_PORT=8860
PORT=$((BASE_PORT + (SLURM_JOB_ID % 100) + 1))
#PORT="8801"
TP_SIZE="4"
WARMUPS="3"
MAX_RUNNING_REQUESTS="16"
CHUNKED_PREFILL_SIZE="4096"

HF_DATASETS_CACHE="/path/to/datasets"

EVAL_WORK_DIR="/path/to/sglang/"
DATASET_LOCAL="/path/to/datasets/AIME2025"
EVAL_DATASET="aime25"
EVAL_BATCH=32
EVAL_TIMEOUT=18000
MAX_NEW_TOKENS=31000
GENERATED_NUM=32
TEMPERATURE=0.65

JOB_ID="${SLURM_JOB_ID:-$$}"  # Use SLURM_JOB_ID if available, otherwise use process ID
SGLANG_LOG="${WORKDIR}/sglang_${JOB_ID}.log"
EVAL_LOG="${WORKDIR}/eval_${JOB_ID}.log"

echo "[INFO] Creating unique enroot image: ${ENROOT_NAME}"
enroot create --name "${ENROOT_NAME}" "${CONTAINER_SQSH}"

cleanup() {
    echo "[INFO] Cleaning up enroot image: ${ENROOT_NAME}"
    enroot remove "${ENROOT_NAME}" 2>/dev/null || true
}
trap cleanup EXIT

enroot start --rw \
  --mount "${MOUNT1}" \
  --mount "${MOUNT2}" \
  --mount "${MOUNT3}" \
  --mount "${MOUNT4}" \
  "${ENROOT_NAME}" bash -lc "
set -euo pipefail

echo '[INFO] Switching to sglang workspace: /sgl-workspace/sglang'
cd '/sgl-workspace/sglang'

echo '[INFO] Starting SGLang server on port ${PORT} ...'
nohup env python3 -m sglang.launch_server \
  --model '${MODEL_PATH}' \
  --served-model-name '${SERVED_NAME}' \
  --port '${PORT}' \
  --trust-remote-code \
  --tensor-parallel-size '${TP_SIZE}' \
  --warmups '${WARMUPS}' \
  --max-running-requests '${MAX_RUNNING_REQUESTS}' \
  --chunked-prefill-size '${CHUNKED_PREFILL_SIZE}' \
  > '${SGLANG_LOG}' 2>&1 &

SERVER_PID=\$!
echo \"[INFO] SGLang PID=\$SERVER_PID, log -> ${SGLANG_LOG}\"

# 激活conda环境
echo '[INFO] Activating venv: ${VENV}'
source '${VENV}/conda.sh'
conda activate '/llm-eval/evalscope'

# 环境变量
export HF_DATASETS_CACHE='${HF_DATASETS_CACHE}'

echo '[INFO] Waiting for SGLang server to be ready ...'
for i in \$(seq 1 180); do
  if (echo > /dev/tcp/127.0.0.1/${PORT}) >/dev/null 2>&1; then
    echo '[INFO] SGLang server is ready.'
    break
  fi
  sleep 2
  if ! kill -0 \"\$SERVER_PID\" 2>/dev/null; then
    echo '[ERROR] SGLang server exited unexpectedly. See ${SGLANG_LOG}'
    exit 1
  fi
  if [ \"\$i\" -eq 180 ]; then
    echo '[ERROR] SGLang server did not come up in time.'
    exit 1
  fi
done

echo '[INFO] Running evalscope ...'
evalscope eval \
  --model '${SERVED_NAME}' \
  --generation-config '{\"do_sample\": true, \"temperature\": '${TEMPERATURE}', \"top_p\": 0.95, \"max_tokens\": '${MAX_NEW_TOKENS}', \"n\": '${GENERATED_NUM}'}' \
  --api-url 'http://localhost:${PORT}/v1/chat/completions' \
  --api-key EMPTY \
  --eval-type service \
  --work-dir '${EVAL_WORK_DIR}' \
  --datasets '${EVAL_DATASET}' \
  --dataset-args '{\"aime25\": {\"local_path\": \"${DATASET_LOCAL}\", \"filters\": {\"remove_until\": \"</think>\"}, \"prompt_template\": \"<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\", \"system_prompt\": \"<|im_start|>system\\nPlease reason step by step, and put your final answer within \\\\boxed{{}}.<|im_end|>\\n\"}}' \
  --eval-batch-size ${EVAL_BATCH} \
  --timeout ${EVAL_TIMEOUT} | tee '${EVAL_LOG}'

STATUS=\$?

echo '[INFO] Stopping SGLang server (PID='\$SERVER_PID') ...'
kill \"\$SERVER_PID\" 2>/dev/null || true
wait \"\$SERVER_PID\" 2>/dev/null || true

exit \"\$STATUS\"
"

echo "[INFO] Done. Check logs:"
echo "  SGLang: ${SGLANG_LOG}"
echo "  EVAL: ${EVAL_LOG}"