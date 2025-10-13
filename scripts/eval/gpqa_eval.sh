#!/usr/bin/env bash
set -euo pipefail

for cmd in python3 evalscope; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "[ERROR] $cmd is required but was not found in PATH." >&2
    exit 1
  }
done

: "${MODEL_PATH:?Set MODEL_PATH to the weights directory or checkpoint.}"
SERVED_NAME=${SERVED_NAME:-$(basename "$MODEL_PATH")}

API_PORT=${API_PORT:-8801}
TP_SIZE=${TP_SIZE:-1}
WARMUPS=${WARMUPS:-1}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-8}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-4096}
SERVER_READY_WAIT=${SERVER_READY_WAIT:-5}

EVAL_BATCH=${EVAL_BATCH:-32}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-18000}
EVAL_DATASET=${EVAL_DATASET:-gpqa}
EVAL_WORK_DIR=${EVAL_WORK_DIR:-$(pwd)}
API_KEY=${API_KEY:-EMPTY}

if [[ -z "${DATASET_ARGS:-}" ]]; then
  if [[ -n "${DATASET_LOCAL:-}" ]]; then
    DATASET_ARGS=$(printf '{"gpqa": {"local_path": "%s", "filters": {"remove_until": "</think>"}, "subset_list": ["gpqa_diamond"]}}' "$DATASET_LOCAL")
  else
    DATASET_ARGS='{"gpqa": {"filters": {"remove_until": "</think>"}, "subset_list": ["gpqa_diamond"]}}'
  fi
fi

if [[ -z "${GENERATION_CONFIG:-}" ]]; then
  GENERATION_CONFIG=$(printf '{"do_sample": true, "temperature": %s, "top_p": %s, "max_tokens": %s, "n": %s}' \
    "${TEMPERATURE:-0.65}" \
    "${TOP_P:-0.95}" \
    "${MAX_NEW_TOKENS:-26000}" \
    "${GENERATED_NUM:-8}")
fi

python3 -m sglang.launch_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --port "${API_PORT}" \
  --trust-remote-code \
  --tensor-parallel-size "${TP_SIZE}" \
  --warmups "${WARMUPS}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" &
SERVER_PID=$!

sleep "${SERVER_READY_WAIT}"

evalscope eval \
  --model "${SERVED_NAME}" \
  --generation-config "${GENERATION_CONFIG}" \
  --api-url "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
  --api-key "${API_KEY}" \
  --eval-type service \
  --work-dir "${EVAL_WORK_DIR}" \
  --datasets "${EVAL_DATASET}" \
  --dataset-args "${DATASET_ARGS}" \
  --eval-batch-size "${EVAL_BATCH}" \
  --timeout "${EVAL_TIMEOUT}"

kill "$SERVER_PID" >/dev/null 2>&1 || true
wait "$SERVER_PID" >/dev/null 2>&1 || true
