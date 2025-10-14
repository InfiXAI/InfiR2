## 🤖 SFT & RL with FP8

FP8 SFT on Qwen2.5-Math-1.5B/7B matches BF16 losses and stays within 1–2 points on AIME24, AIME25, GPQA, LiveCodeBench v5. The InfiAlign-SFT-72k → 165k curriculum training strategy yields up to 22% faster training, 14% lower peak memory, 19% higher throughput. Everything below explains how to reproduce those results with the released scripts.

---

## 🧠 Supervised Fine-tuning

### Data & Environment
Stage 1 uses InfiAlign-SFT-72k, Stage 2 switches to InfiAlign-SFT-165k (maintaining 32k context). Provide HuggingFace and Megatron checkpoints, and propagate env vars such as NCCL sockets, proxy settings. Logging can run offline via W&B + TensorBoard.

```bash
# Stage configuration template
HOME_DIR=/path/to/slime
LOG_DIR=/path/to/logs
DATA_DIR=/path/to/InfiAlign-SFT-72k           # swap to -165k for Stage 2
HF_CHECKPOINT=/path/to/base_models_hf/qwen2.5-7B-Instruct/
REF_LOAD=/path/to/base_models_/InfiR2_CPT_FP8_7B_torch_dist/
LOAD_DIR=/path/to/checkpoints/InfiR2_SFT_FP8_7B_stg1/
SAVE_DIR=/path/to/outputs/InfiR2_SFT_FP8_7B_stg1/
```

### Training Recipe
Scripts live in `scripts/SFT/`. Key flags stay identical across stages: `--num-epoch 5`, `--global-batch-size 64`, `--rollout-batch-size 64`, `TP=4`, `PP=1`, `CP=1`, `--max-tokens-per-gpu 32768`, dynamic micro-batching. Optimizer mirrors the paper (Adam, lr=2e-5, cosine w/ 5% warmup, β1=0.9, β2=0.95, no weight decay). Dropout disabled; FP32 accumulation and flash attention enabled. Precision block combines `--bf16`, `--fp8-format e4m3`, `--fp8-recipe blockwise`, `--fp8-param-gather`. Stage 2 simply swaps `DATA_DIR` and `REF_LOAD` to continue from Stage 1 checkpoints.


---

## 🎯 Reinforcement Learning

### Checkpoint Conversion
RL rollout engines expect FP8 E8M0 inference. After SFT Stage 2, convert the torch checkpoint to HuggingFace, then cast to FP8.

```bash
PYTHONPATH=training/Megatron-LM:training/slime \
python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/to/InfiR2_SFT_FP8_stg2 \
  --output-dir /path/to/InfiR2_SFT_FP8_stg2_hf \
  --origin-hf-dir /path/to/models/Qwen2.5-7B

python tools/bf16_cast_fp8.py \
  --input-bf16-hf-path /path/to/InfiR2_SFT_FP8_stg2_hf \
  --output-fp8-hf-path /path/to/InfiR2_SFT_FP8_stg2_hf_e8m0 \
  --force-pow-2-scale True

HF_CHECKPOINT=/path/to/InfiR2_SFT_FP8_stg2_hf_e8m0/
REF_LOAD=/path/to/InfiR2_SFT_FP8_stg2/
DATA_DIR=/path/to/datasets/dapo-math-17k.jsonl
```

### Curriculum Setup
RL scripts (`scripts/RL/`) target nodes × eight GPUs with `--colocate`. Stage 1 caps responses at 8k tokens (temp 1.0) with `train_prompt_bsz=128`, `N_SAMPLES_PER_PROMPT=16`, `NUM_ROLLOUT=10240`. Stage 2 extends to 16k (temp 1.1) using the same batch geometry for the compress → expand curriculum. Reward modeling uses `--rm-type deepscaler` plus dynamic reward filters and `--balance-data`. Evaluation runs every 20 steps on AIME 2024 with a 30k token ceiling.

### Optimization Notes
GRPO (`--advantage-estimator grpo`) handles policy updates. KL/entropy terms default to zero; enable as needed via `--use-kl-loss`, `--kl-loss-type low_var_kl` (For reference only, KL loss is not used in training). Optimizer: Adam lr=1e-6, constant schedule, β1=0.9, β2=0.98, weight decay 0.1. Backprop remains BF16 (`PRECISE_ARGS` keeps only `--bf16`); inference leverages the FP8 E8M0 checkpoint with `--rollout-num-gpus-per-engine 4`. Adjust length penalties or `OVER_SAMPLING_BATCH_SIZE` to trade throughput vs memory.

### Operations & Monitoring
Rerunning the same script will automatically reproduce the results. Stick to the EvalScope revision cited in the README [PR #734](https://github.com/modelscope/evalscope/pull/734) for comparable metrics. Validate Stage 1/2 RL checkpoints on AIME24, AIME25, GPQA, LiveCodeBench v5 to confirm FP8 inference parity.

Follow this workflow to reproduce the FP8 SFT results from InfiR2 and extend them into the two-stage GRPO RL curriculum with strong efficiency and long-context reasoning performance.
