## 🎯 FP8 Reinforcement Learning (RL)

### Overview
InfiR2 extends FP8 efficiency into reinforcement learning through a two-stage DAPO curriculum. Stage 1 compresses responses to 8k tokens, Stage 2 expands to 16k. Both use FP8 inference (E8M0) for rollouts while keeping DAPO training in BF16. This guide explains data preparation, checkpoint conversion, launch commands for multi-node execution, and monitoring.

---

### 1. Prerequisites
- **Upstream SFT checkpoint**: Stage 2 FP8 SFT output (torch distributed) is required.
- **Converted models**:  
  1. Convert torch checkpoint to HuggingFace format.  
  2. Cast the HF model to FP8 E8M0 for rollout engines.
- **Datasets**: `dapo-math-17k.jsonl` (curriculum data used in the paper).
- **Cluster**: Production runs expect four nodes (8 GPUs each) for training *and* dedicated GPUs per rollout engine. Adjust as needed.
- **Ray**: Head node accessible at `http://${MASTER_ADDR}:8265`, workers joined with identical environment variables.

```bash
# Convert SFT Stage 2 checkpoint
PYTHONPATH=training/Megatron-LM:training/slime \
python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/to/InfiR2_SFT_FP8_stg2 \
  --output-dir /path/to/InfiR2_SFT_FP8_stg2_hf \
  --origin-hf-dir /path/to/models/Qwen2.5-7B

python tools/bf16_cast_fp8.py \
  --input-bf16-hf-path /path/to/InfiR2_SFT_FP8_stg2_hf \
  --output-fp8-hf-path /path/to/InfiR2_SFT_FP8_stg2_hf_e8m0 \
  --force-pow-2-scale True
```

---

### 2. Configuration Template
Fill the following variables in `scripts/RL/InfiR2_RL_FP8_7B_stage{1,2}_4node.sh`.

```bash
HOME_DIR=/path/to/slime
LOG_DIR=/path/to/logs
DATA_DIR=/path/to/datasets/dapo-math-17k.jsonl
HF_CHECKPOINT=/path/to/InfiR2_SFT_FP8_stg2_hf_e8m0/
REF_LOAD=/path/to/InfiR2_SFT_FP8_stg2/            # torch distributed checkpoint
LOAD_DIR=/path/to/RL_stage1/                      # Stage 1 output (Stage 2 reuses)
SAVE_DIR=/path/to/RL_stage1/
```

Scripts automatically handle W&B (offline), TensorBoard logging, and Ray runtime environment variables.

---

### 3. Stage 1: Compress (8k)

```bash
ROLLOUT_MAX_LEN=8192                           # Maximum generated tokens
ROLLOUT_TEMPERATURE=1.0                        # Sampling temperature
TRAIN_PROMPT_BSZ=128                           # Prompts per training batch
N_SAMPLES_PER_PROMPT=16                        # Responses per prompt
NUM_ROLLOUT=10240                              # Total responses collected
EVAL_INTERVAL=20                               # Steps between eval runs
EVAL_N_SAMPLES=16                              # Samples per eval prompt
EVAL_MAX_RESPONSE_LEN=30000                    # Cap for evaluation decoding
ROLLOUT_NUM_GPUS_PER_ENGINE=4                  # GPUs dedicated to each SGLang engine
```

Training and rollout processes use four nodes × eight GPUs. Reward modeling relies on `--rm-type deepscaler` with dynamic sampling filters to avoid zero-variance rewards (follow DAPO).

```bash
# Launch Ray cluster (example with four nodes)
MASTER_ADDR=10.0.0.1
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 \
  --dashboard-host=0.0.0.0 --dashboard-port=8265
ray start --address="http://${MASTER_ADDR}:8265" --num-gpus 8

# Submit Stage 1
bash scripts/RL/InfiR2_RL_FP8_7B_stage1_4node.sh \
  --actor-num-nodes 4 \
  --actor-num-gpus-per-node 8 \
  --rdzv-endpoint ${MASTER_ADDR}:29400
```

---

### 4. Stage 2: Expand (16k)

```bash
ROLLOUT_MAX_LEN=16384                          # Extended response window
ROLLOUT_TEMPERATURE=1.1                        # Higher exploration temperature
TRAIN_PROMPT_BSZ=128                           # Same batch geometry as Stage 1
N_SAMPLES_PER_PROMPT=16                        # Maintain 2048 global batch
NUM_ROLLOUT=10240                              # Keep curriculum length
EVAL_INTERVAL=20                               # Evaluation cadence
EVAL_N_SAMPLES=16                              # Samples per eval prompt
EVAL_MAX_RESPONSE_LEN=30000                    # Evaluation token cap
```

Initialize from Stage 1 by pointing `REF_LOAD` to the Stage 1 torch checkpoint and updating `LOAD_DIR` / `SAVE_DIR` to the Stage 2 directory. Reward filters and evaluation cadence stay unchanged.

```bash
bash scripts/RL/InfiR2_RL_FP8_7B_stage2_4node.sh \
  --actor-num-nodes 4 \
  --actor-num-gpus-per-node 8 \
  --rdzv-endpoint ${MASTER_ADDR}:29400
```

---

### 5. Optimization Details

```bash
ADVANTAGE_ESTIMATOR=grpo                      # Policy-gradient estimator
KL_COEF=0.0                                   # Symmetric KL weight (enable if needed)
ENTROPY_COEF=0.0                              # Entropy bonus (enable if needed)
LEARNING_RATE=1e-6                            # Adam learning rate
LR_DECAY_STYLE=constant                       # Keep LR fixed during RL
ADAM_BETA1=0.9                                # Adam β1
ADAM_BETA2=0.98                               # Adam β2
WEIGHT_DECAY=0.1                              # Regularization
PRECISION_FLAG=bf16                           # Training precision (FP8 inference only)
```

When GPU memory is limited or the hardware scale is reduced, reduce `OVER_SAMPLING_BATCH_SIZE`, or increase TP or CP to reduce memory usage, and adjust `--rollout-num-gpus-per-engine`.

---

### 6. Monitoring 

- **TensorBoard**: `${LOG_DIR}/tensorboard` (enable via `--tensorboard-dir`).  
- **W&B**: Offline by default. Sync using `wandb sync`.  

---

