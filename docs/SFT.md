## 🤖 Supervised Fine-tuning with FP8

### Overview
The InfiR2 SFT experiments demonstrate that FP8 supervised fine-tuning matches BF16 loss dynamics on Qwen2.5-Math-1.5B and 7B. Using the InfiAlign-SFT-72k → 165k curriculum yields up to **22% faster training**, **14% lower peak memory**, and **19% higher throughput**. This document distills the full recipe for reproducing those results, with concrete launch plans for single-node and multi-node runs. In practice we train both 1.5B and 7B models on two nodes (8 GPUs per node) and recommend the same setup for best parity with the paper.

---

### Available Scripts

We support both 7B and 1.5B models with flexible training configurations:

- 7B SFT
  - Stage1: [InfiR2_SFT_FP8_7B_stage1.sh](scripts/SFT/InfiR2_SFT_FP8_7B_stage1.sh).
  - Stage2: [InfiR2_SFT_FP8_7B_stage2.sh](scripts/SFT/InfiR2_SFT_FP8_7B_stage2.sh).
- 1.5B SFT
  - Stage1: [InfiR2_SFT_FP8_1.5B_stage1.sh](scripts/SFT/InfiR2_SFT_FP8_1.5B_stage1.sh).
  - Stage2: [InfiR2_SFT_FP8_1.5B_stage2.sh](scripts/SFT/InfiR2_SFT_FP8_1.5B_stage2.sh).


**Training Strategy Explained:**

The two stage balances data mixing, curriculum-guided SFT, and DPO to boost reasoning across various benchmarks. For more details, please refer to [InfiAlign](https://arxiv.org/abs/2508.05496).

**When to use which script:**
You can directly run the script on your model. If you want to get the best performance, please run the scripts for stage1 and stage2 in sequence.

### Configuration

The training scripts use Megatron-LM with comprehensive configuration options. Below are the key parameters you need to modify:

#### 1. Data & Environment Setup
**Datasets**  
- Stage 1: `InfiAlign-SFT-72k`
- Stage 2: `InfiAlign-SFT-165k`

**Context Length**: 32k for both stages.
- **Checkpoints required**  
  - HuggingFace base model, e.g. Qwen2.5-7B-Instruct  
  - Megatron torch-distributed base weights matching the HF model  

- **Launch prerequisites**  
  - Ray head node (`http://127.0.0.1:8265` by default)  
  - Environment variables in `runtime-env-json` for `PYTHONPATH`, NCCL socket interfaces, `CUDA_DEVICE_MAX_CONNECTIONS`, proxy settings, etc.  
  - W&B and TensorBoard configured for offline logging

```bash 
# Set your Slime path
HOME_DIR=/path/to/slime 

# Data paths
DATA_DIR=/path/to/InfiAlign-SFT-72k      # Swap to -165k for Stage 2
```

#### 2. Model Settings

HF_CHECKPOINT specifies the path for the model weights, while REF_LOAD specifies the path for the model configuration.

```bash
# Set your model config path
HF_CHECKPOINT=/path/to/base_models_hf/qwen2.5-7B-Instruct/

# Set your model weight path
REF_LOAD=/path/to/base_models_/qwen2.5-7B_torch_dist/

# Set your model load dir
LOAD_DIR=/path/to/checkpoints/InfiR2_SFT_FP8_7B_stg1/

# Set your outputs path
SAVE_DIR=/path/to/outputs/InfiR2_SFT_FP8_7B_stg1/
```

#### 3. Core Training Recipe
**Scripts**: `scripts/SFT/InfiR2_SFT_FP8_{model}_stage{1,2}.sh`

```bash
NUM_EPOCH=5                                     # --num-epoch
GLOBAL_BATCH_SIZE=64                            # --global-batch-size
ROLLOUT_BATCH_SIZE=64                           # --rollout-batch-size
TENSOR_PARALLEL=4                               # --tensor-model-parallel-size
PIPELINE_PARALLEL=1                             # --pipeline-model-parallel-size
CONTEXT_PARALLEL=1                              # --context-parallel-size
MAX_TOKENS_PER_GPU=32768                        # Dynamic batching cap
LEARNING_RATE=2e-5                              # Adam learning rate
LR_WARMUP_FRACTION=0.05                         # Cosine warmup ratio
LR_DECAY_STYLE=cosine                           # Learning-rate scheduler
ADAM_BETA1=0.9                                  # Adam β1
ADAM_BETA2=0.95                                 # Adam β2
WEIGHT_DECAY=0                                  # L2 regularization
```

Dropout is disabled, gradients and softmax accumulate in FP32, and flash attention is enabled. Precision combines `--bf16`, `--fp8-format e4m3`, `--fp8-recipe blockwise`, and `--fp8-param-gather`.
- **Stage hand-off**  
  - Keep the same script flags and environment  
  - Change `DATA_DIR` to your SFT data
  - Point `REF_LOAD` to the Stage 1 checkpoint if training Stage 2


### Running the Training 

#### Single-node Launch (8 GPUs)
Suitable for prototypes or limited hardware. Adjust global batch or sequence length if memory constrained.

```bash
# 7B Stage 1 single-node
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

bash scripts/SFT/InfiR2_SFT_FP8_7B_stage1.sh \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8

# Stage 2 reuses the same command with updated DATA_DIR/REF_LOAD
bash scripts/SFT/InfiR2_SFT_FP8_7B_stage2.sh \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8
```

**Notes**
- Maintain `TP=4` to match the official setup; reduce if GPUs per node are limited.  
- Use shorter sequence lengths (`--max-tokens-per-gpu`) if VRAM is insufficient at 32k context.  
- Expect higher wall-clock time than multi-node runs.

---

#### Multi-node Launch (Recommended, 2 Nodes × 8 GPUs)
Matches the configuration used for both 1.5B and 7B SFT runs in the paper.

```bash
# Launch Ray head on Node 1
MASTER_ADDR=10.0.0.1
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 \
  --dashboard-host=0.0.0.0 --dashboard-port=8265

# Launch Ray worker on Node 2
ray start --address="http://${MASTER_ADDR}:8265" --num-gpus 8

# Submit Stage 1 job from Node 1
bash scripts/SFT/InfiR2_SFT_FP8_7B_stage1.sh \
  --actor-num-nodes 2 \
  --actor-num-gpus-per-node 8 \
  --rdzv-endpoint ${MASTER_ADDR}:29400

# Submit Stage 2 job using Stage 1 checkpoint as REF_LOAD
bash scripts/SFT/InfiR2_SFT_FP8_7B_stage2.sh \
  --actor-num-nodes 2 \
  --actor-num-gpus-per-node 8 \
  --rdzv-endpoint ${MASTER_ADDR}:29400
```

Resume: For stage transitions, double-check that `LOAD_DIR` matches the previous stage’s final checkpoint path.

---

### Monitoring Training

- **TensorBoard**  
  - Log directory: `${LOG_DIR}/tensorboard/InfiR2-sft-fp8-${MODEL}-stg{1,2}`  
  - Launch: `tensorboard --logdir ${LOG_DIR}/tensorboard`
- **W&B**  
  - Offline mode by default. Sync with `wandb sync <run-dir>` after training if internet access is available.

---

### Evaluation

Evaluate Stage 1 and Stage 2 outputs on AIME24, AIME25, GPQA, LiveCodeBench v5 (via EvalScope [PR #734](https://github.com/modelscope/evalscope/pull/734)) to confirm FP8 parity with reported metrics.

By following this guide you can reliably reproduce the FP8 SFT results for both 1.5B and 7B models on two nodes while retaining single-node fallback procedures for smaller clusters.
