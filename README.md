# InfiR2


<p align="center">
  <b>InfiR2: A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models</b>
</p>

<p align="center">
  <a href="https://scholar.google.com/citations?hl=zh-CN&user=1LA3TSAAAAAJ">Wenjun Wang*</a>,
  <a href="https://scholar.google.com/citations?user=VRlUiqQAAAAJ">Shuo Cai*</a>,
  <a href="https://scholar.google.com/citations?user=I6SAtGMAAAAJ&hl=en">Congkai Xie</a>, Mingfa Feng, Yiming Zhang, Zhen Li, Kejing Yang, Ming Li, Jiannong Cao, Hongxia Yang <br>

</p>


<p align="center">
  <a href="https://arxiv.org/abs/2509.22536">📄 Paper</a> &nbsp; | &nbsp;
  <a href="https://huggingface.co/collections/InfiX-ai/infir2-68edca7ae3c3f052b2db0eed">🤗 Huggingface </a> &nbsp; | &nbsp;
  <a href="https://infix-ai.com/research/infir2/">🌐 Project Website</a> &nbsp; | &nbsp;
</p>



## 🔥 Update
- [x] [2025.10.8] We release the [code](https://github.com/InfiXAI/InfiR2) and [model](https://huggingface.co/collections/InfiX-ai/infir2-68edca7ae3c3f052b2db0eed).
- [x] [2025.9.26] We release the [arxiv paper](https://arxiv.org/abs/2509.22536).

---

### Table of Contents
- [🌟 Overview](#-overview)
- [🚀 Preparation](#-preparation)
- [🤖 Continual Pre-training with FP8](#-continual-pre-training-with-fp8)
- [🌈 Supervised Fine-tuning with FP8](#-supervised-fine-tuning-with-fp8)
- [📊 Evaluation](#-evaluation-paper2video)
- [🙏 Acknowledgements](#-acknowledgements)
- [📌 Citation](#-citation)

---

## 🌟 Overview


We introduce an end-to-end FP8 training recipe that seamlessly integrates continual pre-training and supervised fine-tuning. Our methodology employs a fine-grained, hybrid-granularity quantization strategy to maintain numerical fidelity while maximizing computational efficiency. Through extensive experiments, including the continue pre-training of models on a 160B-token corpus, we demonstrate that our recipe is not only remarkably stable but also essentially lossless, achieving performance on par with the BF16 baseline across a suite of reasoning benchmarks. Crucially, this is achieved with substantial efficiency improvements, including up to a 22% reduction in training time, a 14% decrease in peak memory usage, and a 19% increase in throughput. Our results establish FP8 as a practical and robust alternative to BF16, and we will release the accompanying code to further democratize large-scale model training.

<div align="center">
  <img src="assets/fp8_recipe.png" alt="Our approach" width="100%">
</div>




## 🚀 Preparation

### Environment Setup

We support environment setup via **Conda** and **Docker**. Both methods are based on the official setup guide from the [THUDM/slime](https://github.com/THUDM/slime) repository. Please follow the instructions in the links below.

---

### Option 1: Conda Setup

*   **Instructions**: Please follow the detailed guide at [**THUDM/slime Conda Build Documentation**](https://github.com/THUDM/slime/blob/main/docs/README.md).

---

### Option 2: Docker Setup

*   **Instructions**: Please refer to the official Docker configuration files and guide at [**THUDM/slime Docker Directory**](https://github.com/THUDM/slime/tree/main/docker).

To clone this repository, please use:
```bash
git clone --recursive https://github.com/InfiXAI/InfiR2
```


## 🤖 Continual Pre-training with FP8

We provide continual pre-training (CPT) scripts with FP8 quantization. Our FP8 training recipe achieves **up to 22% reduction in training time**, **14% decrease in peak memory usage**, and **19% increase in throughput** compared to BF16 baseline, while maintaining performance parity on reasoning benchmarks.

### Available Scripts

We support both 7B and 1.5B models with flexible training configurations:

- **7B Model**
  - Complete Training: [InfiR2_CPT_FP8_7B.sh](scripts/CPT/InfiR2_CPT_FP8_7B.sh) - Full warmup+stable+decay pipeline
  - Decay Only: [InfiR2_CPT_FP8_7B_decay.sh](scripts/CPT/InfiR2_CPT_FP8_7B_decay.sh) - Optional standalone decay phase
- **1.5B Model**
  - Complete Training: [InfiR2_CPT_FP8_1.5B.sh](scripts/CPT/InfiR2_CPT_FP8_1.5B.sh) - Full warmup+stable+decay pipeline
  - Decay Only: [InfiR2_CPT_FP8_1.5B_decay.sh](scripts/CPT/InfiR2_CPT_FP8_1.5B_decay.sh) - Optional standalone decay phase

**Training Strategy Explained:**

The standard scripts (without `_decay` suffix) provide a **complete training pipeline** that includes all three phases: warmup → stable → decay. These scripts will train your model from start to finish in a single run.

The standalone decay scripts (with `_decay` suffix) are provided for **flexibility**. They allow you to:
- Enter the decay phase from any checkpoint saved during the stable phase
- Experiment with different decay schedules without re-running the entire training
- Resume training with a different learning rate schedule

**When to use which script:**
- **For most users**: Use the complete training scripts (e.g., `InfiR2_CPT_FP8_7B.sh`) to run the full pipeline
- **For advanced experimentation**: Use the decay-only scripts when you want to start decay from a specific stable-phase checkpoint without completing the full training first

### Configuration

The training scripts use Megatron-LM with comprehensive configuration options. Below are the key parameters you need to modify:

#### 1. Environment and Paths

```bash
# Set your Megatron-LM path
CODE_DIR=/path/to/your/Megatron-LM

# Data paths
DATA_PATH="/path/to/your/training/data"        # Your pre-training corpus
DATA_CACHE_PATH="/path/to/data/cache"          # Cache directory for processed data
TOKENIZER_MODEL="/path/to/Qwen2.5-7B"          # HuggingFace tokenizer path

# Checkpoint and logging
CHECKPOINT_PATH=$OUT_DIR/checkpoints            # Auto-generated checkpoint path
```

#### 2. FP8 Quantization Settings

```bash
FP8_RECIPE=blockwise                            # Quantization granularity (blockwise recommended)
```

Our implementation uses a hybrid-granularity quantization strategy with E4M3 format for optimal numerical fidelity.

#### 3. Distributed Training Configuration

```bash
# Multi-node settings (configure via command line)
NNODES=1                                        # Number of nodes
GPUS_PER_NODE=8                                 # GPUs per node
RDZV_ENDPOINT="localhost:29400"                 # Master node address

# Parallelism strategy
GBS=128                                         # Global batch size
MBS=1                                           # Micro batch size per GPU
TP_SIZE=4                                       # Tensor parallel size
PP_SIZE=1                                       # Pipeline parallel size
CP_SIZE=1                                       # Context parallel size
```

**Parallelism Guidelines:**
- For 7B models: Use `TP_SIZE=4` with 8 GPUs (recommended)
- For 1.5B models: Use `TP_SIZE=2` or `TP_SIZE=4` depending on sequence length
- Adjust `GBS` and `MBS` based on your GPU memory and desired throughput

#### 4. Model Architecture

The scripts are pre-configured for Qwen2.5 architecture. Key parameters:

```bash
SEQ_LENGTH=32768                                # Training sequence length
# 7B Model: 28 layers, 3584 hidden size, 28 attention heads, 4 KV heads
# 1.5B Model: Different layer/hidden configurations (check respective scripts)
```

#### 5. Training Hyperparameters

**Warmup and Stable Phase:**
```bash
MAX_STEPS=40000                                 # Total training steps
WARMUP_STEPS=5000                               # Learning rate warmup steps
WSD_DECAY_STEPS=5000                            # Decay steps for WSD scheduler
SEED=1236                                       # Random seed
```

**Decay Phase:**
```bash
MAX_STEPS=40000                                 # Total training steps
WARMUP_STEPS=0                                  # No warmup in decay phase
WSD_DECAY_STEPS=5000                            # Cosine decay steps
```

**Optimizer Settings:**
```bash
--lr 1e-4                                       # Peak learning rate
--min-lr 1e-5                                   # Minimum learning rate
--weight-decay 0.1                              # Weight decay coefficient
--adam-beta1 0.9                                # Adam beta1
--adam-beta2 0.95                               # Adam beta2
--clip-grad 1.0                                 # Gradient clipping
--lr-decay-style WSD                            # Warmup-Stable-Decay scheduler
```

#### 6. Logging and Checkpointing

```bash
--save-interval 1000                            # Save checkpoint every N steps
--eval-interval 100                             # Evaluate every N steps
--eval-iters 10                                 # Number of evaluation iterations
--log-interval 1                                # Log metrics every N steps
```

Logs and checkpoints will be saved to:
- Checkpoints: `exp/$EXP/checkpoints/`
- TensorBoard: `exp/$EXP/tensorboard/`
- Training logs: `exp/$EXP/logs/`

### Running the Training

#### Single Node Training

**Option 1: Complete Training Pipeline (Recommended)**

Run the full warmup+stable+decay training in one go:

```bash
bash scripts/CPT/InfiR2_CPT_FP8_7B.sh
```

This single script will complete all three training phases automatically.

**Option 2: Using Standalone Decay Script (Advanced)**

If you want to enter the decay phase from a specific checkpoint in the stable phase:

```bash
# First, identify your stable-phase checkpoint
# Then run the decay script with the checkpoint
bash scripts/CPT/InfiR2_CPT_FP8_7B_decay.sh \
    --load exp/InfiR2_CPT_FP8_7B/checkpoints/iter_0035000
```

#### Multi-Node Training

For distributed training across multiple nodes:

```bash
# Run on each node with the same master address
bash scripts/CPT/InfiR2_CPT_FP8_7B.sh \
    --nnodes 4 \
    --rdzv_endpoint master_node_ip:29400
```

**Multi-Node Setup:**
1. Ensure all nodes have access to the same shared filesystem for data and checkpoints
2. Configure the master node IP address (typically rank 0 node)
3. Ensure the port (default 29400) is open and accessible across all nodes
4. Launch the same script on all nodes simultaneously

#### Advanced: Flexible Decay Phase Entry

The standalone decay scripts provide flexibility to enter the decay phase from any stable-phase checkpoint. **When using the decay-only script**, you **must** specify the checkpoint using the `--load` parameter:

```bash
bash scripts/CPT/InfiR2_CPT_FP8_7B_decay.sh \
    --load exp/InfiR2_CPT_FP8_7B/checkpoints/iter_0035000
```

**Key Feature**: The decay script uses `--override-opt_param-scheduler`, which resets the learning rate scheduler while preserving model weights and optimizer states. This makes it versatile for any training scenario where you need to modify the learning rate schedule mid-training, such as:
- Starting decay from an arbitrary stable-phase checkpoint
- Experimenting with different decay schedules
- Fine-tuning the learning rate strategy for specific downstream tasks

#### Automatic Training Resumption

**If training terminates unexpectedly** (e.g., due to system failure, timeout, or interruption), simply **re-run the same launch command** to automatically resume from the last saved checkpoint.

The script will automatically:
- Detect the most recent checkpoint in `CHECKPOINT_PATH`
- Resume training from that checkpoint
- Continue with the same training configuration

No manual intervention is required for checkpoint resumption.

### Monitoring Training

The training progress can be monitored through:

1. **TensorBoard**: View real-time metrics
   ```bash
   tensorboard --logdir exp/InfiR2_CPT_FP8_7B/tensorboard
   ```

2. **W&B (Weights & Biases)**: Configure project name in the script
   ```bash
   --wandb-project "your_project_name"
   --wandb-exp-name $EXP
   ```

### Checkpoint Management

**Automatic Resumption:**

The training scripts automatically resume from the latest checkpoint in `CHECKPOINT_PATH` if available. If training is interrupted, simply re-run the same command to continue training seamlessly.

**Manual Checkpoint Loading (Advanced):**

For advanced use cases, you can manually load from a specific checkpoint using the `--load` parameter. This is particularly useful when using the standalone decay scripts:

```bash
bash scripts/CPT/InfiR2_CPT_FP8_7B_decay.sh \
    --load /path/to/specific/checkpoint
```

**Optional: Custom Phase Transition Workflow**

While the complete training scripts handle all phases automatically, you can manually control phase transitions for advanced experimentation:

1. Run the complete training script and save checkpoints during the stable phase
2. Identify a checkpoint from the stable phase (e.g., `iter_0035000`)
3. Launch the standalone decay script with that checkpoint:
   ```bash
   bash scripts/CPT/InfiR2_CPT_FP8_7B_decay.sh \
       --load exp/InfiR2_CPT_FP8_7B/checkpoints/iter_0035000
   ```

This approach allows you to enter the decay phase from any stable-phase checkpoint without waiting for the complete training to finish. The `--override-opt_param-scheduler` flag ensures the learning rate scheduler is properly reset while preserving model weights and optimizer states.

## 🌈 Supervised Fine-tuning with FP8

We provide two-stage SFT training scripts with FP8 quantization following [InfiAlign](https://arxiv.org/abs/2508.05496). The training process uses Ray for distributed execution and supports multi-node training configurations.

- 7B SFT
  - Stage1: [InfiR2_SFT_FP8_7B_stage1.sh](scripts/SFT/InfiR2_SFT_FP8_7B_stage1.sh).
  - Stage2: [InfiR2_SFT_FP8_7B_stage2.sh](scripts/SFT/InfiR2_SFT_FP8_7B_stage2.sh).
- 1.5B SFT
  - Stage1: [InfiR2_SFT_FP8_1.5B_stage1.sh](scripts/SFT/InfiR2_SFT_FP8_1.5B_stage1.sh).
  - Stage2: [InfiR2_SFT_FP8_1.5B_stage2.sh](scripts/SFT/InfiR2_SFT_FP8_1.5B_stage2.sh).

#### Configuration

**Dataset:** Modify the `DATA_DIR` variable to point to your training data:
```bash
DATA_DIR=/path/to/stage1_data
```

**Model Configuration:**
- `HF_CHECKPOINT`: Path to the base model in HuggingFace format (e.g., Qwen2.5-7B)
- `REF_LOAD`: Path to the base model weights in PyTorch distributed format


```bash
HF_CHECKPOINT=/path/to/base_models_hf/qwen2.5-7B/
REF_LOAD=/path/to/base_models_/qwen2.5-7B_torch_dist/
```
#### Running
First, start Ray cluster:
```bash
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
```

Then launch the training:
```bash
bash scripts/SFT/InfiR2_SFT_FP8_stage1.sh
```

---

## 🎯 Reinforcement Learning with FP8

Our RL training pipeline consists of two stages: first compressing the response length, then expanding it. Before RL training, you need to convert the SFT checkpoint to FP8 E8M0 format for efficient FP8 inference during rollout generation.

### Model Conversion for RL

After completing SFT Stage 2, convert the model to HuggingFace format, then to FP8 E8M0 format:

```bash
# Step 1: Convert PyTorch distributed checkpoint to HuggingFace format
PYTHONPATH=training/Megatron-LM:training/slime python tools/convert_torch_dist_to_hf.py \
    --input-dir /path/to/InfiR2_SFT_FP8_stg2 \
    --output-dir /path/to/InfiR2_SFT_FP8_stg2_hf \
    --origin-hf-dir /path/to/models/Qwen2.5-7B

# Step 2: Convert BF16 HuggingFace model to FP8 E8M0 format
python tools/bf16_cast_fp8.py \
    --input-bf16-hf-path /path/to/InfiR2_SFT_FP8_stg2_hf \
    --output-fp8-hf-path /path/to/InfiR2_SFT_FP8_stg2_hf_e8m0 \
    --force-pow-2-scale True
```

The FP8 E8M0 model will be used for inference during the RL rollout phase, significantly improving generation efficiency.

- Stage 1: [InfiR2_RL_FP8_7B_stage1_4node.sh](scripts/RL/InfiR2_RL_FP8_7B_stage1_4node.sh) with 8K response lengths.
- Stage 2: [InfiR2_RL_FP8_7B_stage2_4node.sh](scripts/RL/InfiR2_RL_FP8_7B_stage2_4node.sh) with 8K response lengths and higher temperature.

#### Configuration

**Dataset:** Set the `DATA_DIR` to your RL training data:
```bash
DATA_DIR=/path/to/data/dapo-math-17k.jsonl
```

**Model Configuration:**
- `HF_CHECKPOINT`: Path to the FP8 E8M0 converted model (for inference)
- `REF_LOAD`: Path to the SFT Stage 2 checkpoint in PyTorch distributed format

```bash
HF_CHECKPOINT=/path/to/InfiR2_SFT_FP8_stg2_hf_e8m0/
REF_LOAD=/path/to/InfiR2_SFT_FP8_stg2/
```

#### Running 
The way to launch RL training is the same as SFT. First start ray and then run the script.

This curriculum-based strategy ensures stable training and optimal performance across different response length requirements.


## 📊 Evaluation

We use the open-source [evalscope](https://github.com/modelscope/evalscope) framework for all model evaluations to ensure reproducibility. Our evaluation suite includes four reasoning benchmarks with provided evaluation scripts.

### Environment Setup

We have verified that our models work correctly with the latest version of evalscope, achieving consistent performance results. However, to strictly reproduce the exact evaluation results reported in our paper, please use the following specific version of evalscope:

**Recommended Version for Reproduction:**
- Repository: [evalscope](https://github.com/modelscope/evalscope)
- Branch: `main`
- Pull Request: [Add qwen-code best practice doc #734](https://github.com/modelscope/evalscope/pull/734)

**Installation:**

Follow the official documentation at [https://evalscope.readthedocs.io/zh-cn/latest/get_started/installation.html](https://evalscope.readthedocs.io/zh-cn/latest/get_started/installation.html)

```bash
git clone https://github.com/modelscope/evalscope.git
cd evalscope/
pip install -e .
```

### Evaluation Benchmarks

We provide evaluation scripts for four key reasoning benchmarks:

| Benchmark | Script | Max Tokens | Samples | Temperature |
|-----------|--------|------------|---------|-------------|
| AIME 2024 | [aime24_eval.sh](scripts/eval/aime24_eval.sh) | 31,000 | 32 | 0.65 |
| AIME 2025 | [aime25_eval.sh](scripts/eval/aime25_eval.sh) | 31,000 | 32 | 0.65 |
| GPQA | [gpqa_eval.sh](scripts/eval/gpqa_eval.sh) | 26,000 | 8 | 0.65 |
| LiveCodeBench | [livecodebenchv5_eval.sh](scripts/eval/livecodebenchv5_eval.sh) | 27,000 | 8 | 0.65 |

### Running Evaluations

Each script uses slurm for job scheduling and SGLang for efficient inference serving. The evaluation pipeline consists of:

1. Starting an SGLang server with the model
2. Running evalscope with the specified benchmark

## 🙏 Acknowledgements

* We would like to express our gratitude for the following open-source projects: [Slime](https://github.com/THUDM/slime), [Megatron](https://github.com/NVIDIA/Megatron-LM), [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) and [Qwen2.5](https://github.com/QwenLM/Qwen2.5-Math)。

---

## 📌 Citation


If you find our work useful, please cite:

```bibtex
@misc{wang2025infir2comprehensivefp8training,
      title={InfiR2: A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models}, 
      author={Wenjun Wang and Shuo Cai and Congkai Xie and Mingfa Feng and Yiming Zhang and Zhen Li and Kejing Yang and Ming Li and Jiannong Cao and Hongxia Yang},
      year={2025},
      eprint={2509.22536},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.22536}, 
}
```
