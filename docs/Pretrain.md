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