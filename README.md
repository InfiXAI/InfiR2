# InfiR2


<p align="center">
  <b>InfiR2: A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models</b>
</p>

<p align="center">
  <a href="https://zeyu-zhu.github.io/webpage/">Wenjun Wang*</a>,
  <a href="https://qhlin.me/">Shuo Cai*</a>,
  <a href="https://scholar.google.com/citations?user=h1-3lSoAAAAJ&hl=en">Congkai Xie</a> <br>
</p>


<p align="center">
  <a href="https://arxiv.org/abs/2509.22536">📄 Paper</a> &nbsp; | &nbsp;
  <a href="https://huggingface.co/datasets/ZaynZhu/Paper2Video">📊 Dataset</a> &nbsp; | &nbsp;
  <a href="https://showlab.github.io/Paper2Video/">🌐 Project Website</a> &nbsp; | &nbsp;
</p>



## 🔥 Update
- [x] [2025.10.8] We release the [code](https://github.com/showlab/Paper2Video) and [model](https://huggingface.co/datasets/ZaynZhu/Paper2Video).
- [x] [2025.9.26] We release the [arxiv paper](https://arxiv.org/abs/2509.22536).

---

### Table of Contents
- [🌟 Overview](#-overview)
- [🚀 Preparation]()
- [🤖 Pretrain with FP8]()
- [🌈 Supervised with FP8]()
- [📊 Evaluation](#-evaluation-paper2video)
- [🙏 Acknowledgements](#-acknowledgements)
- [📌 Citation](#-citation)

---

## 🌟 Overview


We introduce an end-to-end FP8 training recipe that seamlessly integrates continual pre-training and supervised fine-tuning. Our methodology employs a fine-grained, hybrid-granularity quantization strategy to maintain numerical fidelity while maximizing computational efficiency. Through extensive experiments, including the continue pre-training of models on a 160B-token corpus, we demonstrate that our recipe is not only remarkably stable but also essentially lossless, achieving performance on par with the BF16 baseline across a suite of reasoning benchmarks. Crucially, this is achieved with substantial efficiency improvements, including up to a 22% reduction in training time, a 14% decrease in peak memory usage, and a 19% increase in throughput. Our results establish FP8 as a practical and robust alternative to BF16, and we will release the accompanying code to further democratize large-scale model training.

<div align="center">
  <img src="https://github.com/InfiXAI/InfiR2/blob/main/figures/fp8_recipe.pdf" alt="Our approach" width="100%">
</div>




## 🚀 Preparation

### Environment Setup

写一个requirments，包含conda环境

包括git clone 代码 + pip install 环境


设置megatron和transformer_engine的路径
```bash

export PROJECT_ROOT=$(pwd)

export PYTHONPATH=${PROJECT_ROOT}/third_party/megatron:${PROJECT_ROOT}/third_party/transformer_engine:$PYTHONPATH
```



## 🤖 Pretrain with FP8

写运行脚本的位置

#### Dataset

具体修改数据集中哪个位置

#### Model

具体脚本中修改哪个位置

#### Example Usage

Run the following command to launch a full generation:

```bash
python pipeline.py \
    --model_name_t gpt-4.1 \
    --model_name_v gpt-4.1 \
    --model_name_talking hallo2 \
    --result_dir /path/to/output \
    --paper_latex_root /path/to/latex_proj \
    --ref_img /path/to/ref_img.png \
    --ref_audio /path/to/ref_audio.wav \
    --talking_head_env /path/to/hallo2_env \
    --gpu_list [0,1,2,3,4,5,6,7]
```

## 🌈 Supervised Fine-tuning with FP8

We provide two-stage SFT training scripts with FP8 quantization following [InfiAlign](https://arxiv.org/abs/2508.05496). The training process uses Ray for distributed execution and supports multi-node training configurations.

- SFT Stage1: [InfiR2_SFT_FP8_stage1.sh](scripts/SFT/InfiR2_SFT_FP8_stage1.sh).

- SFT Stage2: [InfiR2_SFT_FP8_stage2.sh](scripts/SFT/InfiR2_SFT_FP8_stage2.sh).

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
#### Running 🚀
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
PYTHONPATH=/root/Megatron-LM:/root/slime python tools/convert_torch_dist_to_hf.py \
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

- Stage 1: [InfiR2_RL_FP8_stage1_4node.sh](scripts/RL/InfiR2_RL_FP8_stage1_4node.sh) with 8K response lengths.
- Stage 2: [InfiR2_RL_FP8_stage2_4node.sh](scripts/RL/InfiR2_RL_FP8_stage2_4node.sh) with 8K response lengths and higher temperature.

#### Configuration

**Dataset:** Set the `DATA_DIR` to your RL training data:
```bash
DATA_DIR=/path/to/data/dapo-math-17k.jsonl
```

**Model Configuration:**
- `HF_CHECKPOINT`: Path to the FP8 E8M0 converted model (for inference)
- `REF_LOAD`: Path to the SFT Stage 2 checkpoint in PyTorch distributed format

Example:
```bash
HF_CHECKPOINT=/path/to/InfiR2_SFT_FP8_stg2_hf_e8m0/
REF_LOAD=/path/to/InfiR2_SFT_FP8_stg2/
```

**Key Parameters:**
- `ROLLOUT_MAX_RESPONSE_LEN=8192`: Response length limit for Stage 1
- `MAX_TOKENS_PER_GPU=8192`: Memory-efficient token packing

#### Running 🚀
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

1. **AIME 2024** - [aime24_eval.sh](eval/aime24_eval.sh)
2. **AIME 2025** - [aime25_eval.sh](eval/aime25_eval.sh)
3. **GPQA Diamond** - [gpqa_eval.sh](eval/gpqa_eval.sh)
4. **LiveCodeBench v5** - [livecodebenchv5_eval.sh](eval/livecodebenchv5_eval.sh)

### Running Evaluations

Each script uses SLURM for job scheduling and SGLang for efficient inference serving. The evaluation pipeline consists of:

1. Starting an SGLang server with the model
2. Running evalscope with the specified benchmark
3. Automatically collecting and logging results

**Example: Evaluating on AIME 2024**

```bash
# Configure the script with your paths
# Edit eval/aime24_eval.sh to set:
# - MODEL_PATH: /path/to/your/InfiR2_model
# - DATASET_LOCAL: /path/to/aime_2024
# - Other necessary paths

# Submit the evaluation job
sbatch eval/aime24_eval.sh
```

**Key Evaluation Parameters by Benchmark:**

| Benchmark | Script | Max Tokens | Samples | Temperature |
|-----------|--------|------------|---------|-------------|
| AIME 2024 | [aime24_eval.sh](eval/aime24_eval.sh) | 31,000 | 32 | 0.65 |
| AIME 2025 | [aime25_eval.sh](eval/aime25_eval.sh) | 31,000 | 32 | 0.65 |
| GPQA | [gpqa_eval.sh](eval/gpqa_eval.sh) | 26,000 | 8 | 0.65 |
| LiveCodeBench | [livecodebenchv5_eval.sh](eval/livecodebenchv5_eval.sh) | 27,000 | 8 | 0.65 |


## 🙏 Acknowledgements

* The souces of the presentation videos are SlideLive and YouTuBe.
* We thank all the authors who spend a great effort to create presentation videos!
* We thank [CAMEL](https://github.com/camel-ai/camel) for open-source well-organized multi-agent framework codebase.
* We thank the authors of [Hallo2](https://github.com/fudan-generative-vision/hallo2.git) and [Paper2Poster](https://github.com/Paper2Poster/Paper2Poster.git) for their open-sourced codes.
* We thank [Wei Jia](https://github.com/weeadd) for his effort in collecting the data and implementing the baselines. We also thank all the participants involved in the human studies.
* We thank all the **Show Lab @ NUS** members for support!

---

## 📌 Citation


If you find our work useful, please cite:

```bibtex
@misc{wang2025infir2comprehensivefp8training,
      title={InfiR2: A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models}, 
      author={Wenjun Wang and Shuo Cai and Congkai Xie and Mingfa Feng and Yiming Zhang and Zhen Li and Kejing Yang and Ming Li and Jiannong Cao and Yuan Xie and Hongxia Yang},
      year={2025},
      eprint={2509.22536},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.22536}, 
}
```