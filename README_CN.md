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
  <a href="https://huggingface.co/datasets/ZaynZhu/Paper2Video">🤗 Huggingface </a> &nbsp; | &nbsp;
  <a href="https://infix-ai.com/research/infir2/">🌐 Project Website</a> &nbsp; | &nbsp;
</p>


## 🔥 更新
- [x] [2025.10.8] 我们发布了[代码](https://github.com/InfiXAI/InfiR2)和[模型](https://huggingface.co/datasets/ZaynZhu/Paper2Video)。
- [x] [2025.9.26] 我们发布了 [arXiv 论文](https://arxiv.org/abs/2509.22536)。


### 内容
- [🌟 概览](#-概览)
- [🚀 环境准备](#-环境准备)
- [🤖 FP8 预训练](#-fp8-预训练)
- [🌈 FP8 监督微调](#-fp8-监督微调)
- [🎯 FP8 强化学习](#-fp8-强化学习)
- [📊 评估](#-评估)
- [🙏 致谢](#-致谢)
- [📌 引用](#-引用)

---

## 🌟 概览

我们引入了一个端到端的 **FP8 训练方案**，无缝集成了持续预训练（CPT）和监督微调（SFT）。我们的方法采用了**细粒度的、混合粒度的量化策略**，在保持数值保真度的同时最大化计算效率。通过在 160B Token 语料库上进行模型的持续预训练等大量实验，我们证明了该方案不仅**非常稳定且基本无损**，在所有推理基准测试中均实现了与 BF16 基线相当的性能。至关重要的是，这带来了显著的效率提升，包括**训练时间减少 22%**，**峰值内存使用降低 14%**，以及**吞吐量增加 19%**。我们的结果确立了 FP8 作为 BF16 的一个实用且强大的替代方案，我们将发布配套代码以进一步推动大规模模型训练的普及化。

<div align="center">
  <img src="assets/fp8_recipe.png" alt="我们的方案" width="100%">
</div>

## 🚀 环境准备

### 环境配置

我们支持通过 **Conda** 和 **Docker** 进行环境配置。这两种方法均基于 [THUDM/slime](https://github.com/THUDM/slime) 仓库的官方设置指南。请参考以下链接中的说明。

---

### 方案 1: Conda 配置

* **说明**: 请遵循 [**THUDM/slime Conda 构建文档**](https://github.com/THUDM/slime/blob/main/docs/README.md) 中的详细指南。

---

### 方案 2: Docker 配置

* **说明**: 请参考 [**THUDM/slime Docker 目录**](https://github.com/THUDM/slime/tree/main/docker) 中的官方 Docker 配置文件和指南。

为了拉取此仓库，请使用
```bash
git clone --recursive https://github.com/InfiXAI/InfiR2
```

## 🤖 FP8 预训练

我们提供了使用 FP8 量化进行持续预训练的脚本。

- 7B CPT
  - warmup 和 stable: [`InfiR2_CPT_FP8_7B.sh`](scripts/CPT/InfiR2_CPT_FP8_7B.sh)。
  - decay: [`InfiR2_CPT_FP8_7B_decay.sh`](scripts/CPT/InfiR2_CPT_FP8_7B_decay.sh)。
- 1.5B CPT
  - warmup 和 stable: [`InfiR2_CPT_FP8_1.5B.sh`](scripts/CPT/InfiR2_CPT_FP8_1.5B.sh)。
  - decay: [`InfiR2_CPT_FP8_1.5B_decay.sh`](scripts/CPT/InfiR2_CPT_FP8_1.5B_decay.sh)。

#### Configuration

**请直接在脚本内部修改所需的参数；所有可配置的参数都在那里定义好了。**

#### Example

```bash
bash InfiR2_CPT_FP8_7B.sh --nodes N --rdzv_endpoint master_ip:master_port
```

## 🌈 使用 FP8 进行监督微调

我们提供了遵循 [InfiAlign](https://arxiv.org/abs/2508.05496) 论文，使用 FP8 量化的两阶段 SFT 训练脚本。训练过程使用 Ray 进行分布式执行，并支持多节点训练配置。

- 7B SFT
  - 阶段一: [InfiR2_SFT_FP8_7B_stage1.sh](scripts/SFT/InfiR2_SFT_FP8_7B_stage1.sh).
  - 阶段二: [InfiR2_SFT_FP8_7B_stage2.sh](scripts/SFT/InfiR2_SFT_FP8_7B_stage2.sh).
- 1.5B SFT
  - 阶段一: [InfiR2_SFT_FP8_1.5B_stage1.sh](scripts/SFT/InfiR2_SFT_FP8_1.5B_stage1.sh).
  - 阶段二: [InfiR2_SFT_FP8_1.5B_stage2.sh](scripts/SFT/InfiR2_SFT_FP8_1.5B_stage2.sh).

#### 配置

**数据集:** 修改 `DATA_DIR` 变量，使其指向您的训练数据：
```bash
DATA_DIR=/path/to/stage1_data
```

**模型配置:** 
- HF_CHECKPOINT: 指向 HuggingFace 格式的基础模型路径 (例如 Qwen2.5-7B)
- REF_LOAD: 指向 PyTorch 分布式格式的基础模型权重路径

```bash
HF_CHECKPOINT=/path/to/base_models_hf/qwen2.5-7B/
REF_LOAD=/path/to/base_models_/qwen2.5-7B_torch_dist/
```
#### 运行
首先，启动 Ray 集群：
```bash
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
```

然后，启动训练：
```bash
bash scripts/SFT/InfiR2_SFT_FP8_stage1.sh
```

---

## 🎯 使用 FP8 进行强化学习

我们的 RL 训练流程包含两个阶段：首先压缩响应长度，然后扩展它。在开始 RL 训练之前，您需要将 SFT 检查点转换为 FP8 E8M0 格式，以便在生成 rollout 阶段进行高效的 FP8 推理。

### 用于 RL 的模型转换

完成 SFT 阶段二后，请先将模型转换为 HuggingFace 格式，然后再转换为 FP8 E8M0 格式：

```bash
# 步骤 1: 将 PyTorch 分布式检查点转换为 HuggingFace 格式
PYTHONPATH=training/Megatron-LM:training/slime python tools/convert_torch_dist_to_hf.py \
    --input-dir /path/to/InfiR2_SFT_FP8_stg2 \
    --output-dir /path/to/InfiR2_SFT_FP8_stg2_hf \
    --origin-hf-dir /path/to/models/Qwen2.5-7B

# 步骤 2: 将 BF16 格式的 HuggingFace 模型转换为 FP8 E8M0 格式
python tools/bf16_cast_fp8.py \
    --input-bf16-hf-path /path/to/InfiR2_SFT_FP8_stg2_hf \
    --output-fp8-hf-path /path/to/InfiR2_SFT_FP8_stg2_hf_e8m0 \
    --force-pow-2-scale True
```

转换后的 FP8 E8M0 模型将用于 RL 的 rollout 阶段的推理，从而显著提升生成效率。

- Stage 1: [InfiR2_RL_FP8_7B_stage1_4node.sh](scripts/RL/InfiR2_RL_FP8_7B_stage1_4node.sh) ，响应长度为 8K。
- Stage 2: [InfiR2_RL_FP8_7B_stage2_4node.sh](scripts/RL/InfiR2_RL_FP8_7B_stage2_4node.sh)，响应长度为 8K，并使用更高的温度系数。

#### 配置

**数据集:** 将 `DATA_DIR` 设置为您的 RL 训练数据：
```bash
DATA_DIR=/path/to/data/dapo-math-17k.jsonl
```

**模型配:**
- `HF_CHECKPOINT`: 指向转换后的 FP8 E8M0 模型路径 (用于推理)
- `REF_LOAD`: 指向 SFT 阶段二的 PyTorch 分布式格式检查点路径

```bash
HF_CHECKPOINT=/path/to/InfiR2_SFT_FP8_stg2_hf_e8m0/
REF_LOAD=/path/to/InfiR2_SFT_FP8_stg2/
```

#### 运行 
启动 RL 训练的方式与 SFT 相同。首先启动 Ray，然后运行脚本。这种基于课程学习的策略可以确保在不同响应长度要求下的训练稳定性和最优性能。


## 📊 评测

我们使用 [evalscope](https://github.com/modelscope/evalscope) 框架进行所有模型评测，以确保可复现性。我们的评测套件包含了四个推理基准测试，并提供了相应的评测脚本。

### 环境设置

我们已经验证了我们的模型可以在最新版本的 evalscope 上正确运行，并取得一致的性能结果。然而，为了严格复现我们在论文中报告的评测结果，请使用以下特定版本的 evalscope：

**用于复现的推荐版本::**
- 仓库: [evalscope](https://github.com/modelscope/evalscope)
- 分支: `main`
- 拉取请求: [Add qwen-code best practice doc #734](https://github.com/modelscope/evalscope/pull/734)

**安装:**

请遵循官方文档进行安装 [https://evalscope.readthedocs.io/zh-cn/latest/get_started/installation.html](https://evalscope.readthedocs.io/zh-cn/latest/get_started/installation.html)

```bash
git clone https://github.com/modelscope/evalscope.git
cd evalscope/
pip install -e .
```

### 评测基准

我们为四个关键的推理基准提供了评测脚本：

| 基准 | 脚本 | 最大 Tokens | 样本数 | 温度系数 |
|-----------|--------|------------|---------|-------------|
| AIME 2024 | [aime24_eval.sh](scripts/eval/aime24_eval.sh) | 31,000 | 32 | 0.65 |
| AIME 2025 | [aime25_eval.sh](scripts/eval/aime25_eval.sh) | 31,000 | 32 | 0.65 |
| GPQA | [gpqa_eval.sh](scripts/eval/gpqa_eval.sh) | 26,000 | 8 | 0.65 |
| LiveCodeBench | [livecodebenchv5_eval.sh](scripts/eval/livecodebenchv5_eval.sh) | 27,000 | 8 | 0.65 |

### 运行评测

每个脚本都使用 slurm 进行作业调度，并使用 SGLang 提供高效的推理服务。评测流程包括：

1. 使用模型启动一个 SGLang 服务
2. 运行 evalscope 并指定相应的基准测试

## 🙏 致谢

* 我们在此对以下开源项目表示诚挚的感谢: [Slime](https://github.com/THUDM/slime), [Megatron](https://github.com/NVIDIA/Megatron-LM), [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) and [Qwen2.5](https://github.com/QwenLM/Qwen2.5-Math)。

---

## 📌 引用


如果您觉得我们的工作有用，请引用：

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