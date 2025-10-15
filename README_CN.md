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

* [x] [2025.10.8] 我们发布了 [代码](https://github.com/InfiXAI/InfiR2) 和 [模型](https://huggingface.co/collections/InfiX-ai/infir2-68edca7ae3c3f052b2db0eed)。
* [x] [2025.9.26] 我们发布了 [arxiv 论文](https://arxiv.org/abs/2509.22536)。

---

### 目录

* [🌟 概述](#-概述)
* [🚀 环境准备](#-环境准备)
* [🤖 FP8 持续预训练](#-FP8_持续预训练)
* [🌈 FP8 监督微调](#-FP8_监督微调)
* [📊 模型评测](#-模型评测)
* [🙏 致谢](#-致谢)
* [📌 引用](#-引用)

## 🌟 概述

我们推出了一个**端到端**的 $\text{FP8}$ 训练方案，无缝集成了持续预训练和监督微调。我们的方法采用了一种**细粒度、混合精度粒度的量化策略**，以在保持数值准确性的同时最大化计算效率。通过大量实验，包括在 $\text{1600}$ 亿 $\text{token}$ 语料库上对模型进行持续预训练，我们证明了我们的方案不仅**极其稳定**，而且**基本上是无损的**，在一系列推理基准测试中达到了与 $\text{BF16}$ 基线**相当的性能**。至关重要的是，这在实现性能无损的同时，还带来了显著的效率提升，包括**训练时间减少高达 22%**、**峰值内存使用减少 14%** 和**吞吐量增加 19%**。我们的结果确立了 $\text{FP8}$ 作为 $\text{BF16}$ 实用且强大的替代方案，我们将发布配套代码以进一步推动大规模模型训练的民主化。

<div align="center">
  <img src="assets/fp8_recipe.png" alt="我们的方法" width="100%">
</div>

---

## 🚀 准备工作

克隆此仓库，请使用：
```bash
git clone --recursive [https://github.com/InfiXAI/InfiR2](https://github.com/InfiXAI/InfiR2)
````

### 环境设置

我们支持通过 **Conda** 和 **Docker** 进行环境设置。这两种方法都基于 [THUDM/slime](https://github.com/THUDM/slime) 仓库的官方设置指南。请遵循以下链接中的说明。

-----

### Docker 设置

自定义配置的 $\text{Docker}$ 镜像存储在 [Dockerfile.te\_fp8.cu129](https://www.google.com/search?q=docker/Dockerfile.te_fp8.cu129)。使用以下代码运行 $\text{Docker}$：

```base
docker build --no-cache \
    --file docker/Dockerfile.te_fp8.cu129 \
    --build-arg HTTP_PROXY="$http_proxy" \
    --build-arg HTTPS_PROXY="$https_proxy" \
    --build-arg NO_PROXY="localhost,127.0.0.1" \
    --build-arg SGLANG_VERSION=${SGLANG_VERSION:-v0.5.0rc0-cu129} \
    --build-arg MEGATRON_COMMIT=${MEGATRON_COMMIT:-main} \
    -t infix/te-fp8:cu129 .
```

有关更多详细信息，请参阅 [docker/README.md](https://www.google.com/search?q=docker/README.md)。

-----

## 🤖 FP8 持续预训练

我们提供了使用 $\text{FP8}$ 量化的持续预训练 ($\text{CPT}$) 脚本。我们的 $\text{FP8}$ 训练方案实现了**训练时间减少高达 22%**、**峰值内存使用减少 14%** 和**吞吐量增加 19%**，同时在推理基准测试中保持了与 $\text{BF16}$ 基线相当的性能。有关更多详细信息，请参阅 [docs/Pretrain.md](https://www.google.com/search?q=docs/Pretrain.md)。

### 可用脚本

我们支持 $\text{7B}$ 和 $\text{1.5B}$ 模型的灵活训练配置：

  - **7B 模型**
      - 完整训练：[InfiR2\_CPT\_FP8\_7B.sh](https://www.google.com/search?q=scripts/CPT/InfiR2_CPT_FP8_7B.sh) - 完整的 $\text{warmup}$ + $\text{stable}$ + $\text{decay}$ 流程
      - 仅 $\text{Decay}$：[InfiR2\_CPT\_FP8\_7B\_decay.sh](https://www.google.com/search?q=scripts/CPT/InfiR2_CPT_FP8_7B_decay.sh) - 可选的独立 $\text{decay}$ 阶段
  - **1.5B 模型**
      - 完整训练：[InfiR2\_CPT\_FP8\_1.5B.sh](https://www.google.com/search?q=scripts/CPT/InfiR2_CPT_FP8_1.5B.sh) - 完整的 $\text{warmup}$ + $\text{stable}$ + $\text{decay}$ 流程
      - 仅 $\text{Decay}$：[InfiR2\_CPT\_FP8\_1.5B\_decay.sh](https://www.google.com/search?q=scripts/CPT/InfiR2_CPT_FP8_1.5B_decay.sh) - 可选的独立 $\text{decay}$ 阶段

#### 运行

**选项 1：完整训练流程（推荐）**

一键运行完整的 $\text{warmup}$ + $\text{stable}$ + $\text{decay}$ 训练：

```bash
bash scripts/CPT/InfiR2_CPT_FP8_7B.sh
```

此单个脚本将自动完成所有三个训练阶段。

**选项 2：使用独立 $\text{Decay}$ 脚本（高级）**

如果您想从 $\text{stable}$ 阶段的特定检查点进入 $\text{decay}$ 阶段：

```bash
# 首先，确定您的 stable 阶段检查点
# 然后运行 decay 脚本并指定该检查点
bash scripts/CPT/InfiR2_CPT_FP8_7B_decay.sh \
    --load exp/InfiR2_CPT_FP8_7B/checkpoints/iter_0035000
```

-----

## 🌈 FP8 监督微调

我们提供了遵循 [InfiAlign](https://arxiv.org/abs/2508.05496) 的两阶段 $\text{FP8}$ 量化 $\text{SFT}$ 训练脚本。训练过程使用 $\text{Ray}$ 进行分布式执行，并支持多节点训练配置。有关更多详细信息，请参阅 [docs/SFT.md](https://www.google.com/search?q=docs/SFT.md)。

### 可用脚本

我们支持 $\text{7B}$ 和 $\text{1.5B}$ 模型的灵活训练配置：

  - 7B $\text{SFT}$
      - 阶段 1：[InfiR2\_SFT\_FP8\_7B\_stage1.sh](https://www.google.com/search?q=scripts/SFT/InfiR2_SFT_FP8_7B_stage1.sh)。
      - 阶段 2：[InfiR2\_SFT\_FP8\_7B\_stage2.sh](https://www.google.com/search?q=scripts/SFT/InfiR2_SFT_FP8_7B_stage2.sh)。
  - 1.5B $\text{SFT}$
      - 阶段 1：[InfiR2\_SFT\_FP8\_1.5B\_stage1.sh](https://www.google.com/search?q=scripts/SFT/InfiR2_SFT_FP8_1.5B_stage1.sh)。
      - 阶段 2：[InfiR2\_SFT\_FP8\_1.5B\_stage2.sh](https://www.google.com/search?q=scripts/SFT/InfiR2_SFT_FP8_1.5B_stage2.sh)。

#### 配置

**数据集：** 修改 $\text{DATA\_DIR}$ 变量以指向您的训练数据：

```bash
DATA_DIR=/path/to/stage1_data
```

**模型配置：**

  - `HF_CHECKPOINT`：$\text{HuggingFace}$ 格式的基础模型路径（例如 $\text{Qwen2.5-7B}$）
  - `REF_LOAD`：$\text{PyTorch Distributed}$ 格式的基础模型权重路径

<!-- end list -->

```bash
HF_CHECKPOINT=/path/to/base_models_hf/qwen2.5-7B/
REF_LOAD=/path/to/base_models_/qwen2.5-7B_torch_dist/
```

#### 运行

首先，启动 $\text{Ray}$ 集群：

```bash
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
```

然后启动训练：

```bash
bash scripts/SFT/InfiR2_SFT_FP8_7B_stage1.sh
```

-----

## 🎯 FP8 强化学习

我们的 $\text{RL}$ 训练流程包括两个阶段：首先压缩响应长度，然后扩展响应长度。在 $\text{RL}$ 训练之前，您需要将 $\text{SFT}$ 检查点转换为 $\text{FP8 E8M0}$ 格式，以提高 $\text{rollout}$ 生成过程中的 $\text{FP8}$ 推理效率。有关更多详细信息，请参阅 [docs/RL.md](https://www.google.com/search?q=docs/RL.md)。

### $\text{RL}$ 模型转换

完成 $\text{SFT}$ 阶段 $\text{2}$ 后，将模型转换为 $\text{HuggingFace}$ 格式，然后再转换为 $\text{FP8 E8M0}$ 格式：

```bash
# 步骤 1: 将 PyTorch distributed 检查点转换为 HuggingFace 格式
PYTHONPATH=training/Megatron-LM:training/slime python tools/convert_torch_dist_to_hf.py \
    --input-dir /path/to/InfiR2_SFT_FP8_stg2 \
    --output-dir /path/to/InfiR2_SFT_FP8_stg2_hf \
    --origin-hf-dir /path/to/models/Qwen2.5-7B

# 步骤 2: 将 BF16 HuggingFace 模型转换为 FP8 E8M0 格式
python tools/bf16_cast_fp8.py \
    --input-bf16-hf-path /path/to/InfiR2_SFT_FP8_stg2_hf \
    --output-fp8-hf-path /path/to/InfiR2_SFT_FP8_stg2_hf_e8m0 \
    --force-pow-2-scale True
```

$\text{FP8 E8M0}$ 模型将用于 $\text{RL rollout}$ 阶段的推理，显著提高了生成效率。

  - 阶段 1：[InfiR2\_RL\_FP8\_7B\_stage1\_4node.sh](https://www.google.com/search?q=scripts/RL/InfiR2_RL_FP8_7B_stage1_4node.sh)，响应长度为 $\text{8K}$。
  - 阶段 2：[InfiR2\_RL\_FP8\_7B\_stage2\_4node.sh](https://www.google.com/search?q=scripts/RL/InfiR2_RL_FP8_7B_stage2_4node.sh)，响应长度为 $\text{16K}$，温度更高。

#### 配置

**数据集：** 设置 $\text{DATA\_DIR}$ 为您的 $\text{RL}$ 训练数据：

```bash
DATA_DIR=/path/to/data/dapo-math-17k.jsonl
```

**模型配置：**

  - `HF_CHECKPOINT`：转换后的 $\text{FP8 E8M0}$ 模型路径（用于推理）
  - `REF_LOAD`：$\text{PyTorch Distributed}$ 格式的 $\text{SFT}$ 阶段 $\text{2}$ 检查点路径

<!-- end list -->

```bash
HF_CHECKPOINT=/path/to/your_model/

REF_LOAD=/path/to/your_model/
```

#### 运行

启动 $\text{RL}$ 训练的方式与 $\text{SFT}$ 相同。首先启动 $\text{Ray}$，然后运行脚本。

这种基于课程的策略确保了训练的稳定性，并在不同的响应长度要求下实现了最佳性能。

-----

## 📊 评估

我们使用开源的 [evalscope](https://github.com/modelscope/evalscope) 框架进行所有模型评估，以确保可复现性。我们的评估套件包括四个推理基准测试，并提供了相应的评估脚本。

### 环境设置

我们已验证模型在最新版本的 $\text{evalscope}$ 下可以正常工作，并能达到一致的性能结果。但是，为了严格复现我们论文中报告的准确评估结果，请使用以下特定版本的 $\text{evalscope}$：

**建议用于复现的版本：**

  - 仓库：[evalscope](https://github.com/modelscope/evalscope)
  - 分支：`main`
  - 拉取请求 ($\text{Pull Request}$)：[Add qwen-code best practice doc \#734](https://github.com/modelscope/evalscope/pull/734)

**安装：**

遵循官方文档 [https://evalscope.readthedocs.io/zh-cn/latest/get\_started/installation.html](https://evalscope.readthedocs.io/zh-cn/latest/get_started/installation.html)

```bash
git clone [https://github.com/modelscope/evalscope.git](https://github.com/modelscope/evalscope.git)
cd evalscope/
pip install -e .
```

### 评估基准

我们为四个关键推理基准提供了评估脚本：

| 基准 | 脚本 | 最大 $\text{Token}$ 数 | 样本数 | 温度 |
| :---: | :---: | :---: | :---: | :---: |
| $\text{AIME 2024}$ | [aime24\_eval.sh](https://www.google.com/search?q=scripts/eval/aime24_eval.sh) | $\text{31,000}$ | 32 | 0.65 |
| $\text{AIME 2025}$ | [aime25\_eval.sh](https://www.google.com/search?q=scripts/eval/aime25_eval.sh) | $\text{31,000}$ | 32 | 0.65 |
| $\text{GPQA}$ | [gpqa\_eval.sh](https://www.google.com/search?q=scripts/eval/gpqa_eval.sh) | $\text{26,000}$ | 8 | 0.65 |
| $\text{LiveCodeBench}$ | [livecodebenchv5\_eval.sh](https://www.google.com/search?q=scripts/eval/livecodebenchv5_eval.sh) | $\text{27,000}$ | 8 | 0.65 |

### 运行评估

每个脚本都使用 $\text{slurm}$ 进行作业调度，并使用 $\text{SGLang}$ 进行高效推理服务。评估流程包括：

1.  使用模型启动 $\text{SGLang}$ 服务器
2.  使用指定的基准运行 $\text{evalscope}$

-----

## 🙏 致谢

  * 我们要感谢以下开源项目：[Slime](https://github.com/THUDM/slime), [Megatron](https://github.com/NVIDIA/Megatron-LM), [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) 和 [Qwen2.5](https://github.com/QwenLM/Qwen2.5-Math)。

-----

## 📌 引用

如果我们的工作对您有所帮助，请引用：

```bibtex
@misc{wang2025infir2comprehensivefp8training,
      title={InfiR2: A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models}, 
      author={Wenjun Wang and Shuo Cai and Congkai Xie and Mingfa Feng and Yiming Zhang and Zhen Li and Kejing Yang and Ming Li and Jiannong Cao and Hongxia Yang},
      year={2025},
      eprint={2509.22536},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={[https://arxiv.org/abs/2509.22536](https://arxiv.org/abs/2509.22536)}, 
}
```