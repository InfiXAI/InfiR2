# InfiR2
[English](./README.md)

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

### 目录
- [🌟 概述](#-概述)
- [🚀 准备工作](#-准备工作)
- [🤖 FP8 持续预训练](#-FP8-持续预训练)
- [🌈 FP8 监督微调](#-FP8-监督微调)
- [📊 评测](#-评测)
- [🙏 致谢](#-致谢)
- [📌 引用](#-引用)


## 🌟 概述


我们引入了一种端到端的 FP8 训练方案，该方案无缝集成了持续预训练和监督微调。我们的方法采用了一种细粒度的混合粒度量化策略，以在保持数值保真度的同时最大化计算效率。通过广泛的实验，包括在 160B token 语料库上对模型进行持续预训练，我们证明了我们的方案不仅非常稳定，而且基本上是无损的，在一系列推理基准上实现了与 BF16 基线相当的性能。至关重要的是，这是在显著提高效率的情况下实现的，包括高达 22% 的训练时间减少、14% 的峰值内存使用量降低和 19% 的吞吐量提升。我们的结果证实了 FP8 能够作为 BF16 的一个实用且稳健的替代方案，我们发布了完整的配套代码以进一步普及大规模模型训练。

<div align="center">
  <img src="assets/fp8_recipe.png" alt="我们的方法" width="90%">
</div>

---

- **显存优化与加速.** 与广泛使用的 BF16 精度相比，FP8 能够提供:
  - 端到端训练速度提升高达 22%。
  - 峰值内存使用量节省高达 14%。
  - 端到端吞吐量提升高达 19%。

  Model Size = 1.5B


  <div align="center">

  **Context Length = 32k, TP = 2, CP = 1, MBS = 1**
  |      | Forward | Backward | Total | Ratio | Peak Memory | Ratio | Throughput | Ratio |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | BF16 | 841 ms | 2329 ms | 3170 ms | - | 57.8 GB | - | 345 TFlops | - |
  | FP8  | 875 ms | 2075 ms | 2950 ms | 0.93× | 51.7 GB | 0.89× | 360 TFlops | 1.04× |

  **Context Length = 8k, TP = 1, CP = 1, MBS = 2**
  |      | Forward | Backward | Total | Ratio | Peak Memory | Ratio | Throughput | Ratio |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | BF16 | 463 ms | 1567 ms | 2030 ms | - | 68.1 GB | - | 340 TFlops | - |
  | FP8  | 529 ms | 1061 ms | 1590 ms | 0.78× | 58.3 GB | 0.86× | 376 TFlops | 1.10× |

  </div>


  Model Size = 7B

<div align="center">

**Context Length = 32k, TP = 4, CP = 1, MBS = 1**
|      | Forward | Backward | Total | Ratio | Peak Memory | Ratio | Throughput | Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BF16 | 2790 ms | 6800 ms | 9590 ms | - | 78.1 GB | - | 409 TFlops | - |
| FP8  | 2660 ms | 5700 ms | 8360 ms | 0.87× | 67.4 GB | 0.86× | 461 TFlops | 1.14× |

**Context Length = 32k, TP = 2, CP = 1, MBS = 1**
|      | Forward | Backward | Total | Ratio | Peak Memory | Ratio | Throughput | Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BF16 | 1760 ms | 5320 ms | 7080 ms | - | 53.2 GB | - | 453 TFlops | - |
| FP8  | 2300 ms | 3230 ms | 5530 ms | 0.78× | 50.8 GB | 0.95× | 537 TFlops | 1.19× |
	
</div>


## 🚀 环境准备

为了拉取此仓库，请使用：
```bash
git clone --recursive https://github.com/InfiXAI/InfiR2
```

### 环境设置

我们支持通过 **Docker** 进行环境配置，并提供**自定义 Docker 文件**。详情请参照以下说明：

### Docker 设置

自定义配置的 Docker 镜像存储在 [Dockerfile](docker/Dockerfile)。使用以下代码构建 Docker 镜像：

```bash
docker build --no-cache \
    --file docker/Dockerfile \
    --build-arg HTTP_PROXY="$http_proxy" \
    --build-arg HTTPS_PROXY="$https_proxy" \
    --build-arg NO_PROXY="localhost,127.0.0.1" \
    --build-arg SGLANG_VERSION=${SGLANG_VERSION:-latest} \
    --build-arg MEGATRON_COMMIT=${MEGATRON_COMMIT:-main} \
    -t infir2-training:latest .
```

**主要组件：**
- **基础镜像**: `lmsysorg/sglang:${SGLANG_VERSION}`
- **Megatron-LM**: core_v0.14.0 分支（NVIDIA 官方版本）
- **TransformerEngine**: v2.4.0（commit 3cd6870）- ⚠️ 必须使用此版本以避免精度/显存问题
- **FlashAttention**: v2.7.4.post1 + Hopper 构建
- **其他组件**: slime、mbridge、torch_memory_saver、ray、sglang-router 等

更多详情，请参考 [docker/README.md](docker/README.md)。


## 🤖 FP8 持续预训练

我们提供了使用 FP8 量化的持续预训练 (CPT) 脚本。我们的 FP8 训练方案与 BF16 基线相比，实现了**高达 22% 的训练时间减少**、**14% 的峰值内存使用量降低**以及**19% 的吞吐量提升**，同时在推理基准上保持了相当的性能。更多详情，请参考 [docs/Pretrain.md](docs/Pretrain.md)

### 可用脚本

我们支持 7B 和 1.5B 模型的灵活训练配置：

- **7B 模型**
  - 完整训练: [InfiR2_CPT_FP8_7B.sh](scripts/CPT/InfiR2_CPT_FP8_7B.sh) - 完整的预热+稳定+衰减流程
  - 仅衰减: [InfiR2_CPT_FP8_7B_decay.sh](scripts/CPT/InfiR2_CPT_FP8_7B_decay.sh) - 可选的独立衰减阶段
- **1.5B 模型**
  - 完整训练: [InfiR2_CPT_FP8_1.5B.sh](scripts/CPT/InfiR2_CPT_FP8_1.5B.sh) - 完整的预热+稳定+衰减流程
  - 仅衰减: [InfiR2_CPT_FP8_1.5B_decay.sh](scripts/CPT/InfiR2_CPT_FP8_1.5B_decay.sh) - 可选的独立衰减阶段

#### 运行

**选项 1: 完整训练流程 (推荐)**

一次性运行完整的预热+稳定+退火训练：

```bash
bash scripts/CPT/InfiR2_CPT_FP8_7B.sh
```

这个脚本将自动完成所有三个训练阶段。

**选项 2: 使用独立的退火脚本 (高级)**

如果您想从稳定阶段的某个特定检查点进入退火阶段：

```bash
# 首先，确定您在稳定阶段的检查点
# 然后使用该检查点运行退火脚本
bash scripts/CPT/InfiR2_CPT_FP8_7B_decay.sh \
    --load exp/InfiR2_CPT_FP8_7B/checkpoints/iter_0035000```
```

## 🌈 FP8 监督微调

我们提供了遵循 [InfiAlign](https://arxiv.org/abs/2508.05496) 论文，进行 FP8 量化的两阶段 SFT 训练。训练过程使用 Ray 进行分布式执行，并支持多节点训练配置。更多详情，请参考 [docs/SFT.md](docs/SFT.md)。

### 可用脚本

我们支持 7B 和 1.5B 模型的灵活训练配置：

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
- `HF_CHECKPOINT`: 指向 HuggingFace 格式的模型路径 (例如 Qwen2.5-7B-Instruct)
- `REF_LOAD`: 指向 PyTorch 分布式格式的基础模型权重路径


```bash
HF_CHECKPOINT=/path/to/base_models_hf/qwen2.5-7B-Instruct/
REF_LOAD=/path/to/base_models_/qwen2.5-7B_torch_dist/
```
#### 运行
首先，启动 Ray 集群：
```bash
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
```

然后，启动训练：
```bash
bash scripts/SFT/InfiR2_SFT_FP8_7B_stage1.sh
```

## 🎯 FP8 强化学习

我们的 RL 训练流程包含两个阶段：首先压缩响应长度，然后扩展它。在开始 RL 训练之前，您需要将 SFT 检查点转换为 FP8 E8M0 格式，以便在生成 rollout 阶段进行高效的 FP8 推理。更多详情，请参考 [docs/RL.md](docs/RL.md)。

### 用于 RL 的模型转换

完成 SFT 阶段二后，请先将模型转换为 HuggingFace 格式，然后再转换为 FP8 E8M0 格式：

```bash
# 步骤 1: 将 PyTorch 分布式检查点转换为 HuggingFace 格式
PYTHONPATH=training/Megatron-LM:training/slime python tools/convert_torch_dist_to_hf.py \
    --input-dir /path/to/InfiR2_SFT_FP8_stg2 \
    --output-dir /path/to/InfiR2_SFT_FP8_stg2_hf \
    --origin-hf-dir /path/to/models/Qwen2.5-7B-Instruct

# 步骤 2: 将 BF16 格式的 HuggingFace 模型转换为 FP8 E8M0 格式
python tools/bf16_cast_fp8.py \
    --input-bf16-hf-path /path/to/InfiR2_SFT_FP8_stg2_hf \
    --output-fp8-hf-path /path/to/InfiR2_SFT_FP8_stg2_hf_e8m0 \
    --force-pow-2-scale True
```

转换后的 FP8 E8M0 模型将用于 RL 的 rollout 阶段的推理，从而显著提升生成效率。

- 阶段一: [InfiR2_RL_FP8_7B_stage1_4node.sh](scripts/RL/InfiR2_RL_FP8_7B_stage1_4node.sh)，响应长度为 8K。
- 阶段二: [InfiR2_RL_FP8_7B_stage2_4node.sh](scripts/RL/InfiR2_RL_FP8_7B_stage2_4node.sh)，响应长度为 16K，并使用更高的温度。

#### 配置

**数据集:** 将 `DATA_DIR` 设置为您的 RL 训练数据：
```bash
DATA_DIR=/path/to/data/dapo-math-17k.jsonl
```

**模型配置:**
- `HF_CHECKPOINT`: 指向转换后的 FP8 E8M0 模型路径 (用于推理)
- `REF_LOAD`: 指向 SFT 阶段二的 PyTorch 分布式格式检查点路径

```bash
HF_CHECKPOINT=/path/to/your_model/

REF_LOAD=/path/to/your_model/
```

#### 运行
启动 RL 训练的方式与 SFT 相同。首先启动 Ray，然后运行脚本。

这种基于课程学习的策略可以确保在不同响应长度要求下的训练稳定性和最优性能。


## 📊 评测

我们使用开源的 [evalscope](https://github.com/modelscope/evalscope) 框架进行所有模型评测，以确保可复现性。我们的评测套件包含了四个推理基准测试，并提供了相应的评测脚本。

### 环境设置

我们已经验证了我们的模型可以在最新版本的 evalscope 上正确运行，并取得一致的性能结果。然而，为了严格复现我们在论文中报告的评测结果，请使用以下特定版本的 evalscope：

**用于复现的推荐版本:**
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
<div align="center">

| 基准 | 脚本 | 最大 Tokens | 样本数 | 温度系数 |
|-----------|--------|------------|---------|-------------|
| AIME 2024 | [aime24_eval.sh](scripts/eval/aime24_eval.sh) | 31,000 | 32 | 0.65 |
| AIME 2025 | [aime25_eval.sh](scripts/eval/aime25_eval.sh) | 31,000 | 32 | 0.65 |
| GPQA | [gpqa_eval.sh](scripts/eval/gpqa_eval.sh) | 26,000 | 8 | 0.65 |
| LiveCodeBench | [livecodebenchv5_eval.sh](scripts/eval/livecodebenchv5_eval.sh) | 27,000 | 8 | 0.65 |

</div>

每个脚本都使用 slurm 进行作业调度，并使用 SGLang 提供高效的推理服务。评测流程包括：

1. 使用模型启动一个 SGLang 服务
2. 运行 evalscope 并指定相应的基准测试

### 模型表现
- 7B模型

<div align="center">

<table>
  <thead>
    <tr>
      <th align="left">Model</th>
      <th align="center">AIME 25</th>
      <th align="center">AIME 24</th>
      <th align="center">GPQA</th>
      <th align="center">LiveCodeBench v5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><strong>Deepseek-Distill-Qwen-7B</strong></td>
      <td align="center">43.00</td>
      <td align="center">49.00</td>
      <td align="center">48.20</td>
      <td align="center">37.60</td>
    </tr>
    <tr>
      <td align="left"><strong>Qwen2.5-7B-base (w. InfiAlign)</strong></td>
      <td align="center">33.75</td>
      <td align="center">43.02</td>
      <td align="center">48.11</td>
      <td align="center">39.48</td>
    </tr>
    <tr>
      <td align="left"><strong>InfiR2-7B-Instruct-FP8</strong></td>
      <td align="center">40.62</td>
      <td align="center">55.73</td>
      <td align="center">45.33</td>
      <td align="center">40.31</td>
    </tr>
    </tr>
  </tbody>
</table>

</div>


- 1.5B模型
<div align="center">

<table>
  <thead>
    <tr>
      <th align="left">Model</th>
      <th align="center">AIME 25</th>
      <th align="center">AIME 24</th>
      <th align="center">GPQA</th>
      <th align="center">LiveCodeBench v5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><strong>Deepseek-Distill-Qwen-1.5B</strong></td>
      <td align="center">21.35</td>
      <td align="center">26.87</td>
      <td align="center">32.26</td>
      <td align="center">18.50</td>
    </tr>
    <tr>
      <td align="left"><strong>Qwen2.5-1.5B-base (w. InfiAlign)</strong></td>
      <td align="center">14.58</td>
      <td align="center">10.52</td>
      <td align="center">28.98</td>
      <td align="center">12.99</td>
    </tr>
    <tr>
      <td align="left"><strong>InfiR2-1.5B-Instruct-FP8</strong></td>
      <td align="center">18.45</td>
      <td align="center">17.39</td>
      <td align="center">29.48</td>
      <td align="center">17.10</td>
    </tr>
  </tbody>
</table>

</div>

## 🙏 致谢

我们在此对以下开源项目表示诚挚的感谢：

* **[slime](https://github.com/THUDM/slime)** - 用于 RL 扩展的大语言模型后训练框架，支持 GLM-4.5 和 GLM-4.6 的训练。slime 支持几乎所有与 Megatron-LM 兼容的模型训练。我们正在与 slime 社区积极合作，致力于实现完全训推一致的 FP8 RL 训练。
* **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** - NVIDIA 开发的大规模 transformer 模型训练框架。
* **[TransformerEngine](https://github.com/NVIDIA/TransformerEngine)** - 用于在 NVIDIA GPU 上加速 transformer 模型的 FP8 精度库。
* **[Qwen2.5](https://github.com/QwenLM/Qwen2.5-Math)** - 启发我们工作的基础模型。


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
