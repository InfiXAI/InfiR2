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


## 📊 Evaluation

1.如何将模型转成FP8 \
2.如何设置UE8M0 eval \
3.如何评测AIME24，25等benchmark

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