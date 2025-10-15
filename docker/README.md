# 🧱 Docker Image: InfiR2 Training Environment

This Docker image provides a **TransformerEngine FP8-optimized** environment built on **SGLang** and **Megatron-LM**.  
It is designed for efficient large-scale model fine-tuning and inference, especially on NVIDIA Hopper architecture (H100).

> **Note:** Our Docker build approach is largely inspired by the [slime](https://github.com/THUDM/slime) project. We express our gratitude to the slime team for their excellent work.


### Image Overview

**Base:** `lmsysorg/sglang:${SGLANG_VERSION}`  
**Dockerfile:** `Dockerfile`  
**CUDA:** inherited from SGLang base image  
**PyTorch:** inherited from SGLang base image  
**Main Components:**
- [SGLang](https://github.com/sgl-project/sglang) (with custom patches)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (core_v0.14.0)
- [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) (v2.4.0, commit 3cd6870)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) (v2.7.4.post1 + Hopper build)
- [Apex](https://github.com/NVIDIA/apex) (latest)
- [Grouped GEMM](https://github.com/fanshiqing/grouped_gemm) (v1.1.4)
- [mbridge](https://github.com/ISEEKYAN/mbridge.git)
- [torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver.git)

---

### Build Instructions

You can build the image directly from the root of the repository:

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

 **Notes:**

* `SGLANG_VERSION` – SGLang version tag (default: `latest`). Make sure you have a corresponding patch file at `docker/patch/${SGLANG_VERSION}/sglang.patch`.
* `MEGATRON_COMMIT` – Megatron-LM commit to use (default: `main`). The Dockerfile clones the `core_v0.14.0` branch.
* Use `--no-cache` to force rebuild all layers if updating dependencies.
* The build process applies a custom SGLang patch from `docker/patch/${SGLANG_VERSION}/sglang.patch`. Make sure this file exists before building.
* **Patch Issues:** If you encounter any problems with the SGLang patches, please feel free to [open an issue](https://github.com/InfiXAI/InfiR2/issues) and we will fix it promptly.


### Run the Container

To launch the container with GPU and FP8 support:

```bash
docker run -it --rm \
    --gpus all \
    --shm-size=128g \
    -v $(pwd):/workspace \
    infir2-training:latest bash
```

This will open a shell inside the FP8-optimized environment.

If you want to run interactively with **Ray** or **SGLang router**, you can use:

```bash
docker run -it --rm \
    --gpus all \
    -p 8080:8080 \
    -e RAY_memory_monitor_refresh_ms=0 \
    infir2-training:latest bash
```

Then inside the container:

```bash
sglang-router serve --host 0.0.0.0 --port 8080
```

---

### Key Installed Components

| Component                  | Version / Commit                     | Notes                                       |
| -------------------------- | ------------------------------------ | ------------------------------------------- |
| **SGLang**                 | `${SGLANG_VERSION}`                  | Base framework with custom patches          |
| **Megatron-LM**            | `core_v0.14.0`                       | NVIDIA official branch                      |
| **TransformerEngine**      | `v2.4.0` (commit `3cd6870`)          | ⚠️ Must use v2.4.0 to avoid precision/memory issues |
| **FlashAttention**         | `v2.7.4.post1` + Hopper (`27f501d`)  | Standard + custom Hopper build              |
| **Apex**                   | latest                               | CUDA & C++ extensions enabled               |
| **Grouped GEMM**           | `v1.1.4`                             | Optimized FP8 kernels (arch 8.0/8.9/9.0/9.0a) |
| **mbridge**                | latest (ISEEKYAN fork)               | Installed with `--no-deps`                  |
| **torch_memory_saver**     | latest                               | Memory optimization utility                 |
| **ray[default]**           | latest                               | Distributed computing                       |
| **sglang-router**          | latest                               | Force reinstalled                           |
| **Additional packages**    | httpx[http2], wandb, pylatexenc, blobfile, accelerate, mcp[cli] | Various utilities |


### Example Usage

After starting the container, you can directly test your environment:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "import transformer_engine.pytorch as te; print('TE version:', te.__version__)"
python -c "import flash_attn; print('FlashAttention version:', flash_attn.__version__)"
```

### Notes

- The container includes `nvtop` for GPU monitoring
- SGLang patches are applied during the build process and verified for conflicts
- The default working directory is `/root/`
- Megatron-LM is installed in editable mode at `/root/Megatron-LM`
- Flash Attention 3 (Hopper) interface is available at `flash_attn_3.flash_attn_interface`

### ⚠️ Important Version Requirements

**TransformerEngine v2.4.0 (commit 3cd6870)**  
We specifically use TransformerEngine v2.4.0. **Higher versions are known to cause issues:**
- Precision errors in FP8 training
- Illegal memory access errors on H100 GPUs

Do not upgrade TransformerEngine beyond v2.4.0 unless you have thoroughly tested it with your training workload.

---

## Acknowledgements

This Docker build configuration is largely inspired by and based on the excellent work from the [**slime**](https://github.com/THUDM/slime) project by THUDM. slime is an LLM post-training framework for RL scaling that powers GLM-4.5 and GLM-4.6. 

**About slime:**
- slime supports training for nearly all models compatible with Megatron-LM, not just GLM series models
- We are actively collaborating with the slime community to achieve fully training-inference consistent FP8 RL training
- The latest slime Docker images typically support all training recipes in this repository

We are grateful to the slime team for their contributions to the open-source community.

If you encounter any issues with the SGLang patches or the Docker build process, please don't hesitate to [open an issue](https://github.com/InfiXAI/InfiR2/issues) in our repository. We are committed to addressing and fixing any problems promptly.

