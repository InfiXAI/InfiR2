# 🧱 Docker Image: TransformerEngine FP8 (CUDA 12.9)

This Docker image provides a **TransformerEngine FP8-optimized** environment built on **SGLang (CUDA 12.9)** and **Megatron-LM**.  
It is designed for efficient large-scale model fine-tuning and inference, especially on NVIDIA Hopper architecture (H100).


### Image Overview

**Base:** `infix/sglang:cu129-latest`  
**Dockerfile:** `Dockerfile.te_fp8.cu129`  
**CUDA:** `12.9`    
**PyTorch:** inherited from `sglang:cu129-latest`  
**Main Components:**
- [SGLang](https://github.com/sgl-project/sglang)  
- [Megatron-LM (InfiXAI fork)](https://github.com/InfiXAI/Megatron-LM/tree/v0.0.1-fp8)  
- [TransformerEngine (NVIDIA)](https://github.com/NVIDIA/TransformerEngine)  
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)  
- [apex (NVIDIA)](https://github.com/NVIDIA/apex)

---

### Build Instructions

You can build the image directly from the root of the repository:

```bash
docker build --no-cache \
    --file docker/Dockerfile.te_fp8.cu129 \
    --build-arg HTTP_PROXY="$http_proxy" \
    --build-arg HTTPS_PROXY="$https_proxy" \
    --build-arg NO_PROXY="localhost,127.0.0.1" \
    --build-arg SGLANG_VERSION=${SGLANG_VERSION:-v0.5.0rc0-cu129} \
    --build-arg MEGATRON_COMMIT=${MEGATRON_COMMIT:-main} \
    -t infix/te-fp8:cu129 .
````

 **Notes:**

* `SGLANG_VERSION` – SGLang release tag or commit hash (default: `v0.5.0rc0-cu129`).
* `MEGATRON_COMMIT` – Megatron-LM commit to check out before patching.
* Use `--no-cache` to force rebuild all layers if updating dependencies.


### Run the Container

To launch the container with GPU and FP8 support:

```bash
docker run -it --rm \
    --gpus all \
    --shm-size=128g \
    -v $(pwd):/workspace \
    infix/te-fp8:cu129 bash
```

This will open a shell inside the FP8-optimized environment.

If you want to run interactively with **Ray** or **SGLang router**, you can use:

```bash
docker run -it --rm \
    --gpus all \
    -p 8080:8080 \
    -e RAY_memory_monitor_refresh_ms=0 \
    infix/te-fp8:cu129 bash
```

Then inside the container:

```bash
sglang-router serve --host 0.0.0.0 --port 8080
```

---

### Key Installed Components

| Component                                                          | Version / Commit                     | Notes                           |
| ------------------------------------------------------------------ | ------------------------------------ | ------------------------------- |
| **SGLang**                                                         | `${SGLANG_VERSION}`                  | Base framework                  |
| **Megatron-LM**                                                    | `v0.0.1-fp8`                         | InfiXAI fork with FP8 patch     |
| **TransformerEngine**                                              | `3cd6870`                            | Installed from source           |
| **FlashAttention**                                                 | `v2.7.4.post1` + custom Hopper build |                                 |
| **Apex**                                                           | latest master                        | CUDA & C++ extensions enabled   |
| **Grouped GEMM**                                                   | `v1.1.4`                             | Optimized FP8 kernels           |
| **mbridge**, **torch_memory_saver**, **ray**, **wandb**, **httpx** | latest                               | Utility and orchestration tools |


### Example Usage

After starting the container, you can directly test your environment:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "import transformer_engine.pytorch as te; print('TE version:', te.__version__)"
```

Or launch your FP8 training:

```bash
cd /workspace/Megatron-LM
torchrun --nproc_per_node=8 pretrain_gpt.py --config configs/fp8_example.yaml
```

