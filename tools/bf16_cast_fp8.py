import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file
import triton
import triton.language as tl
from typing import Tuple

import math

fp8_dtype = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_dtype).max
fp8_min = torch.finfo(fp8_dtype).min

bf16_max = torch.finfo(torch.bfloat16).max


# IType  OType
# BF16   E4M3

def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
            x: the dividend.
            y: the divisor.

    Returns:
            The result of the ceiling division.
    """
    return (x + y - 1) // y


@triton.jit
def _blockwise_cast_to_fp8_triton(
    X,
    Y,
    S,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    M,
    N,
    eps,
    fp8_min,
    fp8_max,
    bf16_max,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 128,
    force_pow_2_scale: tl.constexpr = True,
    INF : tl.constexpr= 3.4e38,
):
    pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=1), tl.int64)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = off_m < M
    mask_n = off_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(x)), eps)

    if (_absmax > INF) | (_absmax == 0.0) | (_absmax != _absmax):
        x_s = 1.0
    else:
        x_s = _absmax / fp8_max
        if x_s > INF:
            x_s = bf16_max

        if force_pow_2_scale:
            x_s = tl.exp2(tl.ceil(tl.log2(x_s)))

    s_inv = 1.0 / x_s
    y_q = tl.clamp(x * s_inv, fp8_min, fp8_max).to(Y.dtype.element_ty)

    tl.store(Y + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, y_q, mask=mask)
    tl.store(S + pid_m * stride_sm + pid_n * stride_sn, x_s)


def blockwise_cast_to_fp8_triton(x: torch.Tensor, force_pow_2_scale, block_size=None) -> Tuple[torch.Tensor, torch.Tensor]:
    BLOCK_M, BLOCK_N = 128, 128
    if block_size:
        BLOCK_M, BLOCK_N = block_size[0], block_size[1]
    M, N = x.shape
    y = torch.empty(M, N, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N), dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    if x.is_contiguous():
        kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 8, "num_stages": 2}
    else:
        kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 1, "num_stages": 4}

    kwargs["force_pow_2_scale"] = force_pow_2_scale

    _blockwise_cast_to_fp8_triton[grid](
        x, y, s, *x.stride(), *y.stride(), *s.stride(), M, N, 1e-10, fp8_min, fp8_max, bf16_max, **kwargs
    )
    return y, s

def main(args):

    bf16_path = args.input_bf16_hf_path
    fp8_path = args.output_fp8_hf_path

    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(fp8_path, exist_ok=True)

    # Load original model index
    model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"].copy()

    # Process safetensors files
    safetensor_files = glob(os.path.join(bf16_path, "*.safetensors"))
    safetensor_files.sort()

    loaded_files = {}
    fp8_converted = set()

    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for name, tensor in current_state_dict.items():
            if name.endswith("_scale_inv") or name in ["model.embed_tokens.weight", "lm_head.weight"]:
                new_state_dict[name] = tensor
                continue

            # Convert BF16 weights to FP8
            if tensor.dtype == torch.bfloat16 and tensor.ndim == 2:
                quantized, scale_inv = blockwise_cast_to_fp8_triton(tensor, args.force_pow_2_scale)
                new_state_dict[name] = quantized
                new_state_dict[f"{name}_scale_inv"] = scale_inv.to(torch.float16)
                fp8_converted.add(name)
            else:
                new_state_dict[name] = tensor

        # Save converted file
        new_file_path = os.path.join(fp8_path, file_name)
        save_file(new_state_dict, new_file_path)

        # Update weight map
        for name in new_state_dict:
            if name not in weight_map:
                weight_map[name] = file_name

        # Memory management
        if len(loaded_files) > 2:
            del loaded_files[next(iter(loaded_files))]
            torch.cuda.empty_cache()

    # Update model index
    new_index = {
        "metadata": model_index.get("metadata", {}),
        "weight_map": {k: v for k, v in weight_map.items() if not k.endswith("_scale_inv")}
    }

    # Add scale_inv entries
    for name in fp8_converted:
        scale_name = f"{name}_scale_inv"
        new_index["weight_map"][scale_name] = weight_map[name]

    index_path = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(new_index, f, indent=2)

    # Copy tokenizer_config.json, tokenizer.json, config.json, and special_tokens_map.json, generation_config.json
    for file_name in ["tokenizer_config.json", "tokenizer.json", "config.json", "special_tokens_map.json", "generation_config.json"]:
        src_path = os.path.join(bf16_path, file_name)
        if os.path.exists(src_path):
            dst_path = os.path.join(fp8_path, file_name)
            with open(src_path, "r") as src_file:
                with open(dst_path, "w") as dst_file:
                    dst_file.write(src_file.read())

    # Process config.json, change dtype to float8
    config_path = os.path.join(fp8_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

        config["quantization_config"] = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128,128],
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-bf16-hf-path", type=str, required=True)
    parser.add_argument("--output-fp8-hf-path", type=str, required=True)
    parser.add_argument("--force-pow-2-scale",type=bool, default=True)
    args = parser.parse_args()
    main(args)

