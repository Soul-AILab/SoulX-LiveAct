# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tiny utility to enable vLLM-style FP4 GEMM (W4A4 NVFP4) for arbitrary PyTorch
models on Blackwell (SM100+) GPUs.

What it does
- Replaces nn.Linear modules with a drop-in module that:
  - quantizes activations dynamically per forward call (RTN, no calibration)
  - quantizes weights lazily on first CUDA forward (and caches them)
  - dispatches GEMM via vLLM's cutlass_scaled_fp4_mm
  - supports VLLM_USE_NVFP4_CT_EMULATIONS=1 for software-emulated FP4 on
    pre-Blackwell GPUs (slow, but bit-faithful — useful for quality validation)

Notes
- CUDA-only fast path; CPU and unsupported cases automatically fall back to
  the original nn.Linear.
- Last-dim of activations/weights must be a multiple of 16 (NVFP4 block size).
- Output of vLLM FP4 GEMM is fp16/bf16. fp32 inputs are cast to bf16 (or fall
  back to nn.Linear if cast_inputs=False).
- Mirrors the structure of fp8_gemm.py for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Literal

import torch
import torch.nn as nn


# NVFP4 dynamic-range constants. Per-tensor global_scale is set so that
# `(E4M3_MAX * E2M1_MAX) / amax(tensor)` saturates the FP8 block-scale range.
_FLOAT8_E4M3_MAX = 448.0
_FLOAT4_E2M1_MAX = 6.0
_FP4_BLOCK_SIZE = 16


@dataclass(frozen=True)
class FP4GemmOptions:
    # If True, non-fp16/bf16 inputs will be cast to bf16 for the FP4 GEMM path.
    # If False, non-fp16/bf16 inputs will fall back to the original nn.Linear.
    cast_inputs: bool = True

    # If True, the output will be cast back to the original input dtype when
    # we cast inputs for the fast path.
    cast_output_back: bool = True

    # What to do with the original (FP16/BF16) weights after wrapping.
    fp16_weight_storage: Literal["keep", "cpu_offload", "discard"] = "discard"

    # If True, try to quantize weights immediately while wrapping (only works
    # when the original nn.Linear weights are already on CUDA).
    materialize_fp4_on_wrap: bool = True


def _is_emulation_enabled() -> bool:
    try:
        import vllm.envs as envs
        return bool(envs.VLLM_USE_NVFP4_CT_EMULATIONS)
    except Exception:
        return False


def _ensure_fp4_supported() -> None:
    """Raise RuntimeError if FP4 path cannot run on this GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("FP4 GEMM requires CUDA")
    if _is_emulation_enabled():
        return  # emulation runs on any capability
    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    if sm < 100:
        raise RuntimeError(
            f"FP4 GEMM requires SM100+ (Blackwell), got SM{sm}. "
            "Set VLLM_USE_NVFP4_CT_EMULATIONS=1 for software emulation."
        )


def _compute_global_scale(amax: torch.Tensor) -> torch.Tensor:
    """NVFP4 per-tensor global scale: saturates FP8 block-scale dynamic range.

    Formula: gs = (E4M3_MAX * E2M1_MAX) / max(amax, eps).
    Returns fp32 scalar tensor.
    """
    eps = torch.tensor(1e-12, device=amax.device, dtype=torch.float32)
    amax_f32 = amax.to(torch.float32)
    return (_FLOAT8_E4M3_MAX * _FLOAT4_E2M1_MAX) / torch.maximum(amax_f32, eps)


class FP4Linear(nn.Module):
    """Drop-in nn.Linear replacement using vLLM NVFP4 GEMM."""

    def __init__(self, linear: nn.Linear, *, options: FP4GemmOptions):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(linear)}")
        if options.fp16_weight_storage not in ("keep", "cpu_offload", "discard"):
            raise ValueError(
                f"invalid fp16_weight_storage={options.fp16_weight_storage!r}"
            )
        if options.fp16_weight_storage == "discard" and not options.cast_inputs:
            raise ValueError(
                "fp16_weight_storage='discard' requires cast_inputs=True"
            )

        # NVFP4 requires K % 16 == 0. If not, fall back to nn.Linear.
        if linear.in_features % _FP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"FP4Linear requires in_features % {_FP4_BLOCK_SIZE} == 0, "
                f"got in_features={linear.in_features}"
            )

        self.linear: Optional[nn.Linear] = (
            linear if options.fp16_weight_storage == "keep" else None
        )
        self.options = options

        # Optional CPU copies for fallback.
        self._fp16_weight_cpu: Optional[torch.Tensor] = None
        self._fp16_bias_cpu: Optional[torch.Tensor] = None

        # Bias for fast path when not keeping the original Linear.
        self.bias: Optional[nn.Parameter] = None
        if options.fp16_weight_storage != "keep":
            self.bias = (
                nn.Parameter(linear.bias.detach().clone())
                if linear.bias is not None else None
            )
            self._fp16_weight_cpu = linear.weight.detach().to(
                device="cpu", dtype=torch.bfloat16
            ).contiguous()
            if linear.bias is not None:
                self._fp16_bias_cpu = linear.bias.detach().to(
                    device="cpu", dtype=torch.bfloat16
                ).contiguous()

        # Weight cache: packed FP4 [N, K/2] uint8 + swizzled FP8E4M3 block
        # scales + per-tensor FP32 global scale.
        self.register_buffer("_fp4_weight", None, persistent=False)
        self.register_buffer("_fp4_weight_scale", None, persistent=False)
        self.register_buffer("_fp4_weight_global_scale", None, persistent=False)
        self._weight_cache_device: Optional[torch.device] = None
        self._last_weight_version: Optional[int] = None

        self._emulation = _is_emulation_enabled()

        # CUDA-only quant ops.
        from vllm import _custom_ops as ops
        self._ops = ops

        if self._emulation:
            from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501
                run_nvfp4_emulations,
            )
            self._run_nvfp4_emulations = run_nvfp4_emulations
        else:
            self._run_nvfp4_emulations = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, options: FP4GemmOptions) -> "FP4Linear":
        return cls(linear, options=options)

    def invalidate_weight_cache(self) -> None:
        self._fp4_weight = None
        self._fp4_weight_scale = None
        self._fp4_weight_global_scale = None
        self._weight_cache_device = None
        self._last_weight_version = None

    def _cached_fp4_device(self) -> Optional[torch.device]:
        if (self._fp4_weight is None or self._fp4_weight_scale is None
                or self._fp4_weight_global_scale is None):
            return None
        if self._fp4_weight.device != self._fp4_weight_scale.device:
            return None
        return self._fp4_weight.device

    def materialize_fp4_weight(self, device: torch.device) -> None:
        self._maybe_requantize_weight(device)

    def _maybe_requantize_weight(self, device: torch.device) -> None:
        cache_device = self._cached_fp4_device()
        version: Optional[int] = None
        if self.linear is not None:
            weight = self.linear.weight
            v = getattr(weight, "_version", None)
            version = v if isinstance(v, int) else None
            if (self._fp4_weight is not None
                    and cache_device == device
                    and (version is None or version == self._last_weight_version)):
                return
        else:
            if self._fp4_weight is not None and cache_device == device:
                return

        if self.linear is not None:
            w_src = self.linear.weight.detach()
        elif self._fp16_weight_cpu is not None:
            w_src = self._fp16_weight_cpu
        else:
            raise RuntimeError(
                "FP4Linear has no FP16 weight source available to (re)quantize."
            )

        w_n_k = w_src.to(device=device, dtype=torch.bfloat16,
                         non_blocking=True).contiguous()  # [N, K]

        # Per-tensor global scale from amax. Single scalar that gates the
        # FP8 block-scale dynamic range used inside scaled_fp4_quant.
        w_amax = w_n_k.abs().amax()
        w_global_scale = _compute_global_scale(w_amax)

        # scaled_fp4_quant returns scales already in the swizzled layout
        # required by cutlass_scaled_fp4_mm and run_nvfp4_emulations.
        qweight, w_scale_swizzled = self._ops.scaled_fp4_quant(
            w_n_k, w_global_scale
        )

        self._fp4_weight = qweight                       # [N, K/2] uint8
        self._fp4_weight_scale = w_scale_swizzled        # fp8e4m3, swizzled
        self._fp4_weight_global_scale = w_global_scale   # fp32 scalar
        self._weight_cache_device = self._cached_fp4_device()
        self._last_weight_version = version

        if self.options.fp16_weight_storage == "discard":
            self._fp16_weight_cpu = None
            self._fp16_bias_cpu = None

    def _fallback(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear is not None:
            return self.linear(x)
        if self._fp16_weight_cpu is not None:
            w = self._fp16_weight_cpu.to(device=x.device, dtype=x.dtype)
            b = self._fp16_bias_cpu
            b = b.to(device=x.device, dtype=x.dtype) if b is not None else None
            return torch.nn.functional.linear(x, w, b)
        raise RuntimeError(
            "FP4Linear has no fallback weights (fp16_weight_storage='discard')"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CPU / non-CUDA fall back.
        if not x.is_cuda:
            return self._fallback(x)

        in_dtype = x.dtype
        if in_dtype not in (torch.float16, torch.bfloat16):
            if not self.options.cast_inputs:
                return self._fallback(x)
            x_fp = x.to(torch.bfloat16)
            out_dtype = torch.bfloat16
        else:
            x_fp = x
            out_dtype = in_dtype

        # NVFP4 requires K % 16 == 0; we already validated at __init__.
        # Flatten leading dims to [M, K].
        orig_shape = x_fp.shape
        x_2d = x_fp.reshape(-1, orig_shape[-1]).contiguous()

        self._maybe_requantize_weight(x_2d.device)

        # Dynamic per-batch activation global scale (RTN, no calibration).
        x_amax = x_2d.abs().amax()
        x_global_scale = _compute_global_scale(x_amax)

        if self._emulation:
            # Emulation path takes the raw bf16 input and quantizes internally.
            assert self._run_nvfp4_emulations is not None
            out = self._run_nvfp4_emulations(
                x=x_2d,
                input_global_scale=x_global_scale,
                weight=self._fp4_weight,
                weight_scale_swizzled=self._fp4_weight_scale,
                weight_global_scale=self._fp4_weight_global_scale,
            )
        else:
            # Native cutlass path: explicit activation quant + scaled MM.
            qx, x_scale_swizzled = self._ops.scaled_fp4_quant(
                x_2d, x_global_scale
            )
            alpha = 1.0 / (
                x_global_scale * self._fp4_weight_global_scale
            )
            out = self._ops.cutlass_scaled_fp4_mm(
                qx,
                self._fp4_weight,
                x_scale_swizzled,
                self._fp4_weight_scale,
                alpha,
                out_dtype,
            )

        # Bias is applied post-GEMM (NVFP4 cutlass kernel has no fused bias).
        if self.linear is not None:
            bias = self.linear.bias
        else:
            bias = self.bias
        if bias is not None:
            if bias.device != out.device:
                bias = bias.to(device=out.device, non_blocking=True)
            if bias.dtype != out.dtype:
                bias = bias.to(dtype=out.dtype)
            out = out + bias

        out_shape = orig_shape[:-1] + (out.shape[-1],)
        out = out.reshape(out_shape)

        if (self.options.cast_inputs and self.options.cast_output_back
                and out.dtype != in_dtype):
            return out.to(in_dtype)
        return out


def enable_fp4_gemm(
    model: nn.Module,
    *,
    options: FP4GemmOptions = FP4GemmOptions(),
    module_filter: Optional[Callable[[str, nn.Module], bool]] = None,
    inplace: bool = True,
) -> nn.Module:
    """
    Replace nn.Linear modules in `model` with FP4Linear to accelerate GEMMs
    on Blackwell (SM100+) GPUs.

    Raises RuntimeError if the GPU does not support FP4 and emulation is not
    enabled — caller is expected to catch and fall back to FP8 / BF16.

    Args:
        model: Any torch.nn.Module.
        options: FP4GemmOptions controlling casting / fallback behavior.
        module_filter: Optional predicate (name, module) -> bool.
        inplace: If True, modifies model in-place and returns it.
    """
    _ensure_fp4_supported()

    if not inplace:
        import copy
        model = copy.deepcopy(model)

    def should_wrap(name: str, m: nn.Module) -> bool:
        if not isinstance(m, nn.Linear):
            return False
        if m.in_features % _FP4_BLOCK_SIZE != 0:
            return False  # incompatible shape — leave as nn.Linear
        if module_filter is None:
            return True
        return bool(module_filter(name, m))

    def _recurse(prefix: str, parent: nn.Module) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if should_wrap(full_name, child):
                fp4_mod = FP4Linear.from_linear(child, options=options)
                if (options.materialize_fp4_on_wrap and child.weight.is_cuda):
                    fp4_mod.materialize_fp4_weight(child.weight.device)
                setattr(parent, child_name, fp4_mod)
            else:
                _recurse(full_name, child)

    _recurse("", model)
    return model
