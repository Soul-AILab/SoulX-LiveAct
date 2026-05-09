# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Megatron-style tensor-parallel WanModel for SoulX-LiveAct.

This is an alternative to model_memory_sp.py for distributed inference.

Why this file exists
--------------------
model_memory_sp.py uses Ulysses-style sequence parallelism (xfuser
xFuserLongContextAttention with all-to-all). Its KV cache aging path is
hardcoded to sp_world_size=2 because the ConvKV memory compression
(Conv1d kernel=stride=5) needs at least 5 contiguous frames per rank
inside the rank's local cache slice. With 14 cache slots split N ways,
that's only satisfied for N <= 2.

model_memory_tp.py keeps full sequence on every rank and shards the
TRANSFORMER instead — the QKV/O projections of self-attention, the
two FFN linears, and the KV cache itself, all sharded along the
attention-head dimension. The streaming KV cache becomes naturally
distributed (each rank stores num_heads/N heads' worth) and the cache
aging logic from the single-GPU model_memory.py applies verbatim on
each rank's local head shard, because the depthwise Conv1d
(groups=dim) has no cross-channel mixing.

Layers parallelized in this version (Phase 1 + FFN)
---------------------------------------------------
  - Self-attention QKV projections   (output-sharded by heads)
  - Self-attention O projection      (input-sharded, all-reduce SUM)
  - Self-attention KV cache          (head-sharded; depthwise Conv1d
                                      compression slices channels)
  - FFN two Linears                  (intermediate dim sharded,
                                      input-sharded second linear
                                      with all-reduce SUM)

Layers replicated in this version
---------------------------------
  - Text/CLIP cross-attention   (~10% of compute; Phase 2 candidate)
  - Audio cross-attention       (~1% of compute)
  - LayerNorms, modulation, embeddings, head, patch embedding

Supported world sizes: 1, 2, 4, 8 (any divisor of num_heads=40 up to 8).

Numerical equivalence: at world=1 this module is bit-equivalent to
model_memory.py. At world>1 the math is equivalent to single-GPU
modulo floating-point reduction order.

Cache geometry
--------------
KV cache shape per rank: [1, 14*frame_len, num_heads/world, head_dim].
This is DIFFERENT from model_memory_sp.py, which had per-rank shape
[1, 14*frame_len/world, num_heads, head_dim]. The user-facing wiring
in generate.py / demo.py needs to allocate the cache with the
head-sharded shape when --tp is set.
"""

import copy
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from einops import rearrange
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

from .attention import flash_attention, SingleStreamAttention, sdpa_attention, flex_attention
from fp8_gemm import FP8Linear
import logging

try:
    from sageattention import sageattn
    USE_SAGEATTN = True
    logging.info("Using sageattn")
except Exception:
    USE_SAGEATTN = False

__all__ = ['WanModel']


# ───────────────────────────────────────────────────────────────────────────
# TP helpers
# ───────────────────────────────────────────────────────────────────────────

def _tp_world():
    """TP world size; falls back to 1 if no SP group is initialised."""
    try:
        return get_sequence_parallel_world_size()
    except Exception:
        return 1


def _tp_rank():
    try:
        return get_sequence_parallel_rank()
    except Exception:
        return 0


def _tp_group():
    try:
        return get_sp_group().device_group
    except Exception:
        return None


def _all_reduce_sum_(t):
    """In-place SUM all-reduce across the TP group (no-op if world == 1)."""
    if _tp_world() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=_tp_group())
    return t


def _shard_range(full_dim, world, rank):
    """Return (start, end) into a tensor sharded along last dim by world."""
    assert full_dim % world == 0, (
        f"full_dim={full_dim} not divisible by world={world}; "
        f"TP requires num_heads (and ffn_dim) divisible by world_size"
    )
    chunk = full_dim // world
    return rank * chunk, (rank + 1) * chunk


def _slice_linear_out(linear, world, rank):
    """Output-sharded slice for nn.Linear: weight [out/world, in], bias [out/world]."""
    s, e = _shard_range(linear.weight.shape[0], world, rank)
    w = linear.weight[s:e]
    b = linear.bias[s:e] if linear.bias is not None else None
    return w, b


def _slice_linear_in(linear, world, rank):
    """Input-sharded slice for nn.Linear: weight [out, in/world], full bias kept."""
    s, e = _shard_range(linear.weight.shape[1], world, rank)
    w = linear.weight[:, s:e]
    return w, linear.bias


# ───────────────────────────────────────────────────────────────────────────
# Math helpers (unchanged from model_memory.py)
# ───────────────────────────────────────────────────────────────────────────

def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """Apply RoPE to x with shape [B, L, n_local, d]. n_local can be heads/world."""
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = s
        f = int(seq_len // (h * w))
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        freqs_i = freqs_i.to(device=x_i.device)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output)


# ───────────────────────────────────────────────────────────────────────────
# Norms
# ───────────────────────────────────────────────────────────────────────────

class WanRMSNorm(nn.Module):
    """RMSNorm that auto-detects whether input is full-dim or TP-sharded.

    The trained weight is full-dim. If forward() receives a sharded input
    (last dim = full_dim/world), we compute the partial sum-of-squares
    locally, all-reduce SUM across the TP group to get the full sum, then
    divide by full_dim to recover the correct mean. Then we apply the
    rank's slice of the weight.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        last = x.shape[-1]
        if last == self.dim:
            # Full-dim path — bit-equivalent to single-GPU.
            return self._norm_full(x.float()).type_as(x) * self.weight
        # Sharded path — only reachable when TP world > 1.
        world, rank = _tp_world(), _tp_rank()
        assert last * world == self.dim, (
            f"sharded RMSNorm shape mismatch: last={last}, world={world}, dim={self.dim}"
        )
        x_f = x.float()
        partial_ss = x_f.pow(2).sum(dim=-1, keepdim=True)
        _all_reduce_sum_(partial_ss)
        mean = partial_ss / self.dim
        normed = (x_f * torch.rsqrt(mean + self.eps)).type_as(x)
        # Two cases for the weight:
        #   - Pre-sharded (after shard_wan_model_for_tp): weight already has shape [last]
        #   - Runtime path (legacy / no shard helper called): weight has shape [self.dim]
        if self.weight.shape[0] == last:
            return normed * self.weight
        ws, we = _shard_range(self.dim, world, rank)
        return normed * self.weight[ws:we]

    def _norm_full(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        out = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps
        ).to(origin_dtype)
        return out


# ───────────────────────────────────────────────────────────────────────────
# Self-attention (TP-parallelized)
# ───────────────────────────────────────────────────────────────────────────

class WanSelfAttention(nn.Module):
    """Tensor-parallel self-attention with head-sharded streaming KV cache.

    Module structure (parameter names) is identical to model_memory.py so
    `from_pretrained` loads weights without modification. Sharding happens
    at runtime by slicing weight tensors per rank in the forward pass.
    """
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn_mask = None
        self.memory_proj_k = nn.Conv1d(self.dim, self.dim, kernel_size=5, stride=5, groups=self.dim, bias=False)
        self.memory_proj_v = nn.Conv1d(self.dim, self.dim, kernel_size=5, stride=5, groups=self.dim, bias=False)

    def post_init(self, device):
        self.memory_proj_k = nn.Conv1d(self.dim, self.dim, kernel_size=5, stride=5, groups=self.dim, bias=False).to(
            device, dtype=torch.bfloat16)
        self.memory_proj_v = nn.Conv1d(self.dim, self.dim, kernel_size=5, stride=5, groups=self.dim, bias=False).to(
            device, dtype=torch.bfloat16)
        nn.init.constant_(self.memory_proj_k.weight, 1.0 / 5.0)
        nn.init.constant_(self.memory_proj_v.weight, 1.0 / 5.0)

    # --- TP-aware projections ---
    #
    # Two paths supported:
    #   (a) Pre-sharded weights (set up by shard_wan_model_for_tp). Modules q,k,v,o
    #       have rank-local weights (e.g. q.weight.shape[0] == dim/world). FP8Linear
    #       wrapping is fine because the underlying weight is already sized correctly.
    #       Forward just calls the module directly. Detected via self._tp_sharded.
    #   (b) Full weights, runtime slicing. Used when shard_wan_model_for_tp is NOT
    #       called. Pure nn.Linear only — incompatible with FP8 GEMM because
    #       FP8Linear doesn't expose .weight in a sliceable form. This path is the
    #       original Phase-1 implementation kept for backward compat.

    def _tp_proj_qkv(self, x):
        """Output-sharded Q/K/V from full-dim input x. Result is per-rank head-shard."""
        world, rank = _tp_world(), _tp_rank()
        if world == 1:
            return self.q(x), self.k(x), self.v(x)
        if getattr(self, '_tp_sharded', False):
            # Pre-sharded weights: modules already produce [B, S, dim/world]
            return self.q(x), self.k(x), self.v(x)
        # Runtime slicing path (no FP8 compat)
        wq, bq = _slice_linear_out(self.q, world, rank)
        wk, bk = _slice_linear_out(self.k, world, rank)
        wv, bv = _slice_linear_out(self.v, world, rank)
        return F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv)

    def _tp_proj_o(self, x_local):
        """Input-sharded O-projection: returns full-dim [B, S, dim] after all-reduce SUM."""
        world, rank = _tp_world(), _tp_rank()
        if world == 1:
            return self.o(x_local)
        if getattr(self, '_tp_sharded', False):
            # Pre-sharded: o has [dim, dim/world] weight + bias/world. Module call
            # produces partial output with partial bias; AR-sum gives full result.
            out = self.o(x_local)
            _all_reduce_sum_(out)
            return out
        # Runtime slicing path
        wo, bo = _slice_linear_in(self.o, world, rank)
        out = F.linear(x_local, wo, bias=None)  # bias added after AR
        _all_reduce_sum_(out)
        if bo is not None:
            out = out + bo
        return out

    # --- TP-aware depthwise Conv1d compression ---

    def _tp_conv1d_dw(self, conv, x):
        """Run a depthwise Conv1d on the rank's channel slice.

        Two paths:
          (a) Pre-sharded conv (after shard_wan_model_for_tp): the module already
              has rank-local channels and correct groups; just call it.
          (b) Full conv with runtime slice: slice the conv weight at runtime.
        """
        world, rank = _tp_world(), _tp_rank()
        if world == 1 or getattr(self, '_tp_sharded', False):
            return conv(x)
        # Runtime slice path. Depthwise (groups=dim) means no cross-channel mixing,
        # so slicing channels gives bit-identical output to running full conv and
        # selecting the rank's channel range.
        cs, ce = _shard_range(self.dim, world, rank)
        w_local = conv.weight[cs:ce]
        return F.conv1d(
            x, w_local, bias=None,
            stride=conv.stride[0],
            padding=conv.padding[0],
            groups=ce - cs,
        )

    def k_compress(self, k, n_frame=5):
        B, N, H, C = k.shape  # H = num_heads // world
        assert N % n_frame == 0
        T = N // n_frame
        k = k.view(B, N, H * C).transpose(1, 2)  # [B, dim_local, N]
        k = self._tp_conv1d_dw(self.memory_proj_k, k)
        k = k.view(B, H, C, T).permute(0, 3, 1, 2)
        return k

    def v_compress(self, v, n_frame=5):
        # NB: model_memory.py uses memory_proj_k for v_compress too — keeping that behavior.
        B, N, H, C = v.shape
        assert N % n_frame == 0
        T = N // n_frame
        v = v.view(B, N, H * C).transpose(1, 2)
        v = self._tp_conv1d_dw(self.memory_proj_k, v)
        v = v.view(B, H, C, T).permute(0, 3, 1, 2)
        return v

    def kv_mean(self, kv, n_frame=5):
        B, N, H, C = kv.shape
        assert N % n_frame == 0
        T = N // n_frame
        return kv.view(B, T, n_frame, H, C).mean(dim=2)

    def init_kvidx(self, frame_len, world_size):
        """Build kv_idx0 / kv_idx2 for cache slicing.

        For TP, the cache is FULL sequence (not divided by world_size),
        head-sharded instead. So we ignore the world_size arg and use
        full ranges. Kept the signature for API compatibility with
        generate.py's loader code.
        """
        device = f'cuda:{int(os.getenv("RANK", 0))}'
        self.kv_idx0 = torch.tensor(list(range(6 * frame_len)), device=device)
        self.kv_idx2 = torch.tensor(list(range(14 * frame_len)), device=device)

    # --- KV cache I/O (unchanged shape semantics, just heads dim is /world) ---

    def _move_kv_cache_to_device(self, kv_cache, device):
        kv_cache["k"] = kv_cache["k"].to(device=device, non_blocking=True)
        kv_cache["v"] = kv_cache["v"].to(device=device, non_blocking=True)
        if kv_cache.get("k_scale") is not None:
            kv_cache["k_scale"] = kv_cache["k_scale"].to(device=device, non_blocking=True)
        if kv_cache.get("v_scale") is not None:
            kv_cache["v_scale"] = kv_cache["v_scale"].to(device=device, non_blocking=True)

    def _quantize_kv_tensor(self, kv):
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = kv.detach().abs().amax(dim=-1, keepdim=True).to(torch.float32)
        scale = torch.clamp(scale / fp8_max, min=1e-12)
        q_kv = (kv / scale.to(dtype=kv.dtype)).to(torch.float8_e4m3fn)
        return q_kv.contiguous(), scale.contiguous()

    def _dequantize_kv_tensor(self, q_kv, scale, dtype):
        return q_kv.to(dtype=dtype) * scale.to(device=q_kv.device, dtype=dtype)

    def _load_kv_cache(self, kv_cache, device, dtype):
        if kv_cache["offload_cache"]:
            self._move_kv_cache_to_device(kv_cache, device)
        if kv_cache.get("fp8_kv_cache", False):
            k_cache = self._dequantize_kv_tensor(kv_cache["k"], kv_cache["k_scale"], dtype)
            v_cache = self._dequantize_kv_tensor(kv_cache["v"], kv_cache["v_scale"], dtype)
        else:
            if kv_cache["k"].dtype != dtype:
                kv_cache["k"] = kv_cache["k"].to(dtype=dtype)
            if kv_cache["v"].dtype != dtype:
                kv_cache["v"] = kv_cache["v"].to(dtype=dtype)
            k_cache = kv_cache["k"]
            v_cache = kv_cache["v"]
        return k_cache, v_cache

    def _store_kv_cache(self, kv_cache, k_cache, v_cache):
        if kv_cache.get("fp8_kv_cache", False):
            kv_cache["k"], kv_cache["k_scale"] = self._quantize_kv_tensor(k_cache)
            kv_cache["v"], kv_cache["v_scale"] = self._quantize_kv_tensor(v_cache)
        else:
            kv_cache["k"] = k_cache
            kv_cache["v"] = v_cache
        if kv_cache["offload_cache"]:
            self._move_kv_cache_to_device(kv_cache, 'cpu')

    # --- Forward ---

    def forward(self, x, seq_lens, grid_sizes, freqs, kv_cache={},
                start_idx=None, end_idx=None, update_cache=False):
        """TP self-attention.

        Input  x: [B, S, dim]  (full sequence, full dim, replicated on every rank)
        Output  : [B, S, dim]  (full sequence, full dim, identical across ranks)

        Internally Q/K/V live as [B, S, dim/world] head-shards. The cache is
        stored as [1, 14*fs, num_heads/world, head_dim] — head-sharded.
        """
        world = _tp_world()
        n_local = self.num_heads // world
        d = self.head_dim
        b, s = x.shape[:2]

        # 1. QKV projection (output-sharded → per-rank head-shard)
        q, k, v = self._tp_proj_qkv(x)
        # Per-Q/K RMSNorm — TP-aware (handles sharded input via all-reduce)
        q = self.norm_q(q).view(b, s, n_local, d)
        k = self.norm_k(k).view(b, s, n_local, d)
        v = v.view(b, s, n_local, d)

        # 2. Load KV cache (head-sharded shape on every rank)
        k_cache, v_cache = self._load_kv_cache(
            kv_cache, f'cuda:{int(os.getenv("RANK", 0))}', torch.bfloat16
        )

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        current_start_frame = start_idx // frame_seqlen

        # 3. Cache aging — single-GPU code from model_memory.py:271-293,
        #    operating on the rank-local head-shard. The depthwise Conv1d
        #    has groups=dim_local with no cross-channel mixing, so this
        #    is bit-equivalent to running on full heads and selecting
        #    the rank's slice.
        if update_cache:
            if kv_cache["mean_memory"]:
                k_compress, v_compress = self.kv_mean, self.kv_mean
            else:
                k_compress, v_compress = self.k_compress, self.v_compress
            k_cache[:, 2 * frame_seqlen: 3 * frame_seqlen].copy_(
                k_compress(k_cache[:, 2 * frame_seqlen: 7 * frame_seqlen]))
            v_cache[:, 2 * frame_seqlen: 3 * frame_seqlen].copy_(
                v_compress(v_cache[:, 2 * frame_seqlen: 7 * frame_seqlen]))
            k_cache[:, 3 * frame_seqlen: 4 * frame_seqlen].copy_(
                k_compress(k_cache[:, 7 * frame_seqlen: 12 * frame_seqlen]))
            v_cache[:, 3 * frame_seqlen: 4 * frame_seqlen].copy_(
                v_compress(v_cache[:, 7 * frame_seqlen: 12 * frame_seqlen]))
            k_cache[:, 4 * frame_seqlen: 6 * frame_seqlen].copy_(
                k_cache[:, 12 * frame_seqlen: 14 * frame_seqlen])
            v_cache[:, 4 * frame_seqlen: 6 * frame_seqlen].copy_(
                v_cache[:, 12 * frame_seqlen: 14 * frame_seqlen])

        # 4. Write current iter's k/v into the cache.
        if start_idx != 0:
            k_cache[:, 6 * frame_seqlen:] = k
            v_cache[:, 6 * frame_seqlen:] = v
        else:
            k_cache[:, : 6 * frame_seqlen] = k
            v_cache[:, : 6 * frame_seqlen] = v

        # 5. RoPE then local attention. Only this rank's heads.
        roped_query = causal_rope_apply(
            q, grid_sizes, freqs, start_frame=current_start_frame
        ).type_as(v)
        roped_key = causal_rope_apply(
            k_cache, grid_sizes, freqs, start_frame=0
        ).type_as(v)

        if USE_SAGEATTN:
            x_attn = sageattn(
                roped_query,
                roped_key[:, :end_idx, ...],
                v_cache[:, :end_idx, ...],
                tensor_layout="NHD",
                is_causal=False,
            ).type_as(x)
        else:
            x_attn = sdpa_attention(
                q=roped_query,
                k=roped_key[:, :end_idx, ...],
                v=v_cache[:, :end_idx, ...],
                k_lens=seq_lens,
                window_size=self.window_size,
                attn_mask=self.attn_mask,
            ).type_as(x)

        # 6. Persist cache.
        self._store_kv_cache(kv_cache, k_cache, v_cache)

        # 7. O-projection: combine head-shards via input-sharded matmul + AR.
        x_attn = x_attn.flatten(2)  # [B, S, n_local*d] = [B, S, dim_local]
        x_out = self._tp_proj_o(x_attn)  # [B, S, dim], replicated post-AR
        return x_out, None


# ───────────────────────────────────────────────────────────────────────────
# Cross-attention (replicated in Phase 1 — Phase 2 candidate)
# ───────────────────────────────────────────────────────────────────────────

class WanI2VCrossAttention(nn.Module):
    """Text + CLIP image cross-attention, replicated across ranks.

    Each rank computes the full forward pass; the result is identical on
    every rank (deterministic compute on identical inputs), so no
    communication is needed.
    """
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, cross_kv_cache={}):
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        if USE_SAGEATTN:
            img_x = sageattn(q, k_img, v_img, tensor_layout='NHD')
            x = sageattn(q, k, v, tensor_layout='NHD')
        else:
            img_x = sdpa_attention(q, k_img, v_img, k_lens=None)
            x = sdpa_attention(q, k, v, k_lens=context_lens)

        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


# ───────────────────────────────────────────────────────────────────────────
# Block (TP self-attention + replicated cross-attns + TP FFN)
# ───────────────────────────────────────────────────────────────────────────

class WanAttentionBlock(nn.Module):
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 output_dim=768,
                 norm_input_visual=True,
                 class_range=24,
                 class_interval=4):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanI2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

        self.audio_cross_attn = SingleStreamAttention(
            dim=dim,
            encoder_hidden_states_dim=output_dim,
            num_heads=num_heads,
            qk_norm=False,
            qkv_bias=True,
            eps=eps,
            norm_layer=WanRMSNorm,
        )
        self.norm_x = WanLayerNorm(dim, eps, elementwise_affine=True) if norm_input_visual else nn.Identity()

    # --- TP-aware FFN: first Linear output-sharded, second input-sharded with AR ---
    #
    # Two paths analogous to the QKV/O case:
    #   (a) Pre-sharded (set by shard_wan_model_for_tp): ffn[0] outputs ffn_dim/world
    #       and ffn[2] takes ffn_dim/world input with bias/world. Just call the
    #       Sequential and AR the result. FP8-compatible.
    #   (b) Full weights with runtime slice. Backward compat only, no FP8.

    def _tp_ffn(self, x):
        world, rank = _tp_world(), _tp_rank()
        if world == 1:
            return self.ffn(x)
        if getattr(self, '_tp_ffn_sharded', False):
            # Pre-sharded: ffn[0] produces [..., ffn_dim/world], ffn[2] takes that
            # and produces full [..., dim] partial (with partial bias). AR-sum.
            out = self.ffn(x)
            _all_reduce_sum_(out)
            return out
        # Runtime slicing path
        l1, gelu, l2 = self.ffn[0], self.ffn[1], self.ffn[2]
        w1, b1 = _slice_linear_out(l1, world, rank)
        h = F.linear(x, w1, b1)
        h = gelu(h)
        w2, b2 = _slice_linear_in(l2, world, rank)
        out = F.linear(h, w2, bias=None)
        _all_reduce_sum_(out)
        if b2 is not None:
            out = out + b2
        return out

    def forward(
            self,
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
            kv_cache={},
            start_idx=None,
            end_idx=None,
            update_cache=False,
            cross_kv_cache={},
            audio_embedding=None,
            ref_target_masks=None,
            human_num=None,
            skip_audio=False,
    ):
        dtype = x.dtype
        if len(e.shape) == 3:
            e = (self.modulation.to(e.device) + e).chunk(6, dim=1)
        else:
            e = (self.modulation.unsqueeze(-2).to(e.device) + e)[0].chunk(6, dim=0)

        # Self-attention (TP)
        y, _ = self.self_attn(
            (self.norm1(x).float() * (1 + e[1]) + e[0]).type_as(x),
            seq_lens, grid_sizes, freqs,
            kv_cache=kv_cache, start_idx=start_idx, end_idx=end_idx,
            update_cache=update_cache,
        )
        x = x + y * e[2]
        x = x.to(dtype)

        # Text + CLIP cross-attention (replicated)
        x = x + self.cross_attn(self.norm3(x), context, context_lens, cross_kv_cache=cross_kv_cache)

        # Audio cross-attention (replicated)
        if not skip_audio:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            start_f = start_idx // frame_seqlen
            x_a = self.audio_cross_attn(
                self.norm_x(x), encoder_hidden_states=audio_embedding,
                shape=grid_sizes[0], start_f=start_f, USE_SAGEATTN=USE_SAGEATTN,
            )
            if start_f == 0:
                x_a[:, :frame_seqlen] = 0
            x = x + x_a

        # FFN (TP)
        y = self._tp_ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).to(dtype))
        x = x + y * e[5]
        x = x.to(dtype)
        return x


# ───────────────────────────────────────────────────────────────────────────
# Head, MLP projector, Audio projector — unchanged from model_memory.py
# ───────────────────────────────────────────────────────────────────────────

class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        e = (self.modulation.to(e.device) + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        return self.proj(image_embeds)


class AudioProjModel(ModelMixin, ConfigMixin):
    def __init__(
            self,
            seq_len=5,
            seq_len_vf=12,
            blocks=12,
            channels=768,
            intermediate_dim=512,
            output_dim=768,
            context_tokens=32,
            norm_output_audio=False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B, _, _, S, C = audio_embeds.shape
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(batch_size_vf, window_size_vf * blocks_vf * channels_vf)
        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf))
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1)
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(batch_size_c * N_t, C_a)
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))
        context_tokens = self.proj3(audio_embeds_c).reshape(batch_size_c * N_t, self.context_tokens, self.output_dim)
        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)
        return context_tokens


# ───────────────────────────────────────────────────────────────────────────
# Block offload manager — unchanged from model_memory.py
# ───────────────────────────────────────────────────────────────────────────

from torch.utils.checkpoint import checkpoint


class WanBlockOffloadManager:
    def __init__(self, blocks, onload_device, offload_device='cpu'):
        self.blocks = blocks
        self.onload_device = torch.device(onload_device)
        self.offload_device = torch.device(offload_device)
        self.prefetch_stream = torch.cuda.Stream(device=self.onload_device)
        self.compute_slot = 0
        self.prefetch_slot = 1
        self.pending_slots = set()
        self.slot_block_indices = [None, None]
        self.cuda_blocks = nn.ModuleList([
            copy.deepcopy(self.blocks[0]).to(self.onload_device),
            copy.deepcopy(self.blocks[0]).to(self.onload_device),
        ])
        for block in self.blocks:
            block.to(self.offload_device)
            self._pin_module_memory(block)

    def _copy_tensor(self, dst, src):
        dst.copy_(src, non_blocking=True)

    def _pin_tensor(self, tensor):
        if tensor is None or tensor.device.type != 'cpu' or tensor.is_pinned():
            return tensor
        return tensor.pin_memory()

    def _pin_module_memory(self, module):
        for name, param in module.named_parameters(recurse=False):
            if param is not None:
                param.data = self._pin_tensor(param.data)
        for name, buffer in module.named_buffers(recurse=False):
            if buffer is not None:
                module._buffers[name] = self._pin_tensor(buffer)
        if isinstance(module, FP8Linear):
            module._fp16_weight_cpu = self._pin_tensor(module._fp16_weight_cpu)
            module._fp16_bias_cpu = self._pin_tensor(module._fp16_bias_cpu)
        for child in module.children():
            self._pin_module_memory(child)

    def _copy_fp8_linear(self, dst_module, src_module):
        if dst_module.linear is not None and src_module.linear is not None:
            self._copy_module_state(dst_module.linear, src_module.linear)
        if dst_module.bias is not None and src_module.bias is not None:
            self._copy_tensor(dst_module.bias.data, src_module.bias.data)
        dst_module._fp16_weight_cpu = src_module._fp16_weight_cpu
        dst_module._fp16_bias_cpu = src_module._fp16_bias_cpu
        if src_module._fp8_weight is None or src_module._fp8_weight_scale is None:
            dst_module._fp8_weight = None
            dst_module._fp8_weight_scale = None
            dst_module._weight_cache_device = None
            if dst_module._fp16_weight_cpu is not None:
                dst_module.materialize_fp8_weight(self.onload_device)
        else:
            if dst_module._fp8_weight is None or dst_module._fp8_weight.shape != src_module._fp8_weight.shape:
                dst_module._fp8_weight = src_module._fp8_weight.to(device=self.onload_device, non_blocking=True)
            else:
                self._copy_tensor(dst_module._fp8_weight, src_module._fp8_weight)
            if dst_module._fp8_weight_scale is None or dst_module._fp8_weight_scale.shape != src_module._fp8_weight_scale.shape:
                dst_module._fp8_weight_scale = src_module._fp8_weight_scale.to(device=self.onload_device, non_blocking=True)
            else:
                self._copy_tensor(dst_module._fp8_weight_scale, src_module._fp8_weight_scale)
            dst_module._weight_cache_device = dst_module._cached_fp8_device()
        dst_module._last_weight_version = src_module._last_weight_version

    def _copy_module_state(self, dst_module, src_module):
        if isinstance(dst_module, FP8Linear) and isinstance(src_module, FP8Linear):
            self._copy_fp8_linear(dst_module, src_module)
            return
        dst_params = dict(dst_module.named_parameters(recurse=False))
        src_params = dict(src_module.named_parameters(recurse=False))
        for name, dst_param in dst_params.items():
            src_param = src_params.get(name)
            if src_param is not None:
                self._copy_tensor(dst_param.data, src_param.data)
        dst_buffers = dict(dst_module.named_buffers(recurse=False))
        src_buffers = dict(src_module.named_buffers(recurse=False))
        for name, dst_buffer in dst_buffers.items():
            src_buffer = src_buffers.get(name)
            if src_buffer is not None:
                self._copy_tensor(dst_buffer, src_buffer)
        dst_children = dict(dst_module.named_children())
        src_children = dict(src_module.named_children())
        for name, dst_child in dst_children.items():
            src_child = src_children.get(name)
            if src_child is not None:
                self._copy_module_state(dst_child, src_child)

    def _load_slot(self, slot_idx, block_idx, async_transfer=False):
        def copy_block():
            self._copy_module_state(self.cuda_blocks[slot_idx], self.blocks[block_idx])
            self.slot_block_indices[slot_idx] = block_idx
        if async_transfer:
            with torch.cuda.stream(self.prefetch_stream):
                copy_block()
            self.pending_slots.add(slot_idx)
        else:
            copy_block()
            self.pending_slots.discard(slot_idx)

    def _wait_slot(self, slot_idx):
        if slot_idx in self.pending_slots:
            torch.cuda.current_stream(device=self.onload_device).wait_stream(self.prefetch_stream)
            self.pending_slots.discard(slot_idx)

    def get_block(self, block_idx):
        if self.slot_block_indices[self.compute_slot] == block_idx:
            self._wait_slot(self.compute_slot)
        elif self.slot_block_indices[self.prefetch_slot] == block_idx:
            self._wait_slot(self.prefetch_slot)
            self.compute_slot, self.prefetch_slot = self.prefetch_slot, self.compute_slot
        else:
            self._load_slot(self.compute_slot, block_idx, async_transfer=False)
        next_idx = block_idx + 1
        if next_idx < len(self.blocks) and self.slot_block_indices[self.prefetch_slot] != next_idx:
            self.prefetch_stream.wait_stream(torch.cuda.current_stream(device=self.onload_device))
            self._load_slot(self.prefetch_slot, next_idx, async_transfer=True)
        return self.cuda_blocks[self.compute_slot]

    def unload_all(self):
        torch.cuda.current_stream(device=self.onload_device).wait_stream(self.prefetch_stream)
        self.pending_slots.clear()
        self.slot_block_indices = [None, None]


# ───────────────────────────────────────────────────────────────────────────
# WanModel — same forward as model_memory.py (no input chunk, no all_gather)
# ───────────────────────────────────────────────────────────────────────────

class WanModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    Wan diffusion backbone with tensor-parallel self-attention and FFN.

    Forward pass is identical to model_memory.py — the input x is full
    sequence and full hidden dim on every rank, and the output is also
    full sequence and full dim. Internal attention and FFN parallelize
    via head-shard / hidden-shard. No torch.chunk on input, no
    all_gather on output.
    """
    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='i2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 audio_window=5,
                 intermediate_dim=512,
                 output_dim=768,
                 context_tokens=32,
                 vae_scale=4,
                 norm_input_visual=True,
                 norm_output_audio=True,
                 weight_init=True):
        super().__init__()
        assert model_type == 'i2v', 'MultiTalk model requires model_type == "i2v".'
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.gradient_checkpointing = False

        self.norm_output_audio = norm_output_audio
        self.audio_window = audio_window
        self.intermediate_dim = intermediate_dim
        self.vae_scale = vae_scale

        self.return_layers_cosine = False
        self.cos_sims = []
        self.skip_layer = []
        self.block_offload_manager = None
        self.block_offload_enabled = False

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        cross_attn_type = 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps,
                              output_dim=output_dim, norm_input_visual=norm_input_visual)
            for _ in range(num_layers)
        ])

        self.head = Head(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
        else:
            raise NotImplementedError('Not supported model type.')

        self.audio_proj = AudioProjModel(
            seq_len=audio_window,
            seq_len_vf=audio_window + vae_scale - 1,
            intermediate_dim=intermediate_dim,
            output_dim=output_dim,
            context_tokens=context_tokens,
            norm_output_audio=norm_output_audio,
        )

        if weight_init:
            self.init_weights()

    def init_freqs(self):
        d = self.dim // self.num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

    def enable_block_offload(self, onload_device=None, offload_device='cpu'):
        if onload_device is None:
            onload_device = self.patch_embedding.weight.device
        onload_device = torch.device(onload_device)
        if onload_device.type != 'cuda':
            raise ValueError("WanModel block offload requires a CUDA onload device.")
        self.block_offload_manager = WanBlockOffloadManager(
            self.blocks,
            onload_device=onload_device,
            offload_device=offload_device,
        )
        self.block_offload_enabled = True
        torch.cuda.empty_cache()
        return self

    def forward(
            self,
            x,
            t,
            context,
            clip_fea=None,
            y=None,
            audio=None,
            ref_target_masks=None,
            kv_cache={},
            start_idx=None,
            end_idx=None,
            cross_kv_cache={},
            update_cache=False,
            skip_audio=False,
    ):
        assert clip_fea is not None and y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        _, T, H, W = x[0].shape
        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x[0] = x[0].to(context[0].dtype)

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        x = torch.cat(x)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1).to(x.dtype)

        audio_cond = audio.to(device=x.device, dtype=x.dtype)
        first_frame_audio_emb_s = audio_cond[:, :1, ...]
        latter_frame_audio_emb = audio_cond[:, 1:, ...]
        latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale)
        middle_index = self.audio_window // 2
        latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index + 1, ...]
        latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
        latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index + 1, ...]
        latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_frame_audio_emb_s = torch.concat(
            [latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2)
        audio_embedding = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)
        human_num = len(audio_embedding)
        audio_embedding = torch.concat(audio_embedding.split(1), dim=2).to(x.dtype)

        if ref_target_masks is not None:
            ref_target_masks = ref_target_masks.unsqueeze(0)
            token_ref_target_masks = nn.functional.interpolate(ref_target_masks, size=(N_h, N_w), mode='nearest')
            token_ref_target_masks = token_ref_target_masks.squeeze(0)
            token_ref_target_masks = (token_ref_target_masks > 0)
            token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1)
            token_ref_target_masks = token_ref_target_masks.to(x.dtype)
        else:
            token_ref_target_masks = None

        # NOTE: NO torch.chunk(x, ...) — every rank holds the full sequence.
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            audio_embedding=audio_embedding,
            ref_target_masks=token_ref_target_masks,
            human_num=human_num,
            start_idx=start_idx,
            end_idx=end_idx,
            update_cache=update_cache,
        )

        block_offload_manager = self.block_offload_manager if self.block_offload_enabled else None
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block_index, block in enumerate(self.blocks):
                if block_offload_manager is not None:
                    block = block_offload_manager.get_block(block_index)
                if kv_cache.get(block_index) is None: kv_cache[block_index] = {}
                if cross_kv_cache.get(block_index) is None: cross_kv_cache[block_index] = {}
                x = checkpoint(
                    block, x, kv_cache=kv_cache[block_index], cross_kv_cache=cross_kv_cache[block_index],
                    skip_audio=skip_audio, use_reentrant=False, **kwargs
                )
        else:
            for block_index, block in enumerate(self.blocks):
                if block_offload_manager is not None:
                    block = block_offload_manager.get_block(block_index)
                if kv_cache.get(block_index) is None: kv_cache[block_index] = {}
                if cross_kv_cache.get(block_index) is None: cross_kv_cache[block_index] = {}
                x = block(x, kv_cache=kv_cache[block_index], cross_kv_cache=cross_kv_cache[block_index],
                          skip_audio=skip_audio, **kwargs)

        x = self.head(x, e)
        # NOTE: no get_sp_group().all_gather — output is already replicated on every rank.
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.head.head.weight)


# ───────────────────────────────────────────────────────────────────────────
# Shard helper: pre-slice weights so FP8 GEMM can wrap them transparently.
# ───────────────────────────────────────────────────────────────────────────
#
# Why this exists
# ---------------
# enable_fp8_gemm() (in fp8_gemm.py) wraps each nn.Linear with FP8Linear,
# quantizing the weight to FP8 internally. After wrapping, the original
# nn.Linear is gone (in 'discard' mode) — there's no .weight attribute we
# can slice at runtime in the forward pass. So the runtime-slicing path
# in _tp_proj_qkv/_tp_proj_o/_tp_ffn falls through to "call module
# directly" and silently runs at full rank, defeating the TP speedup.
#
# The fix: pre-slice each TP-relevant Linear's weight BEFORE FP8 wrapping.
# Each rank's nn.Linear ends up with rank-local weight; FP8Linear wraps
# that and quantizes the smaller weight; forward calls the wrapped module
# directly (no runtime slicing) and the output already has the right
# per-rank shape. _all_reduce_sum_ at the right places combines partials
# for input-sharded layers.
#
# After calling this, the model carries:
#   self_attn._tp_sharded = True       — switches QKV/O forward to pre-sharded path
#   block._tp_ffn_sharded  = True      — switches FFN forward to pre-sharded path
# These flags are read in WanSelfAttention._tp_proj_qkv / _tp_proj_o and
# WanAttentionBlock._tp_ffn.
#
# Order of operations in generate.py / demo.py:
#   1. WanModel.from_pretrained(...)
#   2. model.to(dtype=torch.bfloat16)
#   3. shard_wan_model_for_tp(model)            ← here
#   4. enable_fp8_gemm(model, ...)              ← optional, after shard
#   5. model.to(device); model.eval(); torch.compile(...)


def _shard_param_(linear, slice_fn, *, divide_bias_by=None):
    """Replace `linear.weight` (and bias) with rank-local slices.

    `slice_fn` takes the full weight tensor and returns the rank's slice.
    If `divide_bias_by` is set, the bias is divided by that value (used
    for input-sharded layers where each rank's partial output carries
    bias/world; AR-sum of N partials reconstructs the full bias).
    """
    new_w = slice_fn(linear.weight.data).clone().contiguous()
    linear.weight = nn.Parameter(new_w, requires_grad=linear.weight.requires_grad)
    if linear.bias is not None:
        if divide_bias_by is not None:
            new_b = (linear.bias.data / divide_bias_by).clone().contiguous()
        else:
            new_b = slice_fn(linear.bias.data).clone().contiguous()
        linear.bias = nn.Parameter(new_b, requires_grad=linear.bias.requires_grad)
    # Linear.in_features / out_features are read by some external libs; keep them
    # synced with the actual weight shape so e.g. .extra_repr is informative.
    out_features, in_features = linear.weight.shape
    linear.in_features = in_features
    linear.out_features = out_features


def _shard_linear_out_(linear, world, rank):
    """Output-shard an nn.Linear: weight rows [out/world, in], bias rows [out/world]."""
    s, e = _shard_range(linear.weight.shape[0], world, rank)
    _shard_param_(linear, lambda t: t[s:e])


def _shard_linear_in_(linear, world, rank):
    """Input-shard an nn.Linear: weight cols [out, in/world], bias /= world."""
    s, e = _shard_range(linear.weight.shape[1], world, rank)
    _shard_param_(linear, lambda t: t[:, s:e] if t.dim() == 2 else t,
                  divide_bias_by=world)


def _shard_rmsnorm_(norm, world, rank):
    """Slice WanRMSNorm.weight to rank-local. self.dim stays full so the all-reduce
    trick in WanRMSNorm.forward computes the correct global mean."""
    if not isinstance(norm, WanRMSNorm):
        return  # nn.Identity case (qk_norm=False)
    s, e = _shard_range(norm.weight.shape[0], world, rank)
    new_w = norm.weight.data[s:e].clone().contiguous()
    norm.weight = nn.Parameter(new_w, requires_grad=norm.weight.requires_grad)


def _shard_conv1d_dw_(parent, attr, world, rank):
    """Replace a depthwise nn.Conv1d on `parent.<attr>` with a channel-sharded version."""
    old = getattr(parent, attr)
    full = old.weight.shape[0]
    s, e = _shard_range(full, world, rank)
    new_ch = e - s
    new = nn.Conv1d(
        new_ch, new_ch,
        kernel_size=old.kernel_size[0],
        stride=old.stride[0],
        padding=old.padding[0],
        groups=new_ch,
        bias=(old.bias is not None),
    )
    new.weight = nn.Parameter(old.weight.data[s:e].clone().contiguous(),
                              requires_grad=old.weight.requires_grad)
    if old.bias is not None:
        new.bias = nn.Parameter(old.bias.data[s:e].clone().contiguous(),
                                requires_grad=old.bias.requires_grad)
    new.to(device=old.weight.device, dtype=old.weight.dtype)
    setattr(parent, attr, new)


def _shard_block(block, world, rank):
    """Apply TP slicing to one WanAttentionBlock in-place."""
    sa = block.self_attn

    # Output-sharded q, k, v
    _shard_linear_out_(sa.q, world, rank)
    _shard_linear_out_(sa.k, world, rank)
    _shard_linear_out_(sa.v, world, rank)
    # Output-sharded RMSNorm weights, aligned with q/k slicing
    _shard_rmsnorm_(sa.norm_q, world, rank)
    _shard_rmsnorm_(sa.norm_k, world, rank)
    # Input-sharded o (bias /= world)
    _shard_linear_in_(sa.o, world, rank)
    # Channel-sharded depthwise Conv1d (memory compression, depthwise => no mixing)
    _shard_conv1d_dw_(sa, 'memory_proj_k', world, rank)
    _shard_conv1d_dw_(sa, 'memory_proj_v', world, rank)

    sa._tp_sharded = True

    # Output-sharded FFN[0] and input-sharded FFN[2]
    if isinstance(block.ffn, nn.Sequential) and len(block.ffn) >= 3:
        _shard_linear_out_(block.ffn[0], world, rank)
        _shard_linear_in_(block.ffn[2], world, rank)
        block._tp_ffn_sharded = True


def shard_wan_model_for_tp(model):
    """Pre-slice TP-relevant weights for the current rank.

    Must be called AFTER WanModel.from_pretrained() and the model has been
    moved to its compute device, but BEFORE enable_fp8_gemm() if FP8 is
    being used.

    No-op when world_size == 1.

    Sharded layers per WanAttentionBlock:
      self_attn.{q, k, v}        output-sharded (rows; bias rows)
      self_attn.o                input-sharded  (cols; bias /= world)
      self_attn.norm_q.weight    sliced to match q's output shard
      self_attn.norm_k.weight    sliced to match k's output shard
      self_attn.memory_proj_{k,v}   depthwise Conv1d, channel-sharded
      ffn[0]                     output-sharded (rows; bias rows)
      ffn[2]                     input-sharded  (cols; bias /= world)

    Marks self_attn._tp_sharded = True and block._tp_ffn_sharded = True so
    that the forward methods take the pre-sharded code path rather than
    the runtime-slicing fallback. The two paths are numerically equivalent;
    the pre-sharded path is FP8-compatible and slightly faster (no
    per-forward weight slicing).
    """
    world = _tp_world()
    if world == 1:
        return model
    rank = _tp_rank()
    assert model.num_heads % world == 0, (
        f"shard_wan_model_for_tp: num_heads={model.num_heads} not divisible by world={world}"
    )
    assert model.ffn_dim % world == 0, (
        f"shard_wan_model_for_tp: ffn_dim={model.ffn_dim} not divisible by world={world}"
    )
    for block in model.blocks:
        _shard_block(block, world, rank)
    model._tp_sharded = True
    return model
