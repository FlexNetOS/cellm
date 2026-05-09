#!/usr/bin/env python3
"""
Load z-lab/gemma-4-26B-A4B-it-DFlash safetensors into MLX and run a
forward pass through the draft transformer layers (standalone demo).

NOTE: This is a DFlash *draft* model. In production it is paired with
the target model (google/gemma-4-26B-A4B-it) for speculative decoding.
The draft cannot produce meaningful text alone because it borrows the
target's embedding table and LM head.  This script demonstrates loading
the weights and running the draft's internal transformer block for
verification / profiling.
"""

import argparse
import json
import math
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from safetensors.torch import load_file


def load_weights(path: str):
    """Load all safetensors in `path` into a dict of MLX arrays."""
    import torch
    weights = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        print(f"Loading {f.name} ...")
        pt_weights = load_file(str(f), device="cpu")
        for key, tensor in pt_weights.items():
            # Convert torch tensor to mlx array
            if tensor.dtype == torch.bfloat16:
                # mlx bfloat16 support: go through float32 numpy
                arr = tensor.float().numpy()
                weights[key] = mx.array(arr).astype(mx.bfloat16)
            else:
                arr = tensor.numpy()
                weights[key] = mx.array(arr)
    print(f"Loaded {len(weights)} tensors.")
    return weights


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps) * self.weight


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rope):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.rope = rope

    def __call__(self, x_norm, x_ctx, rope_params, cache, w_q, w_k, w_v, w_o, q_norm_w, k_norm_w):
        B, T, D = x_norm.shape
        q = x_norm @ w_q.T
        k = x_ctx @ w_k.T
        v = x_ctx @ w_v.T

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q, *rope_params)
        k = self.rope(k, *rope_params)

        # Simple attention (no KV cache for demo)
        q = q * self.scale
        scores = q @ k.transpose(0, 1, 3, 2)
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        out = scores @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return out @ w_o.T


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate_w, up_w, down_w):
        gate = nn.silu(x @ gate_w.T) * (x @ up_w.T)
        return gate @ down_w.T


class DraftLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rope):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim, rope)
        self.mlp = MLP()

    def __call__(self, x, x_ctx, rope_params, cache, w):
        # w is dict of weight arrays for this layer
        h = x + self.attn(
            w["attn_norm"](x), x_ctx, rope_params, cache,
            w["q_proj"], w["k_proj"], w["v_proj"], w["o_proj"],
            w["q_norm"], w["k_norm"],
        )
        h = h + self.mlp(w["ffn_norm"](h), w["gate_proj"], w["up_proj"], w["down_proj"])
        return h


class DFlashDraftForward(nn.Module):
    """Minimal standalone forward for the draft transformer block."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.num_layers = config["num_hidden_layers"]
        self.intermediate_size = config["intermediate_size"]
        self.rope_theta = config.get("rope_theta", 10000.0)
        self.rope_scaling = config.get("rope_scaling")

        # Build simple RoPE (simplified for demo)
        self.rope = lambda x, offset: self._apply_rope(x, offset)

        self.fc = None
        self.hidden_norm = None
        self.norm = None

    def _apply_rope(self, x, offset):
        # x: [B, H, T, D]
        B, H, T, D = x.shape
        pos = mx.arange(offset, offset + T)
        inv_freq = 1.0 / (self.rope_theta ** (mx.arange(0, D, 2).astype(mx.float32) / D))
        angles = mx.outer(pos.astype(mx.float32), inv_freq)
        cos = mx.repeat(mx.cos(angles).reshape(1, 1, T, D // 2), 2, axis=-1)
        sin = mx.repeat(mx.sin(angles).reshape(1, 1, T, D // 2), 2, axis=-1)
        return x * cos + self._rotate_half(x) * sin

    @staticmethod
    def _rotate_half(x):
        d = x.shape[-1] // 2
        return mx.concatenate([-x[..., d:], x[..., :d]], axis=-1)

    def build_from_weights(self, weights: dict):
        """Populate modules from the raw safetensor weight dict."""
        hs = self.hidden_size

        self.fc = weights.get("fc.weight")
        self.hidden_norm = RMSNorm(hs)
        if "hidden_norm.weight" in weights:
            self.hidden_norm.weight = weights["hidden_norm.weight"]

        self.norm = RMSNorm(hs)
        if "norm.weight" in weights:
            self.norm.weight = weights["norm.weight"]

        self.layer_weights = []
        for i in range(self.num_layers):
            prefix = f"layers.{i}."
            lw = {
                "attn_norm": RMSNorm(hs),
                "q_proj": weights.get(prefix + "self_attn.q_proj.weight"),
                "k_proj": weights.get(prefix + "self_attn.k_proj.weight"),
                "v_proj": weights.get(prefix + "self_attn.v_proj.weight"),
                "o_proj": weights.get(prefix + "self_attn.o_proj.weight"),
                "q_norm": RMSNorm(self.head_dim),
                "k_norm": RMSNorm(self.head_dim),
                "ffn_norm": RMSNorm(hs),
                "gate_proj": weights.get(prefix + "mlp.gate_proj.weight"),
                "up_proj": weights.get(prefix + "mlp.up_proj.weight"),
                "down_proj": weights.get(prefix + "mlp.down_proj.weight"),
            }
            if prefix + "input_layernorm.weight" in weights:
                lw["attn_norm"].weight = weights[prefix + "input_layernorm.weight"]
            if prefix + "post_attention_layernorm.weight" in weights:
                lw["ffn_norm"].weight = weights[prefix + "post_attention_layernorm.weight"]
            if prefix + "self_attn.q_norm.weight" in weights:
                lw["q_norm"].weight = weights[prefix + "self_attn.q_norm.weight"]
            if prefix + "self_attn.k_norm.weight" in weights:
                lw["k_norm"].weight = weights[prefix + "self_attn.k_norm.weight"]
            self.layer_weights.append(lw)

    def _attention(self, x_norm, x_ctx, rope_params, w_q, w_k, w_v, w_o, q_norm, k_norm):
        B, T, D = x_norm.shape
        scale = self.head_dim ** -0.5
        q = x_norm @ w_q.T
        k = x_ctx @ w_k.T
        v = x_ctx @ w_v.T

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q, rope_params[0])
        k = self.rope(k, rope_params[0])

        # Norm q/k
        q = q_norm(q.reshape(-1, self.head_dim)).reshape(q.shape)
        k = k_norm(k.reshape(-1, self.head_dim)).reshape(k.shape)

        # Repeat k/v heads for GQA
        if self.num_kv_heads < self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)

        scores = (q * scale) @ k.transpose(0, 1, 3, 2)
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        out = scores @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return out @ w_o.T

    def __call__(self, target_hidden):
        """
        Standalone draft block forward.
        target_hidden: [B, T, concat_dim]  (from target model layer concat)
        Returns draft hidden states [B, T, hidden_size].
        """
        h_ctx = self.hidden_norm(target_hidden @ self.fc.T)
        # In full DFlash, x would be token embeddings from target model.
        # Here we just use h_ctx as both x and x_ctx for a standalone demo.
        x = h_ctx
        offset = 0
        for lw in self.layer_weights:
            x = x + self._attention(
                lw["attn_norm"](x), h_ctx, (offset, ),
                lw["q_proj"], lw["k_proj"], lw["v_proj"], lw["o_proj"],
                lw["q_norm"], lw["k_norm"],
            )
            gate = nn.silu(lw["ffn_norm"](x) @ lw["gate_proj"].T) * (lw["ffn_norm"](x) @ lw["up_proj"].T)
            x = x + gate @ lw["down_proj"].T
        return self.norm(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="./models/dflash/draft")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    mx.random.seed(args.seed)

    config_path = os.path.join(args.cache_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    print("Config:")
    for k, v in config.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
    print()

    weights = load_weights(args.cache_dir)

    model = DFlashDraftForward(config)
    model.build_from_weights(weights)

    concat_dim = len(config["dflash_config"]["target_layer_ids"]) * config["hidden_size"]
    target_hidden = mx.random.normal((args.batch, args.seq_len, concat_dim)).astype(mx.bfloat16)

    print(f"\nInput target_hidden shape: {target_hidden.shape}")
    print(f"Running forward pass ...")

    mx.eval(target_hidden)
    output = model(target_hidden)
    mx.eval(output)

    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")
    print("\nForward pass succeeded!")


if __name__ == "__main__":
    main()
