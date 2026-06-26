#!/usr/bin/env python3
"""
Convert LFM2.5 HuggingFace model (bfloat16 safetensors) to .cellm format.
Uses PyTorch to load bfloat16 safetensors, then converts to f16 for .cellm.
"""

import json
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def write_cellm(
    output_path: Path, header: dict, tensors_bytes: dict, tensors_shape: dict
):
    """Write a .cellm file with the given header and tensor byte data."""
    # Build tensor index
    data_start_est = 0
    tensor_offsets = {}
    current_offset = data_start_est
    for name in sorted(tensors_bytes.keys()):
        current_offset = (current_offset + 63) & ~63
        tensor_offsets[name] = current_offset
        current_offset += len(tensors_bytes[name])

    tensor_index = []
    for name in sorted(tensors_bytes.keys()):
        tensor_index.append(
            {
                "name": name,
                "offset_bytes": tensor_offsets[name],
                "nbytes": len(tensors_bytes[name]),
                "shape": tensors_shape[name],
                "dtype": "f16",
            }
        )

    header["tensors"] = tensor_index
    header_json = json.dumps(header).encode("utf-8")
    header_len = len(header_json)

    # Recalculate with correct header
    data_start = (5 + 1 + 4 + header_len + 63) & ~63
    tensor_offsets = {}
    current_offset = data_start
    for name in sorted(tensors_bytes.keys()):
        current_offset = (current_offset + 63) & ~63
        tensor_offsets[name] = current_offset
        current_offset += len(tensors_bytes[name])

    for item in tensor_index:
        item["offset_bytes"] = tensor_offsets[item["name"]]

    header["tensors"] = tensor_index
    header_json = json.dumps(header).encode("utf-8")
    header_len = len(header_json)
    data_start = (5 + 1 + 4 + header_len + 63) & ~63

    with open(output_path, "wb") as f:
        f.write(b"CELLM")
        f.write(struct.pack("<B", 1))  # version
        f.write(struct.pack("<I", header_len))
        f.write(header_json)

        current_pos = 5 + 1 + 4 + header_len
        if current_pos < data_start:
            f.write(b"\x00" * (data_start - current_pos))

        for name in sorted(tensors_bytes.keys()):
            tensor_data = tensors_bytes[name]
            pos = f.tell()
            aligned_pos = (pos + 63) & ~63
            if pos < aligned_pos:
                f.write(b"\x00" * (aligned_pos - pos))
            f.write(tensor_data)


def convert(input_dir: Path, output_path: Path):
    """Convert HuggingFace LFM model to .cellm format."""
    # Load config
    with open(input_dir / "config.json") as f:
        config = json.load(f)

    print(f"Model type: {config.get('model_type')}")
    print(f"Hidden size: {config.get('hidden_size')}")
    print(f"Layers: {config.get('num_hidden_layers')}")

    # Find safetensors
    safetensors_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {input_dir}")

    print(
        f"Found {len(safetensors_files)} safetensors file(s): {[f.name for f in safetensors_files]}"
    )

    # Load all tensors via PyTorch (handles bfloat16 natively)
    tensors_bytes = {}
    tensors_shape = {}
    for st_file in safetensors_files:
        print(f"Loading {st_file.name}...")
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)  # torch.Tensor
                print(f"  {name}: {list(tensor.shape)}, dtype={tensor.dtype}")
                if tensor.dtype == torch.bfloat16:
                    # Convert bf16 -> f16
                    tensor_f16 = tensor.to(torch.float16)
                elif tensor.dtype == torch.float32:
                    tensor_f16 = tensor.to(torch.float16)
                elif tensor.dtype == torch.float16:
                    tensor_f16 = tensor
                else:
                    raise ValueError(f"Unsupported dtype {tensor.dtype} for {name}")
                tensors_bytes[name] = tensor_f16.numpy().tobytes()
                tensors_shape[name] = list(tensor.shape)

    n_tensors = len(tensors_bytes)
    total_bytes = sum(len(b) for b in tensors_bytes.values())
    print(f"Loaded {n_tensors} tensors, {total_bytes / 1024 / 1024:.1f} MB in f16")

    # Infer head_dim from k_proj if not in config
    head_dim = config.get("head_dim")
    if head_dim is None:
        for name in sorted(tensors_shape.keys()):
            if "self_attn.k_proj.weight" in name:
                kv_dim = tensors_shape[name][0]
                kv_heads = config.get("num_key_value_heads", 8)
                head_dim = kv_dim // kv_heads
                print(f"Inferred head_dim={head_dim} from {name}")
                break

    # Extract rope_theta from rope_parameters if present
    rope_params = config.get("rope_parameters", {})
    rope_theta = rope_params.get("rope_theta", 1000000.0)

    # Build header
    header = {
        "model_type": "lfm2",
        "source_model_type": config.get("model_type", "lfm2"),
        "source_safetensors_format": "pt",
        "text_tensor_prefix": "model",
        "vocab_size": config.get("vocab_size", 65536),
        "hidden_dim": config.get("hidden_size", 1024),
        "intermediate_size": config.get("intermediate_size", 2560),
        "num_layers": config.get("num_hidden_layers", 14),
        "num_heads": config.get("num_attention_heads", 16),
        "num_kv_heads": config.get("num_key_value_heads", 8),
        "head_dim": head_dim,
        "rms_norm_eps": config.get("norm_eps", 1e-5),
        "rope_theta": rope_theta,
        "bos_token_id": config.get("bos_token_id"),
        "eos_token_id": config.get("eos_token_id", 7),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "tie_word_embeddings": config.get("tie_word_embeddings", True),
        "source_torch_dtype": "bfloat16",
        "source_architectures": config.get("architectures"),
        "source_text_config": config,  # Critical: stores layer_types, conv_L_cache, etc.
    }

    # Write output
    print(f"Writing to {output_path}...")
    write_cellm(output_path, header, tensors_bytes, tensors_shape)

    output_size = output_path.stat().st_size
    print(f"Done! Output size: {output_size / 1024 / 1024:.2f} MB")
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: convert_lfm_hf.py <input_dir> <output.cellm>")
        print("Example: convert_lfm_hf.py models/LFM2.5-230M models/LFM2.5-230M.cellm")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)

    convert(input_dir, output_path)
