#!/usr/bin/env python3
"""
Download and run inference on z-lab/gemma-4-26B-A4B-it-DFlash.

This model is a DFlash draft model for speculative decoding. It is designed to be
used alongside the target model (google/gemma-4-26B-A4B-it) to accelerate inference.

Usage:
  # 1. Download model files (safetensors + config)
  python3 scripts/run_dflash_inference.py --download --cache-dir ./models/dflash

  # 2. Inspect safetensor files (shapes, dtypes, sample weights)
  python3 scripts/run_dflash_inference.py --inspect --cache-dir ./models/dflash

  # 3. Run DFlash speculative decoding with MLX (Apple Silicon)
  python3 scripts/run_dflash_inference.py --mlx --cache-dir ./models/dflash \
      --prompt "Write a quicksort in Python."

  # 4. Load raw safetensors into torch tensors
  python3 scripts/run_dflash_inference.py --load-raw --cache-dir ./models/dflash

Requirements:
  - huggingface_hub
  - safetensors
  - torch (for --load-raw)
  - mlx, mlx_lm, dflash[mlx] (for --mlx speculative decoding)

Note: The full gemma-4-26B-A4B-it target model requires ~26B parameters.
  On a 16GB Mac you may need to use a quantized variant or a smaller target.
"""

import argparse
import os
import sys

REPO_ID = "z-lab/gemma-4-26B-A4B-it-DFlash"
TARGET_REPO_ID = "google/gemma-4-26B-A4B-it"


def download_model(cache_dir: str):
    """Download the DFlash draft model and target model config from HuggingFace."""
    from huggingface_hub import snapshot_download

    os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading DFlash draft model: {REPO_ID}")
    draft_path = snapshot_download(
        repo_id=REPO_ID,
        cache_dir=cache_dir,
        local_dir=os.path.join(cache_dir, "draft"),
        local_dir_use_symlinks=False,
    )
    print(f"Draft model downloaded to: {draft_path}")

    print(f"\nDownloading target model config: {TARGET_REPO_ID}")
    target_path = snapshot_download(
        repo_id=TARGET_REPO_ID,
        cache_dir=cache_dir,
        local_dir=os.path.join(cache_dir, "target"),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.bin", "*.safetensors", "*.gguf"],  # only config/tokenizer
    )
    print(f"Target config downloaded to: {target_path}")
    return draft_path, target_path


def inspect_safetensors(draft_dir: str):
    """Inspect all .safetensors files: list keys, shapes, dtypes, and sample values."""
    from safetensors import safe_open
    import glob

    files = sorted(glob.glob(os.path.join(draft_dir, "*.safetensors")))
    if not files:
        print("No .safetensors files found.")
        return

    total_params = 0
    for path in files:
        print(f"\n{'=' * 60}")
        print(f"File: {os.path.basename(path)}")
        print(f"{'=' * 60}")
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = sorted(f.keys())
            for key in keys:
                shape = f.get_tensor(key).shape
                dtype = f.get_tensor(key).dtype
                n_params = int(f.get_tensor(key).numel())
                total_params += n_params
                print(f"  {key:60s} | shape={str(shape):20s} | dtype={str(dtype):10s} | params={n_params:,}")

    print(f"\n{'=' * 60}")
    print(f"Total parameters across all files: {total_params:,}")
    print(f"Estimated FP16 size: {total_params * 2 / 1024**3:.2f} GB")
    print(f"Estimated FP32 size: {total_params * 4 / 1024**3:.2f} GB")


def load_raw_safetensors(draft_dir: str):
    """Load all safetensors into a single dict of torch tensors."""
    from safetensors.torch import load_file
    import glob
    import torch

    files = sorted(glob.glob(os.path.join(draft_dir, "*.safetensors")))
    if not files:
        print("No .safetensors files found.")
        return {}

    state_dict = {}
    for path in files:
        print(f"Loading {os.path.basename(path)} ...")
        part = load_file(path, device="cpu")
        overlap = set(state_dict.keys()) & set(part.keys())
        if overlap:
            print(f"  WARNING: overlapping keys: {overlap}")
        state_dict.update(part)

    print(f"\nLoaded {len(state_dict)} tensors.")
    total = sum(t.numel() for t in state_dict.values())
    print(f"Total parameters: {total:,}")
    return state_dict


def run_mlx_speculative(cache_dir: str, prompt: str, max_tokens: int = 256, temperature: float = 0.6):
    """Run DFlash speculative decoding using MLX on Apple Silicon."""
    try:
        from dflash.model_mlx import load, load_draft, stream_generate
    except ImportError as e:
        print("ERROR: dflash[mlx] is not installed.")
        print("Install with: pip install 'git+https://github.com/z-lab/dflash.git#egg=dflash[mlx]'")
        print(f"  (underlying error: {e})")
        sys.exit(1)

    target_path = os.path.join(cache_dir, "target")
    draft_path = os.path.join(cache_dir, "draft")

    if not os.path.isdir(target_path) or not os.path.isdir(draft_path):
        print("Models not found locally. Run with --download first.")
        sys.exit(1)

    print("Loading target model with MLX...")
    model, tokenizer = load(target_path)

    print("Loading DFlash draft model with MLX...")
    draft = load_draft(draft_path)

    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )

    print(f"\nPrompt: {prompt}\n")
    print("=" * 60)
    print("Generating...\n")

    tps = 0.0
    for r in stream_generate(
        model, draft, tokenizer, prompt_text,
        block_size=16, max_tokens=max_tokens, temperature=temperature
    ):
        print(r.text, end="", flush=True)
        tps = r.generation_tps

    print(f"\n\nThroughput: {tps:.2f} tok/s")


def run_transformers_draft_only(cache_dir: str, prompt: str, max_tokens: int = 32):
    """
    Load the DFlash draft model standalone via transformers (trust_remote_code).
    NOTE: This is NOT the intended use. The draft model is designed for speculative
    decoding with a target model. Standalone output quality will be poor.
    """
    from transformers import AutoModel, AutoTokenizer
    import torch

    draft_path = os.path.join(cache_dir, "draft")
    if not os.path.isdir(draft_path):
        print("Draft model not found. Run with --download first.")
        sys.exit(1)

    print("Loading DFlash draft model (standalone mode)...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    draft = AutoModel.from_pretrained(
        draft_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=device,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(draft_path, trust_remote_code=True)

    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False
    ).to(device)

    print(f"\nPrompt: {prompt}")
    print("Generating (standalone draft mode - quality will be low)...\n")

    with torch.no_grad():
        # Simple greedy generation; draft model may not have standard generate()
        for _ in range(max_tokens):
            out = draft(input_ids)
            logits = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            # Draft models output hidden states, not logits directly, so this is approximate
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            print(tokenizer.decode(next_token[0], skip_special_tokens=True), end="", flush=True)

    print()


def main():
    p = argparse.ArgumentParser(description="Download and inspect z-lab/gemma-4-26B-A4B-it-DFlash")
    p.add_argument("--cache-dir", default="./models/dflash", help="Directory to cache model files")
    p.add_argument("--download", action="store_true", help="Download model from HuggingFace")
    p.add_argument("--inspect", action="store_true", help="Inspect safetensor file contents")
    p.add_argument("--load-raw", action="store_true", help="Load raw safetensors into torch tensors")
    p.add_argument("--mlx", action="store_true", help="Run DFlash speculative decoding with MLX")
    p.add_argument("--draft-only", action="store_true", help="Load draft model standalone (low quality)")
    p.add_argument("--prompt", default="Write a quicksort in Python.", help="Prompt for generation")
    p.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    args = p.parse_args()

    if not any([args.download, args.inspect, args.load_raw, args.mlx, args.draft_only]):
        p.print_help()
        sys.exit(0)

    if args.download:
        download_model(args.cache_dir)

    draft_dir = os.path.join(args.cache_dir, "draft")

    if args.inspect:
        inspect_safetensors(draft_dir)

    if args.load_raw:
        load_raw_safetensors(draft_dir)

    if args.mlx:
        run_mlx_speculative(args.cache_dir, args.prompt, args.max_tokens)

    if args.draft_only:
        run_transformers_draft_only(args.cache_dir, args.prompt, args.max_tokens)


if __name__ == "__main__":
    main()
