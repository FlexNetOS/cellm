#!/usr/bin/env python3
"""
Run DFlash speculative decoding with a model that fits on 16GB Mac.

Target:  Qwen/Qwen3.5-4B          (~8GB BF16, ~4GB with MLX 4-bit)
Draft:   z-lab/Qwen3.5-4B-DFlash  (~400MB)

Usage:
  source .venv-dflash/bin/activate
  python3 scripts/run_dflash_qwen35_4b.py --prompt "Write a quicksort in Python."

The script downloads both models on first run and caches them under ./models/dflash-qwen35/.
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download

# 4-bit MLX quantized target (~2.5 GB) — fits comfortably on 16 GB Mac
TARGET_REPO = "mlx-community/Qwen3.5-4B-MLX-4bit"
# Full BF16 target (~8 GB) — may OOM on 16 GB Mac:
# TARGET_REPO = "Qwen/Qwen3.5-4B"
DRAFT_REPO = "z-lab/Qwen3.5-4B-DFlash"


def ensure_model(repo_id: str, cache_dir: str, local_dir: str, allow_patterns=None, ignore_patterns=None):
    path = os.path.join(cache_dir, local_dir)
    if os.path.isdir(path) and any(os.scandir(path)):
        print(f"Found cached {repo_id} at {path}")
        return path
    print(f"Downloading {repo_id} ...")
    os.makedirs(path, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=path,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    print(f"Saved to {path}")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="Write a quicksort in Python.", help="User prompt")
    p.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--block-size", type=int, default=16, help="DFlash block size")
    p.add_argument("--cache-dir", default="./models/dflash-qwen35")
    p.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode")
    args = p.parse_args()

    # Download / verify models
    target_path = ensure_model(TARGET_REPO, args.cache_dir, "target")
    draft_path = ensure_model(DRAFT_REPO, args.cache_dir, "draft")

    # Import dflash MLX bindings
    try:
        from dflash.model_mlx import load, load_draft, stream_generate
    except ImportError:
        print("ERROR: dflash[mlx] not installed.")
        print("Run:  source .venv-dflash/bin/activate")
        print("      pip install 'dflash[mlx] @ git+https://github.com/z-lab/dflash.git'")
        sys.exit(1)

    print(f"\nLoading target model: {TARGET_REPO}")
    model, tokenizer = load(target_path)

    print(f"Loading draft model:  {DRAFT_REPO}")
    # dflash's load_draft expects repo_id, not local path
    draft = load_draft(DRAFT_REPO)

    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )

    print(f"\n{'=' * 60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'=' * 60}\n")

    tps = 0.0
    generated = ""
    for r in stream_generate(
        model,
        draft,
        tokenizer,
        prompt_text,
        block_size=args.block_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    ):
        print(r.text, end="", flush=True)
        generated += r.text
        tps = r.generation_tps

    print(f"\n\n{'=' * 60}")
    print(f"Throughput: {tps:.2f} tok/s")
    print(f"Generated {len(generated)} chars")


if __name__ == "__main__":
    main()
