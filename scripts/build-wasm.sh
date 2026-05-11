#!/usr/bin/env bash
# Build the cellm WASM module using wasm-pack.
#
# Prerequisites:
#   cargo install wasm-pack
#
# Usage:
#   ./build-wasm.sh [--release]
#
# Output goes to crates/cellm-wasm/pkg/

set -euo pipefail

cd "$(dirname "$0")/../crates/cellm-wasm"

PROFILE="${1:---dev}"

echo "==> Building cellm-wasm ($PROFILE)…"

wasm-pack build \
  --target web \
  --out-dir pkg \
  $([ "$PROFILE" = "--release" ] && echo "--release") \
  .

echo ""
echo "==> Done! Output in crates/cellm-wasm/pkg/"
echo "==> To test, serve crates/cellm-wasm/www/ with:"
echo "    python3 -m http.server 8080 --directory crates/cellm-wasm/www/"
echo "    (or any static file server)"
