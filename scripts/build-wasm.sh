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

echo "==> Deploying to docs/wasm/…"
mkdir -p ../../docs/wasm/pkg
cp pkg/cellm_wasm.js ../../docs/wasm/pkg/
cp pkg/cellm_wasm_bg.wasm ../../docs/wasm/pkg/
cp www/index.html ../../docs/wasm/index.html

# Fix paths in docs/wasm/index.html for deployment (no ../pkg/ or ../../../docs/)
# On macOS sed -i needs '' for the extension
sed -i '' 's/\.\.\/pkg\//\.\/pkg\//g' ../../docs/wasm/index.html
sed -i '' 's/\.\.\/\.\.\/\.\.\/docs\//\.\.\//g' ../../docs/wasm/index.html

echo "==> Live at: docs/wasm/index.html"
echo "==> To test locally, serve from project root and open docs/wasm/index.html"
