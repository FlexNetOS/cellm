#!/bin/bash
#
# Validate all cellm GLSL compute shaders.
#
# Compiles each .glsl shader to SPIR-V via glslangValidator, then runs
# spirv-val to verify the output is valid.  Prints a summary table.
#
# Usage:
#   ./tools/validate_glsl.sh
#
# Requirements:
#   glslangValidator  (brew install glslang)
#   spirv-val         (brew install spirv-tools)

set -euo pipefail

SHADER_DIR="crates/cellm-kernels/src/shaders"
TMPDIR="${TMPDIR:-/tmp}/cellm-shader-test"
rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"

SHADERS=(
    matmul_f32
    attention_f32
    rms_norm_f32
    rope_f32
    silu_f32
    add_f32
    mul_f32
    softmax_f32
)

PASS=0
FAIL=0
RESULTS=""

echo "=== cellm GLSL Shader Validation ==="
echo ""

for shader in "${SHADERS[@]}"; do
    src="${SHADER_DIR}/${shader}.glsl"
    spv="${TMPDIR}/${shader}.spv"

    if [ ! -f "$src" ]; then
        echo "FAIL  ${shader}  (file not found: $src)"
        FAIL=$((FAIL + 1))
        RESULTS="${RESULTS}FAIL  ${shader}  (missing)\n"
        continue
    fi

    # Compile GLSL to SPIR-V.
    if glslangValidator -V -S comp --target-env vulkan1.1 -o "$spv" "$src" 2>/dev/null; then
        spv_size=$(wc -c < "$spv" | tr -d ' ')
        # Validate SPIR-V.
        if spirv-val "$spv" 2>/dev/null; then
            echo "PASS  ${shader}  (${spv_size} bytes)"
            PASS=$((PASS + 1))
            RESULTS="${RESULTS}PASS  ${shader}  ${spv_size}B\n"
        else
            echo "FAIL  ${shader}  (spirv-val rejected)"
            FAIL=$((FAIL + 1))
            RESULTS="${RESULTS}FAIL  ${shader}  (invalid SPIR-V)\n"
        fi
    else
        echo "FAIL  ${shader}  (glslangValidator failed)"
        FAIL=$((FAIL + 1))
        RESULTS="${RESULTS}FAIL  ${shader}  (compile error)\n"
    fi
done

echo ""
echo "---"
echo -e "$RESULTS"
echo "---"
echo "Total: $((PASS + FAIL))  |  Passed: ${PASS}  |  Failed: ${FAIL}"

rm -rf "$TMPDIR"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
