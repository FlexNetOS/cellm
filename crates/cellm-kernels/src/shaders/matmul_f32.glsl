// Author: Jeffrey Asante (https://jeffasante.github.io/)
#version 450

// Tiled matrix multiply: C[M,N] = A[M,K] * B[K,N]
// Each workgroup computes a 16×16 tile of C using shared-memory tiling along
// the K dimension.  The tile size matches the workgroup size so every thread
// produces exactly one output element.

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer A_buf {
    float a[];
};
layout(set = 0, binding = 1) readonly buffer B_buf {
    float b[];
};
layout(set = 0, binding = 2) writeonly buffer C_buf {
    float c[];
};

layout(push_constant) uniform Params {
    uint M;
    uint N;
    uint K;
} p;

// Shared-memory tiles — one row of A and one column of B per workgroup.
shared float As[16][16];
shared float Bs[16][16];

void main() {
    uint row = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.y; // global row in C
    uint col = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x; // global col in C

    uint lid_y = gl_LocalInvocationID.y; // 0..15
    uint lid_x = gl_LocalInvocationID.x; // 0..15

    float acc = 0.0;

    // Slide across K in 16-element steps.
    uint tiles = (p.K + 15u) / 16u;
    for (uint t = 0u; t < tiles; ++t) {
        uint k_off = t * 16u;

        // Cooperative load of A tile (row = global row, col = k_off + lid_x)
        if (row < p.M && (k_off + lid_x) < p.K) {
            As[lid_y][lid_x] = a[row * p.K + k_off + lid_x];
        } else {
            As[lid_y][lid_x] = 0.0;
        }

        // Cooperative load of B tile (row = k_off + lid_y, col = global col)
        if ((k_off + lid_y) < p.K && col < p.N) {
            Bs[lid_y][lid_x] = b[(k_off + lid_y) * p.N + col];
        } else {
            Bs[lid_y][lid_x] = 0.0;
        }

        barrier();

        // Accumulate the 16×16 inner product.
        for (uint kk = 0u; kk < 16u; ++kk) {
            acc += As[lid_y][kk] * Bs[kk][lid_x];
        }

        barrier();
    }

    // Write result.
    if (row < p.M && col < p.N) {
        c[row * p.N + col] = acc;
    }
}
