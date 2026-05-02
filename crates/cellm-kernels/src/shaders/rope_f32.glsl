// Author: Jeffrey Asante (https://jeffasante.github.io/)
#version 450

// Rotary Position Embedding (RoPE)
// Applied **in-place** to the Q buffer.
//   Q layout: [n_heads * head_dim] floats
//   For each head, pairs (d, d+1) are rotated by angle = pos * θ^(d / head_dim).
//
// Push constants:
//   n_heads   – number of attention heads
//   head_dim  – dimension of each head (must be even)
//   pos       – token position index
//   theta     – base frequency (typically 10_000.0 or 500_000.0)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Q_buf {
    float q[];
};

layout(push_constant) uniform Params {
    uint n_heads;
    uint head_dim;
    uint pos;
    float theta;
} p;

void main() {
    uint tid = gl_GlobalInvocationID.x;

    // Each thread handles one (d/2) pair across all heads.
    // Total pair count = n_heads * head_dim / 2
    uint total_pairs = p.n_heads * (p.head_dim / 2u);

    if (tid >= total_pairs) return;

    uint head = tid / (p.head_dim / 2u);
    uint pair = tid % (p.head_dim / 2u);
    uint d = pair * 2u; // even index within the head
    uint base = head * p.head_dim;

    // Compute rotation angle.
    float exponent = float(d) / float(p.head_dim);
    float freq = 1.0 / pow(p.theta, exponent);
    float angle = float(p.pos) * freq;

    float cos_a = cos(angle);
    float sin_a = sin(angle);

    // Fetch the (even, odd) pair.
    uint i0 = base + d;
    uint i1 = base + d + 1u;
    float x0 = q[i0];
    float x1 = q[i1];

    // Apply 2D rotation.
    q[i0] = x0 * cos_a - x1 * sin_a;
    q[i1] = x0 * sin_a + x1 * cos_a;
}
