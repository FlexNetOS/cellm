// Author: Jeffrey Asante (https://jeffasante.github.io/)
#version 450

// SiLU (Sigmoid Linear Unit)
//   out[i] = x[i] * sigmoid(x[i])
//
// Also known as Swish-1.  Used in the gating path of LLaMA-style FFNs.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer X_buf {
    float x[];
};
layout(set = 0, binding = 1) writeonly buffer O_buf {
    float o[];
};

layout(push_constant) uniform Params {
    uint N;
} p;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.N) return;

    float xi = x[i];
    // Numerically stable sigmoid.
    float sig = 1.0 / (1.0 + exp(-xi));
    o[i] = xi * sig;
}
