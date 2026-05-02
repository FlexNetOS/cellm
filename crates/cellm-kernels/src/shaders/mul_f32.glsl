// Author: Jeffrey Asante (https://jeffasante.github.io/)
#version 450

// Element-wise multiplication
//   out[i] = a[i] * b[i]

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer A_buf {
    float a[];
};
layout(set = 0, binding = 1) readonly buffer B_buf {
    float b[];
};
layout(set = 0, binding = 2) writeonly buffer O_buf {
    float o[];
};

layout(push_constant) uniform Params {
    uint N;
} p;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.N) return;

    o[i] = a[i] * b[i];
}
