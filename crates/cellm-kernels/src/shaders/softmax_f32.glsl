#version 450

//  In-place softmax over the last dimension 
//   x is a flat buffer of N elements.
//   The "last dimension" is the entire buffer (single row).
//
// Algorithm: online softmax using shared-memory reduction.
//   max = reduce(max, x)
//   sum = reduce(sum, exp(x - max))
//   out[i] = exp(x[i] - max) / sum
//
// Since a single workgroup handles the entire buffer, N must be ≤ 256
// for this implementation.  The host dispatches one workgroup per row
// of a larger tensor.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer X_buf {
    float x[];
};

layout(push_constant) uniform Params {
    uint N;
} p;

shared float smem[256];
shared float row_max;
shared float row_sum;

void main() {
    uint tid = gl_LocalInvocationID.x;

    //  Load x into shared memory 
    float val = 0.0;
    if (tid < p.N) {
        val = x[tid];
    }
    smem[tid] = val;
    barrier();

    //  Find maximum 
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            smem[tid] = max(smem[tid], smem[tid + stride]);
        }
        barrier();
    }
    if (tid == 0u) {
        row_max = smem[0];
    }
    barrier();

    float local_max = row_max;

    //  Compute exp(x - max) and store back to smem 
    if (tid < p.N) {
        val = exp(val - local_max);
        smem[tid] = val;
    } else {
        smem[tid] = 0.0;
    }
    barrier();

    //  Reduce sum 
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        barrier();
    }
    if (tid == 0u) {
        row_sum = smem[0];
    }
    barrier();

    //  Normalise and write back 
    float inv_sum = 1.0 / max(row_sum, 1e-10);
    if (tid < p.N) {
        x[tid] = val * inv_sum;
    }
}
