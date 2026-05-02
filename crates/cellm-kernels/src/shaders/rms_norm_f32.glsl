// Author: Jeffrey Asante (https://jeffasante.github.io/)
#version 450

//  RMS Normalization
//   out[i] = x[i] / rms(x) * w[i]
//   rms(x) = sqrt(mean(x²) + ε)
//
// Two-dispatch design (host dispatches twice):
//   Dispatch 0 (pass = 0): N_workgroups = ceil(N / 256)
//     Each workgroup reduces its chunk of X into a partial sum-of-squares
//     and writes it to scratch[gl_WorkGroupID.x].
//   Dispatch 1 (pass = 1): N_workgroups = 1
//     The single workgroup reduces all partials from scratch[], computes
//     rms, and normalises every element of X into O.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer X_buf {
    float x[];
};
layout(set = 0, binding = 1) readonly buffer W_buf {
    float w[];
};
layout(set = 0, binding = 2) writeonly buffer O_buf {
    float o[];
};

// Scratch buffer: one float per workgroup from pass 0.
layout(set = 0, binding = 3) buffer Scratch {
    float scratch[];
};

layout(push_constant) uniform Params {
    uint N;
    float eps;
    uint num_workgroups; // number of workgroups in pass 0 (ignored in pass 1)
    uint pass; // 0 = reduce, 1 = normalise
} p;

shared float smem[256];

void main() {
    uint tid = gl_LocalInvocationID.x;

    if (p.pass == 0u) {
        //  Pass 0: partial sum-of-squares per workgroup
        float local_sum = 0.0;
        uint wg_start = gl_WorkGroupID.x * 256u;
        for (uint i = wg_start + tid; i < p.N; i += 256u * p.num_workgroups) {
            float xi = x[i];
            local_sum += xi * xi;
        }
        smem[tid] = local_sum;
        barrier();

        // Reduction within workgroup.
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                smem[tid] += smem[tid + stride];
            }
            barrier();
        }

        // Write partial sum.
        if (tid == 0u) {
            scratch[gl_WorkGroupID.x] = smem[0];
        }
    } else {
        //  Pass 1: reduce partials, compute rms, normalise
        // Only workgroup 0 runs this pass (host dispatches 1 workgroup).

        // Step A — reduce scratch[0 .. num_workgroups-1] into a single sum.
        float acc = 0.0;
        for (uint w = tid; w < p.num_workgroups; w += 256u) {
            acc += scratch[w];
        }
        smem[tid] = acc;
        barrier();

        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                smem[tid] += smem[tid + stride];
            }
            barrier();
        }

        float total_sq = smem[0];
        float rms = sqrt(total_sq / float(p.N) + p.eps);
        float inv_rms = 1.0 / max(rms, 1e-10);

        // Step B — apply normalisation.
        for (uint i = tid; i < p.N; i += 256u) {
            o[i] = x[i] * inv_rms * w[i];
        }
    }
}
