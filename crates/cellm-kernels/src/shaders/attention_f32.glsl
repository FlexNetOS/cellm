#version 450

// Single-query GQA (Grouped-Query Attention)
// One query token attends over a full KV sequence.
//   Q : [n_heads * head_dim]
//   K : [seq_len * n_kv_heads * head_dim]
//   V : [seq_len * n_kv_heads * head_dim]
//   O : [n_heads * head_dim]
//
// head_groups = n_heads / n_kv_heads   (integer; validated on host)
//
// Each workgroup handles one query head.  With 256 threads, head_dim must
// be ≤ 256 and a multiple of the workgroup size (common: 64, 128, 256).

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Q_buf {
    float q[];
};
layout(set = 0, binding = 1) readonly buffer K_buf {
    float k[];
};
layout(set = 0, binding = 2) readonly buffer V_buf {
    float v[];
};
layout(set = 0, binding = 3) writeonly buffer O_buf {
    float o[];
};

layout(push_constant) uniform Params {
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint seq_len;
    float attn_scale;
    float soft_cap;
} p;

shared float scores[256]; // one score per sequence position (seq_len ≤ 256)
shared float max_val;
shared float sum_exp;

void main() {
    uint head = gl_WorkGroupID.x; // which query head
    uint tid = gl_LocalInvocationID.x; // thread within the workgroup
    uint hd = p.head_dim;

    //  Determine KV head for this query head (GQA) 
    uint heads_per_kv = p.n_heads / p.n_kv_heads;
    uint kv_head = head / heads_per_kv;

    //  Step 1: compute Q·K scores 
    // Each thread handles one element of the dot-product accumulation,
    // then we reduce across threads and store the per-position score.

    float my_score = 0.0;
    if (tid < p.seq_len) {
        uint pos = tid;
        float dot = 0.0;
        // Dot product Q[head] · K[pos, kv_head]
        for (uint d = 0u; d < hd; ++d) {
            float qv = q[head * hd + d];
            float kv = k[(pos * p.n_kv_heads + kv_head) * hd + d];
            dot += qv * kv;
        }
        my_score = dot * p.attn_scale;
        // Apply soft capping if requested.
        if (p.soft_cap > 0.0) {
            my_score = p.soft_cap * tanh(my_score / p.soft_cap);
        }
    }
    scores[tid] = my_score;

    barrier();

    // Step 2: online softmax
    // Reduction for max across seq_len threads.
    if (tid < p.seq_len) {
        // Find max via shared-memory reduction.
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride && (tid + stride) < p.seq_len) {
                scores[tid] = max(scores[tid], scores[tid + stride]);
            }
            barrier();
        }
        if (tid == 0u) {
            max_val = scores[0];
        }
    }
    barrier();

    // Compute exp(x - max) and sum.
    float local_max = max_val;
    float exp_val = 0.0;
    if (tid < p.seq_len) {
        exp_val = exp(my_score - local_max);
        scores[tid] = exp_val;
    }

    // Reduction for sum.
    barrier();
    if (tid < p.seq_len) {
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride && (tid + stride) < p.seq_len) {
                scores[tid] += scores[tid + stride];
            }
            barrier();
        }
        if (tid == 0u) {
            sum_exp = scores[0];
        }
    }
    barrier();

    // Normalise scores.
    float inv_sum = 1.0 / max(sum_exp, 1e-10);
    if (tid < p.seq_len) {
        scores[tid] = exp_val * inv_sum;
    }
    barrier();

    //  Step 3: weighted sum over V 
    // Each thread accumulates one element of the output head.
    if (tid < hd) {
        float acc = 0.0;
        for (uint pos = 0u; pos < p.seq_len; ++pos) {
            float w = scores[pos];
            float vv = v[(pos * p.n_kv_heads + kv_head) * hd + tid];
            acc += w * vv;
        }
        o[head * hd + tid] = acc;
    }
}
