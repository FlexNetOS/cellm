use std::time::Instant;
use half::f16;

fn main() {
    // Simulate CpuKvStorage attention: scattered f16 reads with NEON dot products
    let head_dim = 64;
    let n_heads = 9;
    let n_kv_heads = 3;
    let n_layers = 30;
    let avg_seq = 165; // average over 330 tokens
    let n_tokens = 330;

    // Simulate KV cache (f16 storage)
    let kv_dim = n_kv_heads * head_dim;
    let total_kv = 1000 * kv_dim; // enough for 1000 tokens
    let k_cache: Vec<f16> = vec![f16::from_f32(0.5); total_kv];
    let v_cache: Vec<f16> = vec![f16::from_f32(0.5); total_kv];
    let q = vec![1.0f32; n_heads * head_dim];
    let mut out = vec![0.0f32; n_heads * head_dim];
    let mut scores = vec![0.0f32; avg_seq];
    
    let group_size = n_heads / n_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    
    // Precompute bases
    let bases: Vec<usize> = (0..avg_seq).map(|t| t * kv_dim).collect();
    
    let t0 = Instant::now();
    for _tok in 0..n_tokens {
        for _layer in 0..n_layers {
            out.fill(0.0);
            for h in 0..n_heads {
                let kv_h = h / group_size;
                let qh = &q[h * head_dim..(h + 1) * head_dim];
                
                for (t, &base) in bases.iter().enumerate() {
                    let kt_base = base + kv_h * head_dim;
                    let kt = &k_cache[kt_base..kt_base + head_dim];
                    let mut dot = 0.0f32;
                    
                    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                    unsafe {
                        use std::arch::aarch64::*;
                        let mut sum0 = vdupq_n_f32(0.0);
                        let mut i = 0;
                        while i + 4 <= head_dim {
                            let h_raw = vld1_u16(kt.as_ptr().add(i) as *const u16);
                            let w = vmovl_u16(h_raw);
                            let sign = vandq_u32(vshlq_n_u32(w, 16), vdupq_n_u32(0x80000000u32));
                            let normal = vshlq_n_u32(
                                vaddq_u32(vandq_u32(w, vdupq_n_u32(0x7fff)), vdupq_n_u32(0x1c000u32)),
                                13,
                            );
                            let not_zero = vtstq_u32(w, vdupq_n_u32(0x7c00));
                            let kf = vreinterpretq_f32_u32(vorrq_u32(sign, vandq_u32(normal, not_zero)));
                            let qf = vld1q_f32(qh.as_ptr().add(i));
                            sum0 = vfmaq_f32(sum0, kf, qf);
                            i += 4;
                        }
                        dot = vaddvq_f32(sum0);
                    }
                    scores[t] = dot * scale;
                }
                
                // softmax
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for s in scores.iter_mut() { *s *= inv; }
                
                let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
                for (t, &base) in bases.iter().enumerate() {
                    let vt_base = base + kv_h * head_dim;
                    let vt = &v_cache[vt_base..vt_base + head_dim];
                    let w = scores[t];
                    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                    unsafe {
                        use std::arch::aarch64::*;
                        let wv = vdupq_n_f32(w);
                        let mut i = 0;
                        while i + 4 <= head_dim {
                            let h_raw = vld1_u16(vt.as_ptr().add(i) as *const u16);
                            let ww = vmovl_u16(h_raw);
                            let sign = vandq_u32(vshlq_n_u32(ww, 16), vdupq_n_u32(0x80000000u32));
                            let normal = vshlq_n_u32(
                                vaddq_u32(vandq_u32(ww, vdupq_n_u32(0x7fff)), vdupq_n_u32(0x1c000u32)),
                                13,
                            );
                            let not_zero = vtstq_u32(ww, vdupq_n_u32(0x7c00));
                            let vf = vreinterpretq_f32_u32(vorrq_u32(sign, vandq_u32(normal, not_zero)));
                            let ov = vld1q_f32(out_h.as_ptr().add(i));
                            vst1q_f32(out_h.as_mut_ptr().add(i), vfmaq_f32(ov, wv, vf));
                            i += 4;
                        }
                    }
                }
            }
        }
    }
    let elapsed = t0.elapsed();
    println!("NEON attention time: {:?}", elapsed);
    println!("Per token: {:?}", elapsed / n_tokens);
    println!("Per layer per token: {:?}", elapsed / (n_tokens * n_layers as u32));
}
