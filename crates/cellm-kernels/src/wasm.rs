// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! WASM SIMD kernel implementations for cellm, mirroring the NEON intrinsics
//! in `cpu_kernels.rs` but using `std::arch::wasm32` v128 SIMD intrinsics.
//!
//! Each function is gated with `#[cfg(target_arch = "wasm32")]` for the SIMD
//! path and provides a scalar fallback under `#[cfg(not(target_arch = "wasm32"))]`.

use std::f32;
use rayon::prelude::*;
use half::f16;

#[inline(always)]
fn unpack_i4(packed_row: &[u8], idx: usize) -> f32 {
    let byte = packed_row[idx / 2];
    let nibble = if idx % 2 == 0 {
        byte & 0x0f
    } else {
        (byte >> 4) & 0x0f
    };
    (nibble as i8 - 8) as f32
}

pub fn rms_norm_f32(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());

    let mut mean_sq = 0.0f32;
    for &v in x {
        mean_sq += v * v;
    }
    mean_sq /= x.len() as f32;
    let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();

    // Skip Rayon for small vectors where dispatch overhead dominates.
    if x.len() < 2048 {
        for i in 0..x.len() {
            out[i] = x[i] * inv_rms * weight[i];
        }
    } else {
        out.par_iter_mut().zip(x.par_iter()).zip(weight.par_iter()).for_each(|((o, &xi), &wi)| {
            *o = xi * inv_rms * wi;
        });
    }
}

pub fn matmul_f32(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, out: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    if n == 1 {
        // Matrix-vector product – parallelise across rows with WASM SIMD.
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let row = &a[i * k..(i + 1) * k];

            #[cfg(target_arch = "wasm32")]
            {
                let mut acc;
                let mut kk = 0;
                unsafe {
                    use std::arch::wasm32::*;
                    let mut sum0 = f32x4_splat(0.0);
                    let mut sum1 = f32x4_splat(0.0);
                    let mut sum2 = f32x4_splat(0.0);
                    let mut sum3 = f32x4_splat(0.0);

                    while kk + 16 <= k {
                        let av0 = v128_load(row.as_ptr().add(kk) as *const v128);
                        let av1 = v128_load(row.as_ptr().add(kk + 4) as *const v128);
                        let av2 = v128_load(row.as_ptr().add(kk + 8) as *const v128);
                        let av3 = v128_load(row.as_ptr().add(kk + 12) as *const v128);
                        let bv0 = v128_load(b.as_ptr().add(kk) as *const v128);
                        let bv1 = v128_load(b.as_ptr().add(kk + 4) as *const v128);
                        let bv2 = v128_load(b.as_ptr().add(kk + 8) as *const v128);
                        let bv3 = v128_load(b.as_ptr().add(kk + 12) as *const v128);
                        sum0 = f32x4_add(sum0, f32x4_mul(av0, bv0));
                        sum1 = f32x4_add(sum1, f32x4_mul(av1, bv1));
                        sum2 = f32x4_add(sum2, f32x4_mul(av2, bv2));
                        sum3 = f32x4_add(sum3, f32x4_mul(av3, bv3));
                        kk += 16;
                    }
                    let res = f32x4_add(f32x4_add(sum0, sum1), f32x4_add(sum2, sum3));
                    acc = f32x4_extract_lane::<0>(res)
                        + f32x4_extract_lane::<1>(res)
                        + f32x4_extract_lane::<2>(res)
                        + f32x4_extract_lane::<3>(res);
                }
                while kk < k {
                    acc += row[kk] * b[kk];
                    kk += 1;
                }
                *o = acc;
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += row[kk] * b[kk];
                }
                *o = acc;
            }
        });
    } else {
        // Matrix-matrix product – parallelise across output rows.
        out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
            let a_row = &a[i * k..(i + 1) * k];
            for kk in 0..k {
                let av = a_row[kk];
                let b_row = &b[kk * n..(kk + 1) * n];
                for j in 0..n {
                    out_row[j] += av * b_row[j];
                }
            }
        });
    }
}

pub fn matmul_i8_f32(
    a_i8: &[i8],
    a_scales_f16: &[u16],
    m: usize,
    k: usize,
    b: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(a_i8.len(), m * k);
    debug_assert_eq!(a_scales_f16.len(), m);
    debug_assert_eq!(out.len(), m);

    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let row = &a_i8[i * k..(i + 1) * k];
        let scale = f16::from_bits(a_scales_f16[i]).to_f32();

        #[cfg(target_arch = "wasm32")]
        {
            let mut dot = 0.0f32;
            let mut i_inner = 0;
            unsafe {
                use std::arch::wasm32::*;
                let mut sum0 = f32x4_splat(0.0);
                let mut sum1 = f32x4_splat(0.0);
                let mut sum2 = f32x4_splat(0.0);
                let mut sum3 = f32x4_splat(0.0);

                while i_inner + 16 <= k {
                    // Load 16 i8 values
                    let wv = v128_load(row.as_ptr().add(i_inner) as *const v128);
                    // Load 16 f32 values (4 × v128)
                    let xv0 = v128_load(b.as_ptr().add(i_inner) as *const v128);
                    let xv1 = v128_load(b.as_ptr().add(i_inner + 4) as *const v128);
                    let xv2 = v128_load(b.as_ptr().add(i_inner + 8) as *const v128);
                    let xv3 = v128_load(b.as_ptr().add(i_inner + 12) as *const v128);

                    // Extend low 8 i8 → i16
                    let wv16_low = i16x8_extend_low_i8x16(wv);
                    // Extend high 8 i8 → i16
                    let wv16_high = i16x8_extend_high_i8x16(wv);

                    // Low 4 i16 → i32 → f32
                    let w_i32_0 = i32x4_extend_low_i16x8(wv16_low);
                    let w_i32_1 = i32x4_extend_high_i16x8(wv16_low);
                    let w_i32_2 = i32x4_extend_low_i16x8(wv16_high);
                    let w_i32_3 = i32x4_extend_high_i16x8(wv16_high);

                    let w_f0 = f32x4(
                        f32x4_extract_lane::<0>(w_i32_0) as f32,
                        f32x4_extract_lane::<1>(w_i32_0) as f32,
                        f32x4_extract_lane::<2>(w_i32_0) as f32,
                        f32x4_extract_lane::<3>(w_i32_0) as f32,
                    );
                    let w_f1 = f32x4(
                        f32x4_extract_lane::<0>(w_i32_1) as f32,
                        f32x4_extract_lane::<1>(w_i32_1) as f32,
                        f32x4_extract_lane::<2>(w_i32_1) as f32,
                        f32x4_extract_lane::<3>(w_i32_1) as f32,
                    );
                    let w_f2 = f32x4(
                        f32x4_extract_lane::<0>(w_i32_2) as f32,
                        f32x4_extract_lane::<1>(w_i32_2) as f32,
                        f32x4_extract_lane::<2>(w_i32_2) as f32,
                        f32x4_extract_lane::<3>(w_i32_2) as f32,
                    );
                    let w_f3 = f32x4(
                        f32x4_extract_lane::<0>(w_i32_3) as f32,
                        f32x4_extract_lane::<1>(w_i32_3) as f32,
                        f32x4_extract_lane::<2>(w_i32_3) as f32,
                        f32x4_extract_lane::<3>(w_i32_3) as f32,
                    );

                    sum0 = f32x4_add(sum0, f32x4_mul(w_f0, xv0));
                    sum1 = f32x4_add(sum1, f32x4_mul(w_f1, xv1));
                    sum2 = f32x4_add(sum2, f32x4_mul(w_f2, xv2));
                    sum3 = f32x4_add(sum3, f32x4_mul(w_f3, xv3));
                    i_inner += 16;
                }
                let res = f32x4_add(f32x4_add(sum0, sum1), f32x4_add(sum2, sum3));
                dot = f32x4_extract_lane::<0>(res)
                    + f32x4_extract_lane::<1>(res)
                    + f32x4_extract_lane::<2>(res)
                    + f32x4_extract_lane::<3>(res);
            }
            while i_inner < k {
                dot += (row[i_inner] as f32) * b[i_inner];
                i_inner += 1;
            }
            *o = dot * scale;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += (row[kk] as f32) * b[kk];
            }
            *o = acc * scale;
        }
    });
}

pub fn matmul_f16_f32(
    a_f16: &[u16],
    m: usize,
    k: usize,
    b: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(a_f16.len(), m * k);
    debug_assert_eq!(out.len(), m);

    out.par_iter_mut().with_min_len(32).enumerate().for_each(|(i, o)| {
        let row = &a_f16[i * k..(i + 1) * k];

        #[cfg(target_arch = "wasm32")]
        {
            let mut dot = 0.0f32;
            let mut i_inner = 0;

            // WASM SIMD has no native f16→f32 conversion, so we convert a block
            // of 4 f16 values to f32 using scalar helpers, then accumulate with
            // SIMD multiply-add.
            unsafe {
                use std::arch::wasm32::*;
                let mut sum0 = f32x4_splat(0.0);
                let mut sum1 = f32x4_splat(0.0);
                let mut sum2 = f32x4_splat(0.0);
                let mut sum3 = f32x4_splat(0.0);

                while i_inner + 16 <= k {
                    // Convert 4 groups of 4 f16 → f32 using scalar helpers,
                    // then load them into v128 registers.
                    let wf0 = {
                        let mut tmp = [0.0f32; 4];
                        for j in 0..4 {
                            tmp[j] = f16::from_bits(row[i_inner + j]).to_f32();
                        }
                        v128_load(tmp.as_ptr() as *const v128)
                    };
                    let wf1 = {
                        let mut tmp = [0.0f32; 4];
                        for j in 0..4 {
                            tmp[j] = f16::from_bits(row[i_inner + 4 + j]).to_f32();
                        }
                        v128_load(tmp.as_ptr() as *const v128)
                    };
                    let wf2 = {
                        let mut tmp = [0.0f32; 4];
                        for j in 0..4 {
                            tmp[j] = f16::from_bits(row[i_inner + 8 + j]).to_f32();
                        }
                        v128_load(tmp.as_ptr() as *const v128)
                    };
                    let wf3 = {
                        let mut tmp = [0.0f32; 4];
                        for j in 0..4 {
                            tmp[j] = f16::from_bits(row[i_inner + 12 + j]).to_f32();
                        }
                        v128_load(tmp.as_ptr() as *const v128)
                    };

                    let xv0 = v128_load(b.as_ptr().add(i_inner) as *const v128);
                    let xv1 = v128_load(b.as_ptr().add(i_inner + 4) as *const v128);
                    let xv2 = v128_load(b.as_ptr().add(i_inner + 8) as *const v128);
                    let xv3 = v128_load(b.as_ptr().add(i_inner + 12) as *const v128);

                    sum0 = f32x4_add(sum0, f32x4_mul(wf0, xv0));
                    sum1 = f32x4_add(sum1, f32x4_mul(wf1, xv1));
                    sum2 = f32x4_add(sum2, f32x4_mul(wf2, xv2));
                    sum3 = f32x4_add(sum3, f32x4_mul(wf3, xv3));
                    i_inner += 16;
                }
                let res = f32x4_add(f32x4_add(sum0, sum1), f32x4_add(sum2, sum3));
                dot = f32x4_extract_lane::<0>(res)
                    + f32x4_extract_lane::<1>(res)
                    + f32x4_extract_lane::<2>(res)
                    + f32x4_extract_lane::<3>(res);
            }
            while i_inner < k {
                dot += f16::from_bits(row[i_inner]).to_f32() * b[i_inner];
                i_inner += 1;
            }
            *o = dot;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += f16::from_bits(row[kk]).to_f32() * b[kk];
            }
            *o = acc;
        }
    });
}

pub fn matmul_i4_f32(
    a_i4: &[u8],
    a_scales_f16: &[u16],
    m: usize,
    k: usize,
    gs: usize,
    b: &[f32],
    out: &mut [f32],
) {
    let row_stride = k.div_ceil(2);
    let spr = a_scales_f16.len() / m;
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let row = &a_i4[i * row_stride..(i + 1) * row_stride];
        let rs = &a_scales_f16[i * spr..(i + 1) * spr];
        let mut dot = 0.0f32;
        for j in 0..k {
            let b_idx = j / 2;
            let n = if j % 2 == 0 { row[b_idx] & 0xf } else { row[b_idx] >> 4 };
            let q = (n as i8) - 8;
            let scale = f16::from_bits(rs[j / gs]).to_f32();
            dot += (q as f32) * scale * b[j];
        }
        *o = dot;
    });
}

pub fn softmax_f32_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum == 0.0 {
        return;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

pub fn rope_non_interleaved_inplace_f32(
    x: &mut [f32],
    _n_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    pos: usize,
    theta: f32,
) {
    let half = rotary_dim / 2;
    // Skip Rayon for small head counts where dispatch overhead dominates.
    if x.len() < 2048 {
        for head in x.chunks_exact_mut(head_dim) {
            for i in 0..half {
                let inv_freq = theta.powf(-(2.0 * i as f32) / rotary_dim as f32);
                let angle = pos as f32 * inv_freq;
                let (sin, cos) = angle.sin_cos();
                let x0 = head[i];
                let x1 = head[half + i];
                head[i] = x0 * cos - x1 * sin;
                head[half + i] = x1 * cos + x0 * sin;
            }
        }
    } else {
        x.par_chunks_exact_mut(head_dim).for_each(|head| {
            for i in 0..half {
                let inv_freq = theta.powf(-(2.0 * i as f32) / rotary_dim as f32);
                let angle = pos as f32 * inv_freq;
                let (sin, cos) = angle.sin_cos();
                let x0 = head[i];
                let x1 = head[half + i];
                head[i] = x0 * cos - x1 * sin;
                head[half + i] = x1 * cos + x0 * sin;
            }
        });
    }
}

pub fn rope_interleaved_inplace_f32(
    x: &mut [f32],
    _n_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
) {
    let half = head_dim / 2;
    x.par_chunks_exact_mut(head_dim).for_each(|head| {
        for i in 0..half {
            let inv_freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
            let angle = pos as f32 * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = head[2 * i];
            let x1 = head[2 * i + 1];
            head[2 * i] = x0 * cos - x1 * sin;
            head[2 * i + 1] = x1 * cos + x0 * sin;
        }
    });
}

pub fn attention_single_token_gqa_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let qkv_ratio = n_heads / n_kv_heads;

    // Parallelise across attention heads.
    out.par_chunks_exact_mut(head_dim).enumerate().for_each(|(h, out_h)| {
        let kv_h = h / qkv_ratio;
        let qh = &q[h * head_dim..(h + 1) * head_dim];

        // Thread-local score buffer.
        let mut scores = vec![0.0f32; seq];
        for t in 0..seq {
            let kt_base = (t * n_kv_heads + kv_h) * head_dim;
            let kt = &k[kt_base..kt_base + head_dim];
            let mut dot = 0.0f32;

            #[cfg(target_arch = "wasm32")]
            unsafe {
                use std::arch::wasm32::*;
                let mut sumv = f32x4_splat(0.0);
                let mut i = 0;
                while i + 4 <= head_dim {
                    let qv = v128_load(qh.as_ptr().add(i) as *const v128);
                    let kv = v128_load(kt.as_ptr().add(i) as *const v128);
                    sumv = f32x4_add(sumv, f32x4_mul(qv, kv));
                    i += 4;
                }
                dot = f32x4_extract_lane::<0>(sumv)
                    + f32x4_extract_lane::<1>(sumv)
                    + f32x4_extract_lane::<2>(sumv)
                    + f32x4_extract_lane::<3>(sumv);
                while i < head_dim {
                    dot += qh[i] * kt[i];
                    i += 1;
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                for i in 0..head_dim {
                    dot += qh[i] * kt[i];
                }
            }
            scores[t] = dot * scale;
        }

        softmax_f32_inplace(&mut scores);

        out_h.fill(0.0);
        for t in 0..seq {
            let vt_base = (t * n_kv_heads + kv_h) * head_dim;
            let vt = &v[vt_base..vt_base + head_dim];
            let w = scores[t];

            #[cfg(target_arch = "wasm32")]
            unsafe {
                use std::arch::wasm32::*;
                let wv = f32x4_splat(w);
                let mut i = 0;
                while i + 4 <= head_dim {
                    let ov = v128_load(out_h.as_ptr().add(i) as *const v128);
                    let vv = v128_load(vt.as_ptr().add(i) as *const v128);
                    v128_store(
                        out_h.as_mut_ptr().add(i) as *mut v128,
                        f32x4_add(ov, f32x4_mul(wv, vv)),
                    );
                    i += 4;
                }
                while i < head_dim {
                    out_h[i] += w * vt[i];
                    i += 1;
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                for i in 0..head_dim {
                    out_h[i] += w * vt[i];
                }
            }
        }
    });
}
