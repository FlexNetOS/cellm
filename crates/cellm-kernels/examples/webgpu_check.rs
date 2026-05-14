use cellm_kernels::WebGpuBackend;
use half::f16;

fn assert_close(name: &str, got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len(), "{name}: length mismatch");
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        if diff > worst {
            worst = diff;
            worst_i = i;
        }
    }
    assert!(
        worst <= tol,
        "{name}: max diff {worst} at {worst_i}; got {:?}, expected {:?}",
        got,
        expected
    );
}

fn rms_ref(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&v, &w)| v * inv * w)
        .collect()
}

fn softmax_ref(x: &[f32]) -> Vec<f32> {
    let mx = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = x.iter().map(|&v| (v - mx).exp()).collect();
    let sum = out.iter().sum::<f32>();
    for v in &mut out {
        *v /= sum;
    }
    out
}

fn layer_norm_ref(x: &[f32], rows: usize, cols: usize, weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| {
            let d = *v - mean;
            d * d
        }).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..cols {
            out[r * cols + c] = (row[c] - mean) * inv * weight[c] + bias[c];
        }
    }
    out
}

fn rms_batch_ref(x: &[f32], rows: usize, cols: usize, weight: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for c in 0..cols {
            out[r * cols + c] = row[c] * inv * weight[c];
        }
    }
    out
}

async fn run() {
    let Some(gpu) = WebGpuBackend::create().await else {
        eprintln!("No native WebGPU adapter available; skipping numeric check.");
        return;
    };

    let weight = vec![
        1.0, 2.0, 3.0,
        -1.0, 0.5, 4.0,
        0.25, -2.0, 1.5,
        3.0, 0.0, -0.5,
    ];
    let x = vec![0.5, -1.0, 2.0];
    let w_buf = gpu.upload_f32(&weight);
    let mut matmul = vec![0.0; 4];
    gpu.matmul_f32(&w_buf, 4, 3, &x, &mut matmul).await;
    let matmul_ref = vec![4.5, 7.0, 5.125, 0.5];
    assert_close("matmul_f32", &matmul, &matmul_ref, 1e-5);

    let weight_f16: Vec<u16> = weight.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let w16_buf = gpu.upload_f16(&weight_f16);
    let mut matmul16 = vec![0.0; 4];
    gpu.matmul_f16(&w16_buf, 4, 3, &x, &mut matmul16).await;
    assert_close("matmul_f16", &matmul16, &matmul_ref, 1e-3);

    let rows = 3u32;
    let out_dim = 4u32;
    let in_dim = 3u32;
    let xb = vec![
        0.5, -1.0, 2.0,
        1.0, 0.0, -2.0,
        -0.25, 3.0, 0.75,
    ];
    let mut batch = vec![0.0; (rows * out_dim) as usize];
    gpu.matmul_batch_f32(&w_buf, rows, out_dim, in_dim, &xb, &mut batch).await;
    let mut batch_ref = vec![0.0; batch.len()];
    for r in 0..rows as usize {
        for o in 0..out_dim as usize {
            let mut sum = 0.0;
            for k in 0..in_dim as usize {
                sum += weight[o * in_dim as usize + k] * xb[r * in_dim as usize + k];
            }
            batch_ref[r * out_dim as usize + o] = sum;
        }
    }
    assert_close("matmul_batch_f32", &batch, &batch_ref, 1e-5);

    let norm_x: Vec<f32> = (0..130).map(|i| (i as f32 - 37.0) / 19.0).collect();
    let norm_w: Vec<f32> = (0..130).map(|i| 0.75 + (i % 7) as f32 * 0.05).collect();
    let norm_w_buf = gpu.upload_f32(&norm_w);
    let mut norm = vec![0.0; norm_x.len()];
    gpu.rms_norm(&norm_w_buf, 1e-6, &norm_x, &mut norm).await;
    assert_close("rms_norm", &norm, &rms_ref(&norm_x, &norm_w, 1e-6), 1e-5);

    let rows_b = 3usize;
    let cols_b = 5usize;
    let xb_norm: Vec<f32> = (0..rows_b * cols_b).map(|i| (i as f32 - 4.0) * 0.25).collect();
    let wb_norm: Vec<f32> = (0..cols_b).map(|i| 0.8 + i as f32 * 0.1).collect();
    let bb_norm: Vec<f32> = (0..cols_b).map(|i| -0.2 + i as f32 * 0.03).collect();
    let wb_buf = gpu.upload_f32(&wb_norm);
    let bb_buf = gpu.upload_f32(&bb_norm);
    let mut ln_out = vec![0.0; xb_norm.len()];
    gpu.layer_norm_batch(
        &wb_buf,
        &bb_buf,
        rows_b as u32,
        cols_b as u32,
        1e-5,
        &xb_norm,
        &mut ln_out,
    ).await;
    assert_close(
        "layer_norm_batch",
        &ln_out,
        &layer_norm_ref(&xb_norm, rows_b, cols_b, &wb_norm, &bb_norm, 1e-5),
        1e-5,
    );

    let mut rmsb_out = vec![0.0; xb_norm.len()];
    gpu.rms_norm_batch(
        &wb_buf,
        rows_b as u32,
        cols_b as u32,
        1e-6,
        &xb_norm,
        &mut rmsb_out,
    ).await;
    assert_close(
        "rms_norm_batch",
        &rmsb_out,
        &rms_batch_ref(&xb_norm, rows_b, cols_b, &wb_norm, 1e-6),
        1e-5,
    );

    let mut softmax: Vec<f32> = (0..130).map(|i| ((i as f32 * 0.13).sin() * 3.0) - 1.0).collect();
    let softmax_expected = softmax_ref(&softmax);
    gpu.softmax(&mut softmax).await;
    assert_close("softmax", &softmax, &softmax_expected, 1e-5);

    let mut gate = vec![-2.0f32, -0.5, 0.0, 0.75, 3.0, 6.0];
    let up = vec![0.25f32, -1.0, 2.0, 0.5, -0.25, 0.125];
    let silu_expected: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| {
        (g / (1.0 + f32::exp(-g))) * u
    }).collect();
    gpu.silu_mul(&mut gate, &up).await;
    assert_close("silu_mul", &gate, &silu_expected, 1e-5);

    let mut rope = vec![
        0.5, -1.0, 0.25, 2.0,
        -0.75, 1.5, -2.0, 0.125,
    ];
    let mut rope_expected = rope.clone();
    let theta = 10000.0f32;
    let pos = 3usize;
    let head_dim = 4usize;
    let rotary_dim = 4usize;
    for h in 0..2usize {
        let rope_start = h * head_dim + head_dim - rotary_dim;
        for i in 0..rotary_dim / 2 {
            let inv_freq = theta.powf(-((2 * i) as f32) / rotary_dim as f32);
            let angle = pos as f32 * inv_freq;
            let (s, c) = angle.sin_cos();
            let x0 = rope_expected[rope_start + i];
            let x1 = rope_expected[rope_start + rotary_dim / 2 + i];
            rope_expected[rope_start + i] = x0 * c + x1 * s;
            rope_expected[rope_start + rotary_dim / 2 + i] = -x0 * s + x1 * c;
        }
    }
    gpu.rope_half(&mut rope, 2, head_dim, rotary_dim, pos, theta).await;
    assert_close("rope_half", &rope, &rope_expected, 1e-5);

    println!("WebGPU numeric check passed.");
}

fn main() {
    pollster::block_on(run());
}
