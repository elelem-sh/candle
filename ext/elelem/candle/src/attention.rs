use candle_core::Result;
use candle_nn::attention::cpu_flash::causal::causal_decode_f32_interleaved;
use gemm::{gemm, Parallelism};
use rayon::prelude::*;

pub struct Shape {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub queries: usize,
    pub offset: usize,
}

pub fn causal(q: &[f32], cache: &[f32], s: &Shape, scale: f32) -> Result<Vec<f32>> {
    if s.queries == 1 {
        let kv_len = s.offset + 1;
        return causal_decode_f32_interleaved(
            q, cache, s.heads, s.kv_heads, s.head_dim, kv_len, scale,
        )?
        .flatten_all()?
        .to_vec1::<f32>();
    }
    let d = s.head_dim;
    let l = s.queries;
    let kv_len = s.offset + l;
    let group = s.heads / s.kv_heads;
    let rows = group * l;
    let stride = (s.kv_heads * 2 * d) as isize;
    let groups: Vec<Vec<f32>> = (0..s.kv_heads)
        .into_par_iter()
        .map(|g| {
            let mut scores = vec![0f32; rows * kv_len];
            let mut out = vec![0f32; rows * d];
            unsafe {
                gemm(
                    rows,
                    kv_len,
                    d,
                    scores.as_mut_ptr(),
                    1,
                    kv_len as isize,
                    false,
                    q.as_ptr().add(g * rows * d),
                    1,
                    d as isize,
                    cache.as_ptr().add(g * 2 * d),
                    stride,
                    1,
                    0.0,
                    scale,
                    false,
                    false,
                    false,
                    Parallelism::None,
                );
            }
            for (r, row) in scores.chunks_exact_mut(kv_len).enumerate() {
                let limit = s.offset + r % l + 1;
                let (live, masked) = row.split_at_mut(limit);
                let max = live.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for x in live.iter_mut() {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                let inv = 1.0 / sum;
                live.iter_mut().for_each(|x| *x *= inv);
                masked.fill(0.0);
            }
            unsafe {
                gemm(
                    rows,
                    d,
                    kv_len,
                    out.as_mut_ptr(),
                    1,
                    d as isize,
                    false,
                    scores.as_ptr(),
                    1,
                    kv_len as isize,
                    cache.as_ptr().add(g * 2 * d + d),
                    1,
                    stride,
                    0.0,
                    1.0,
                    false,
                    false,
                    false,
                    Parallelism::None,
                );
            }
            out
        })
        .collect();
    let mut ctx = vec![0f32; l * s.heads * d];
    for (g, out) in groups.iter().enumerate() {
        for h in 0..group {
            for i in 0..l {
                let src = (h * l + i) * d;
                let dst = (i * s.heads + g * group + h) * d;
                ctx[dst..dst + d].copy_from_slice(&out[src..src + d]);
            }
        }
    }
    Ok(ctx)
}
