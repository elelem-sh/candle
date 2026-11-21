use candle_core::utils::barrier_pool;
use gemm::{gemm, Parallelism};
use half::f16;
use half::slice::HalfFloatSliceExt;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Shape {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub queries: usize,
    pub offset: usize,
}

pub struct KvCache {
    k: Vec<Vec<f16>>,
    v: Vec<Vec<f16>>,
    head_dim: usize,
}

impl KvCache {
    pub fn new(kv_heads: usize, head_dim: usize) -> Self {
        Self {
            k: vec![Vec::new(); kv_heads],
            v: vec![Vec::new(); kv_heads],
            head_dim,
        }
    }

    pub fn len(&self) -> usize {
        self.k[0].len() / self.head_dim
    }

    pub fn truncate(&mut self, len: usize) {
        for rows in self.k.iter_mut().chain(self.v.iter_mut()) {
            rows.truncate(len * self.head_dim);
        }
    }

    pub fn append(&mut self, k: &[f32], v: &[f32], positions: usize) {
        let d = self.head_dim;
        let heads = self.k.len();
        for h in 0..heads {
            for (cache, rows) in [(&mut self.k[h], k), (&mut self.v[h], v)] {
                cache.reserve(positions * d);
                for p in 0..positions {
                    let at = (p * heads + h) * d;
                    cache.extend(rows[at..at + d].iter().map(|x| f16::from_f32(*x)));
                }
            }
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma",
    target_feature = "f16c"
))]
mod simd {
    use half::f16;
    use std::arch::x86_64::*;

    const P: usize = 4;
    const T: usize = 64;
    const B: usize = 32;

    #[inline(always)]
    unsafe fn load(x: *const f16) -> __m256 {
        _mm256_cvtph_ps(_mm_loadu_si128(x as *const __m128i))
    }

    #[inline(always)]
    unsafe fn sum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)))
    }

    #[inline(always)]
    unsafe fn block<const H: usize, const N: usize>(
        q: *const f32,
        k: *const f16,
        d: usize,
        scale: f32,
        out: *mut f32,
        kv_len: usize,
        p: usize,
    ) {
        let mut acc = [[_mm256_setzero_ps(); N]; H];
        for c in (0..d).step_by(8) {
            let mut x = [_mm256_setzero_ps(); N];
            for (i, x) in x.iter_mut().enumerate() {
                *x = load(k.add((p + i) * d + c));
            }
            for (h, acc) in acc.iter_mut().enumerate() {
                let qv = _mm256_loadu_ps(q.add(h * d + c));
                for (a, x) in acc.iter_mut().zip(&x) {
                    *a = _mm256_fmadd_ps(qv, *x, *a);
                }
            }
        }
        for (h, acc) in acc.iter().enumerate() {
            for (i, a) in acc.iter().enumerate() {
                *out.add(h * kv_len + p + i) = sum(*a) * scale;
            }
        }
    }

    pub fn scores(q: &[f32], k: &[f16], d: usize, scale: f32, out: &mut [f32]) {
        let group = q.len() / d;
        let kv_len = out.len() / group;
        let blocks = kv_len / P * P;
        unsafe {
            for p in (0..kv_len).step_by(P) {
                for h in (0..group).step_by(2) {
                    let q = q.as_ptr().add(h * d);
                    let o = out.as_mut_ptr().add(h * kv_len);
                    match (group - h >= 2, p < blocks) {
                        (true, true) => block::<2, P>(q, k.as_ptr(), d, scale, o, kv_len, p),
                        (false, true) => block::<1, P>(q, k.as_ptr(), d, scale, o, kv_len, p),
                        (true, false) => (p..kv_len)
                            .for_each(|p| block::<2, 1>(q, k.as_ptr(), d, scale, o, kv_len, p)),
                        (false, false) => (p..kv_len)
                            .for_each(|p| block::<1, 1>(q, k.as_ptr(), d, scale, o, kv_len, p)),
                    }
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn exp(x: __m256) -> __m256 {
        let x = _mm256_max_ps(x, _mm256_set1_ps(-87.0));
        let fx = _mm256_round_ps(
            _mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693_359_4), x);
        let x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.121_944_4e-4), x);
        let mut y = _mm256_set1_ps(1.987_569_1e-4);
        for c in [
            1.398_199_9e-3,
            8.333_452e-3,
            4.166_579_6e-2,
            1.666_666_6e-1,
            5e-1,
        ] {
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(c));
        }
        y = _mm256_fmadd_ps(
            y,
            _mm256_mul_ps(x, x),
            _mm256_add_ps(x, _mm256_set1_ps(1.0)),
        );
        let e = _mm256_slli_epi32(
            _mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)),
            23,
        );
        _mm256_mul_ps(y, _mm256_castsi256_ps(e))
    }

    pub fn softmax(row: &mut [f32]) {
        let n = row.len() / 8 * 8;
        unsafe {
            let p = row.as_mut_ptr();
            let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
            for i in (0..n).step_by(8) {
                m = _mm256_max_ps(m, _mm256_loadu_ps(p.add(i)));
            }
            let mut buf = [0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), m);
            let max = buf
                .iter()
                .chain(&row[n..])
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mv = _mm256_set1_ps(max);
            let mut s = _mm256_setzero_ps();
            for i in (0..n).step_by(8) {
                let e = exp(_mm256_sub_ps(_mm256_loadu_ps(p.add(i)), mv));
                _mm256_storeu_ps(p.add(i), e);
                s = _mm256_add_ps(s, e);
            }
            let mut total = sum(s);
            for x in &mut row[n..] {
                *x = (*x - max).exp();
                total += *x;
            }
            let inv = _mm256_set1_ps(1.0 / total);
            for i in (0..n).step_by(8) {
                _mm256_storeu_ps(p.add(i), _mm256_mul_ps(_mm256_loadu_ps(p.add(i)), inv));
            }
            for x in &mut row[n..] {
                *x /= total;
            }
        }
    }

    pub fn accumulate(out: &mut [f32], v: &[f16], d: usize, w: &[f32]) {
        let group = out.len() / d;
        let kv_len = w.len() / group;
        out.fill(0.0);
        unsafe {
            for start in (0..kv_len).step_by(B) {
                let end = (start + B).min(kv_len);
                for (h, o) in out.chunks_exact_mut(d).enumerate() {
                    let w = &w[h * kv_len + start..h * kv_len + end];
                    for t in (0..d).step_by(T) {
                        let mut acc = [_mm256_setzero_ps(); T / 8];
                        for (i, a) in acc.iter_mut().enumerate() {
                            *a = _mm256_loadu_ps(o.as_ptr().add(t + i * 8));
                        }
                        for (p, w) in w.iter().enumerate() {
                            let w = _mm256_set1_ps(*w);
                            let row = v.as_ptr().add((start + p) * d + t);
                            for (i, a) in acc.iter_mut().enumerate() {
                                *a = _mm256_fmadd_ps(w, load(row.add(i * 8)), *a);
                            }
                        }
                        for (i, a) in acc.iter().enumerate() {
                            _mm256_storeu_ps(o.as_mut_ptr().add(t + i * 8), *a);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma",
    target_feature = "f16c"
)))]
mod simd {
    use half::f16;

    pub fn scores(q: &[f32], k: &[f16], d: usize, scale: f32, out: &mut [f32]) {
        let kv_len = out.len() / (q.len() / d);
        for (q, out) in q.chunks_exact(d).zip(out.chunks_exact_mut(kv_len)) {
            for (o, k) in out.iter_mut().zip(k.chunks_exact(d)) {
                *o = q.iter().zip(k).map(|(a, b)| a * b.to_f32()).sum::<f32>() * scale;
            }
        }
    }

    pub fn softmax(row: &mut [f32]) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for x in row.iter_mut() {
            *x = (*x - max).exp();
            sum += *x;
        }
        let inv = 1.0 / sum;
        row.iter_mut().for_each(|x| *x *= inv);
    }

    pub fn accumulate(out: &mut [f32], v: &[f16], d: usize, w: &[f32]) {
        let kv_len = w.len() / (out.len() / d);
        for (o, w) in out.chunks_exact_mut(d).zip(w.chunks_exact(kv_len)) {
            o.fill(0.0);
            for (w, v) in w.iter().zip(v.chunks_exact(d)) {
                o.iter_mut().zip(v).for_each(|(o, v)| *o += w * v.to_f32());
            }
        }
    }
}

fn decode(q: &[f32], cache: &KvCache, s: &Shape, scale: f32) -> Vec<f32> {
    let d = s.head_dim;
    let kv_len = s.offset + 1;
    let group = s.heads / s.kv_heads;
    let mut ctx = vec![0f32; s.heads * d];
    let ptr = ctx.as_mut_ptr() as usize;
    let next = AtomicUsize::new(0);
    barrier_pool().execute(|_| loop {
        let g = next.fetch_add(1, Ordering::Relaxed);
        if g >= s.kv_heads {
            break;
        }
        let out = unsafe {
            std::slice::from_raw_parts_mut((ptr as *mut f32).add(g * group * d), group * d)
        };
        let q = &q[g * group * d..(g + 1) * group * d];
        let mut scores = vec![0f32; group * kv_len];
        simd::scores(q, &cache.k[g][..kv_len * d], d, scale, &mut scores);
        scores.chunks_exact_mut(kv_len).for_each(simd::softmax);
        simd::accumulate(out, &cache.v[g][..kv_len * d], d, &scores);
    });
    ctx
}

pub fn causal(q: &[f32], cache: &KvCache, s: &Shape, scale: f32) -> Vec<f32> {
    if s.queries == 1 {
        return decode(q, cache, s, scale);
    }
    let d = s.head_dim;
    let l = s.queries;
    let kv_len = s.offset + l;
    let group = s.heads / s.kv_heads;
    let rows = group * l;
    let groups: Vec<Vec<f32>> = (0..s.kv_heads)
        .into_par_iter()
        .map(|g| {
            let mut k = vec![0f32; kv_len * d];
            let mut v = vec![0f32; kv_len * d];
            cache.k[g][..kv_len * d].convert_to_f32_slice(&mut k);
            cache.v[g][..kv_len * d].convert_to_f32_slice(&mut v);
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
                    k.as_ptr(),
                    d as isize,
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
                let (live, masked) = row.split_at_mut(s.offset + r % l + 1);
                simd::softmax(live);
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
                    v.as_ptr(),
                    1,
                    d as isize,
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
    ctx
}
