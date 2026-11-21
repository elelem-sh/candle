use crate::attention::{self, KvCache, Shape};
use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_core::{bail, DType, Device, Module, Result, Storage, Tensor};
use candle_nn::Embedding;
use candle_transformers::quantized_nn::RmsNorm;
use std::io::{Read, Seek};
use std::sync::Arc;

struct Gguf<'a, R: Read + Seek> {
    ct: gguf_file::Content,
    reader: &'a mut R,
    device: Device,
}

impl<R: Read + Seek> Gguf<'_, R> {
    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(self.reader, name, &self.device)
    }

    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        QMatMul::from_arc(Arc::new(self.tensor(name)?))
    }

    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        RmsNorm::from_qtensor(self.tensor(name)?, eps)
    }

    fn u32(&self, key: &str) -> Result<usize> {
        match self.ct.metadata.get(key) {
            Some(v) => Ok(v.to_u32()? as usize),
            None => bail!("cannot find {key} in metadata"),
        }
    }

    fn f32(&self, key: &str) -> Result<f64> {
        match self.ct.metadata.get(key) {
            Some(v) => Ok(v.to_f32()? as f64),
            None => bail!("cannot find {key} in metadata"),
        }
    }
}

fn cpu_slice(t: &Tensor) -> Result<Vec<f32>> {
    let t = t.contiguous()?;
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        Storage::Cpu(cpu) => Ok(cpu.as_slice::<f32>()?
            [layout.start_offset()..layout.start_offset() + t.elem_count()]
            .to_vec()),
        _ => bail!("expected cpu storage"),
    }
}

struct Rotary {
    sin: Tensor,
    cos: Tensor,
}

impl Rotary {
    fn new(dtype: DType, head_dim: usize, max: usize, theta: f64, dev: &Device) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, len)?.to_dtype(q.dtype())?;
        Ok((
            candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?,
            candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}

struct Attention {
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    o: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary: Arc<Rotary>,
    cache: KvCache,
}

impl Attention {
    fn forward(&mut self, x: &Tensor, offset: usize, p: &mut Profile) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let hidden = self.heads * self.head_dim;
        let (q, k, v) = p.time(MATMUL, || {
            Ok((self.q.forward(x)?, self.k.forward(x)?, self.v.forward(x)?))
        })?;
        let (q, k) = p.time(NORM_ROPE, || {
            let q = q
                .reshape((b, l, self.heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b, l, self.kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let q = self.q_norm.forward(&q.flatten(0, 2)?)?.reshape((
                b,
                self.heads,
                l,
                self.head_dim,
            ))?;
            let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((
                b,
                self.kv_heads,
                l,
                self.head_dim,
            ))?;
            self.rotary.apply(&q, &k, offset)
        })?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        p.time(KV_COPY, || {
            let v = v
                .reshape((b, l, self.kv_heads, self.head_dim))?
                .squeeze(0)?;
            let k_rows = cpu_slice(&k.squeeze(0)?.transpose(0, 1)?)?;
            let v_rows = cpu_slice(&v)?;
            self.cache.append(&k_rows, &v_rows, l);
            Ok(())
        })?;
        let ctx = p.time(ATTENTION, || {
            let shape = Shape {
                heads: self.heads,
                kv_heads: self.kv_heads,
                head_dim: self.head_dim,
                queries: l,
                offset,
            };
            let ctx = attention::causal(&cpu_slice(&q)?, &self.cache, &shape, scale)?;
            Tensor::from_vec(ctx, (b, l, hidden), x.device())
        })?;
        p.time(MATMUL, || ctx.reshape((b, l, hidden))?.apply(&self.o))
    }
}

struct Layer {
    attn: Attention,
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl Layer {
    fn forward(&mut self, x: &Tensor, offset: usize, p: &mut Profile) -> Result<Tensor> {
        let h = p.time(NORM_ROPE, || self.ln1.forward(x))?;
        let h = self.attn.forward(&h, offset, p)?;
        let x = (x + h)?;
        let h = p.time(NORM_ROPE, || self.ln2.forward(&x))?;
        let (gate, up) = p.time(MATMUL, || {
            Ok((self.gate.forward(&h)?, self.up.forward(&h)?))
        })?;
        let gated = p.time(OTHER, || gate.silu()? * up)?;
        let down = p.time(MATMUL, || self.down.forward(&gated))?;
        x + down
    }
}

pub const MATMUL: usize = 0;
pub const ATTENTION: usize = 1;
pub const KV_COPY: usize = 2;
pub const NORM_ROPE: usize = 3;
pub const LM_HEAD: usize = 4;
pub const OTHER: usize = 5;
pub const SECTIONS: [&str; 6] = [
    "matmul",
    "attention",
    "kv_copy",
    "norm_rope",
    "lm_head",
    "other",
];

#[derive(Default)]
pub struct Profile {
    pub on: bool,
    pub ms: [f64; 6],
}

impl Profile {
    fn time<T>(&mut self, section: usize, f: impl FnOnce() -> Result<T>) -> Result<T> {
        if !self.on {
            return f();
        }
        let started = std::time::Instant::now();
        let out = f();
        self.ms[section] += started.elapsed().as_secs_f64() * 1000.0;
        out
    }
}

pub struct ModelWeights {
    embed: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: QMatMul,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let dtype = match ct.metadata.get("general.dtype").map(|v| v.to_u32()) {
            Some(Ok(0)) => DType::F32,
            _ => DType::F16,
        };
        let mut gg = Gguf {
            ct,
            reader,
            device: device.clone(),
        };
        let heads = gg.u32("qwen3.attention.head_count")?;
        let kv_heads = gg.u32("qwen3.attention.head_count_kv")?;
        let head_dim = gg.u32("qwen3.attention.key_length")?;
        let blocks = gg.u32("qwen3.block_count")?;
        let hidden = gg.u32("qwen3.embedding_length")?;
        let max = gg.u32("qwen3.context_length")?;
        let eps = gg.f32("qwen3.attention.layer_norm_rms_epsilon")?;
        let theta = gg.f32("qwen3.rope.freq_base")?;

        let embed = Embedding::new(gg.tensor("token_embd.weight")?.dequantize(device)?, hidden);
        let rotary = Arc::new(Rotary::new(dtype, head_dim, max, theta, device)?);
        let mut layers = Vec::with_capacity(blocks);
        for i in 0..blocks {
            let p = format!("blk.{i}");
            layers.push(Layer {
                ln1: gg.rms_norm(&format!("{p}.attn_norm.weight"), eps)?,
                ln2: gg.rms_norm(&format!("{p}.ffn_norm.weight"), eps)?,
                attn: Attention {
                    q: gg.qmatmul(&format!("{p}.attn_q.weight"))?,
                    k: gg.qmatmul(&format!("{p}.attn_k.weight"))?,
                    v: gg.qmatmul(&format!("{p}.attn_v.weight"))?,
                    o: gg.qmatmul(&format!("{p}.attn_output.weight"))?,
                    q_norm: gg.rms_norm(&format!("{p}.attn_q_norm.weight"), eps)?,
                    k_norm: gg.rms_norm(&format!("{p}.attn_k_norm.weight"), eps)?,
                    heads,
                    kv_heads,
                    head_dim,
                    rotary: rotary.clone(),
                    cache: KvCache::new(kv_heads, head_dim),
                },
                gate: gg.qmatmul(&format!("{p}.ffn_gate.weight"))?,
                up: gg.qmatmul(&format!("{p}.ffn_up.weight"))?,
                down: gg.qmatmul(&format!("{p}.ffn_down.weight"))?,
            });
        }
        let norm = gg.rms_norm("output_norm.weight", eps)?;
        let lm_head = match gg.tensor("output.weight") {
            Ok(t) => t,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        Ok(Self {
            embed,
            layers,
            norm,
            lm_head: QMatMul::from_arc(Arc::new(lm_head))?,
        })
    }

    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.attn.cache.len())
    }

    pub fn truncate(&mut self, len: usize) {
        for layer in &mut self.layers {
            layer.attn.cache.truncate(len);
        }
    }

    pub fn forward(
        &mut self,
        input: &Tensor,
        p: &mut Profile,
        stop: &mut dyn FnMut() -> bool,
    ) -> Result<Option<Tensor>> {
        let offset = self.len();
        let (b, l) = input.dims2()?;
        if b != 1 {
            bail!("batch size {b} is not supported");
        }
        let mut h = self.embed.forward(input)?;
        for layer in &mut self.layers {
            if stop() {
                self.truncate(offset);
                return Ok(None);
            }
            h = match layer.forward(&h, offset, p) {
                Ok(h) => h,
                Err(e) => {
                    self.truncate(offset);
                    return Err(e);
                }
            };
        }
        p.time(LM_HEAD, || {
            let h = self.norm.forward(&h.narrow(1, l - 1, 1)?)?;
            self.lm_head.forward(&h)?.squeeze(1)
        })
        .map(Some)
    }
}
