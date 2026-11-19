use anyhow::{bail, Result};
use candle_core::quantized::gguf_file::{Content, TensorInfo, Value, VersionedMagic};
use candle_core::quantized::GgmlDType;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

const IQ4_XS: u32 = 23;
const IQ4_XS_BLOCK: usize = 136;
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

pub struct Weights {
    file: BufReader<File>,
    file_len: u64,
    extra: Vec<u8>,
    pos: u64,
}

impl Read for Weights {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos < self.file_len {
            let n = buf.len().min((self.file_len - self.pos) as usize);
            let n = self.file.read(&mut buf[..n])?;
            self.pos += n as u64;
            return Ok(n);
        }
        let start = (self.pos - self.file_len) as usize;
        let n = buf.len().min(self.extra.len().saturating_sub(start));
        buf[..n].copy_from_slice(&self.extra[start..start + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for Weights {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        let end = self.file_len + self.extra.len() as u64;
        self.pos = match to {
            SeekFrom::Start(n) => n,
            SeekFrom::End(n) => end.saturating_add_signed(n),
            SeekFrom::Current(n) => self.pos.saturating_add_signed(n),
        };
        if self.pos < self.file_len {
            self.file.seek(SeekFrom::Start(self.pos))?;
        }
        Ok(self.pos)
    }
}

pub fn open(path: &str) -> Result<(Content, Weights)> {
    let mut file = BufReader::new(File::open(path)?);
    let file_len = file.get_ref().metadata()?.len();
    let magic = match u32le(&mut file)? {
        0x46554747 => match u32le(&mut file)? {
            2 => VersionedMagic::GgufV2,
            3 => VersionedMagic::GgufV3,
            v => bail!("{path}: unsupported gguf version {v}"),
        },
        _ => bail!("{path}: not a gguf file"),
    };
    let tensor_count = u64le(&mut file)?;
    let kv_count = u64le(&mut file)?;
    let mut metadata = HashMap::new();
    for _ in 0..kv_count {
        let key = string(&mut file)?;
        let kind = u32le(&mut file)?;
        metadata.insert(key, value(&mut file, kind)?);
    }
    let mut raw = Vec::new();
    for _ in 0..tensor_count {
        let name = string(&mut file)?;
        let dims = u32le(&mut file)?;
        let mut shape = (0..dims)
            .map(|_| u64le(&mut file).map(|d| d as usize))
            .collect::<Result<Vec<_>>>()?;
        shape.reverse();
        let dtype = u32le(&mut file)?;
        let offset = u64le(&mut file)?;
        raw.push((name, shape, dtype, offset));
    }
    let alignment = match metadata.get("general.alignment") {
        Some(v) => v.to_u32()? as u64,
        None => 32,
    };
    let tensor_data_offset = file.stream_position()?.div_ceil(alignment) * alignment;

    let mut extra = Vec::new();
    let mut tensor_infos = HashMap::new();
    for (name, shape, dtype, offset) in raw {
        let (ggml_dtype, offset) = if dtype == IQ4_XS {
            let blocks = shape.iter().product::<usize>() / 256;
            let mut bytes = vec![0u8; blocks * IQ4_XS_BLOCK];
            file.seek(SeekFrom::Start(tensor_data_offset + offset))?;
            file.read_exact(&mut bytes)?;
            let at = file_len + extra.len() as u64 - tensor_data_offset;
            iq4_xs_to_q8_0(&bytes, &mut extra);
            (GgmlDType::Q8_0, at)
        } else {
            (dtype_from(dtype)?, offset)
        };
        tensor_infos.insert(
            name,
            TensorInfo {
                ggml_dtype,
                shape: shape.into(),
                offset,
            },
        );
    }
    file.seek(SeekFrom::Start(0))?;
    let content = Content {
        magic,
        metadata,
        tensor_infos,
        tensor_data_offset,
    };
    Ok((
        content,
        Weights {
            file,
            file_len,
            extra,
            pos: 0,
        },
    ))
}

fn dtype_from(id: u32) -> Result<GgmlDType> {
    Ok(match id {
        0 => GgmlDType::F32,
        1 => GgmlDType::F16,
        2 => GgmlDType::Q4_0,
        3 => GgmlDType::Q4_1,
        6 => GgmlDType::Q5_0,
        7 => GgmlDType::Q5_1,
        8 => GgmlDType::Q8_0,
        9 => GgmlDType::Q8_1,
        10 => GgmlDType::Q2K,
        11 => GgmlDType::Q3K,
        12 => GgmlDType::Q4K,
        13 => GgmlDType::Q5K,
        14 => GgmlDType::Q6K,
        15 => GgmlDType::Q8K,
        30 => GgmlDType::BF16,
        id => bail!("unsupported ggml type {id}"),
    })
}

fn iq4_xs_to_q8_0(src: &[u8], dst: &mut Vec<u8>) {
    for block in src.chunks_exact(IQ4_XS_BLOCK) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..];
        for ib in 0..8 {
            let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf;
            let high = ((scales_h >> (2 * ib)) & 3) as u8;
            let scale = d * ((low | (high << 4)) as i32 - 32) as f32;
            dst.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
            let q = &qs[ib * 16..ib * 16 + 16];
            dst.extend(q.iter().map(|b| KVALUES_IQ4NL[(b & 0xf) as usize] as u8));
            dst.extend(q.iter().map(|b| KVALUES_IQ4NL[(b >> 4) as usize] as u8));
        }
    }
}

fn value<R: Read>(r: &mut R, kind: u32) -> Result<Value> {
    Ok(match kind {
        0 => Value::U8(bytes::<1, R>(r)?[0]),
        1 => Value::I8(bytes::<1, R>(r)?[0] as i8),
        2 => Value::U16(u16::from_le_bytes(bytes(r)?)),
        3 => Value::I16(i16::from_le_bytes(bytes(r)?)),
        4 => Value::U32(u32le(r)?),
        5 => Value::I32(i32::from_le_bytes(bytes(r)?)),
        6 => Value::F32(f32::from_le_bytes(bytes(r)?)),
        7 => Value::Bool(bytes::<1, R>(r)?[0] != 0),
        8 => Value::String(string(r)?),
        9 => {
            let kind = u32le(r)?;
            let len = u64le(r)?;
            Value::Array((0..len).map(|_| value(r, kind)).collect::<Result<_>>()?)
        }
        10 => Value::U64(u64le(r)?),
        11 => Value::I64(i64::from_le_bytes(bytes(r)?)),
        12 => Value::F64(f64::from_le_bytes(bytes(r)?)),
        k => bail!("unknown gguf value type {k}"),
    })
}

fn bytes<const N: usize, R: Read>(r: &mut R) -> Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}

fn u32le<R: Read>(r: &mut R) -> Result<u32> {
    Ok(u32::from_le_bytes(bytes(r)?))
}

fn u64le<R: Read>(r: &mut R) -> Result<u64> {
    Ok(u64::from_le_bytes(bytes(r)?))
}

fn string<R: Read>(r: &mut R) -> Result<String> {
    let mut b = vec![0u8; u64le(r)? as usize];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}
