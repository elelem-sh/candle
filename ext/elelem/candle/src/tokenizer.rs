use anyhow::{anyhow, bail, Result};
use candle_core::quantized::gguf_file::{Content, Value};
use tokenizers::models::bpe::{Vocab, BPE};
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

const QWEN2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub fn from_gguf(ct: &Content) -> Result<Tokenizer> {
    let model = text(ct, "tokenizer.ggml.model")?;
    let pre = text(ct, "tokenizer.ggml.pre")?;
    if model != "gpt2" || pre != "qwen2" {
        bail!("unsupported tokenizer {model}/{pre}, only gpt2/qwen2");
    }
    let tokens = strings(ct, "tokenizer.ggml.tokens")?;
    let merges = strings(ct, "tokenizer.ggml.merges")?
        .into_iter()
        .map(|m| {
            m.split_once(' ')
                .map(|(a, b)| (a.to_string(), b.to_string()))
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| anyhow!("malformed tokenizer.ggml.merges"))?;
    let types = list(ct, "tokenizer.ggml.token_type")?
        .iter()
        .map(|v| v.to_i32().map_err(Into::into))
        .collect::<Result<Vec<_>>>()?;

    let vocab = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect::<Vocab>();
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| anyhow!(e))?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer
        .with_normalizer(Some(NFC))
        .map_err(|e| anyhow!(e))?
        .with_pre_tokenizer(Some(Sequence::new(vec![
            Split::new(
                SplitPattern::Regex(QWEN2.into()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map_err(|e| anyhow!(e))?
            .into(),
            ByteLevel::new(false, false, false).into(),
        ])))
        .with_decoder(Some(ByteLevel::new(false, false, false)));

    let added = |kind: i32| {
        tokens
            .iter()
            .zip(&types)
            .filter(|(_, t)| **t == kind)
            .map(|(s, _)| AddedToken::from(s.clone(), kind == 3).normalized(false))
            .collect::<Vec<_>>()
    };
    tokenizer
        .add_special_tokens(added(3))
        .map_err(|e| anyhow!(e))?;
    tokenizer.add_tokens(added(4)).map_err(|e| anyhow!(e))?;
    Ok(tokenizer)
}

fn list<'a>(ct: &'a Content, key: &str) -> Result<&'a Vec<Value>> {
    ct.metadata
        .get(key)
        .ok_or_else(|| anyhow!("missing {key}"))?
        .to_vec()
        .map_err(Into::into)
}

fn strings(ct: &Content, key: &str) -> Result<Vec<String>> {
    list(ct, key)?
        .iter()
        .map(|v| v.to_string().cloned().map_err(Into::into))
        .collect()
}

pub fn text(ct: &Content, key: &str) -> Result<String> {
    Ok(ct
        .metadata
        .get(key)
        .ok_or_else(|| anyhow!("missing {key}"))?
        .to_string()?
        .clone())
}
