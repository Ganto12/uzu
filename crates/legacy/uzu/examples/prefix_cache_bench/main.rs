//! Benchmark and correctness gate for the disk prefix cache (`uzu_engine::engine::language_model::prefix_cache`).
//!
//! ```text
//! prefix_cache_bench <model_dir> <cache_dir> <mode> [arguments]
//! ```
//!
//! Modes:
//!
//! - `bench <prefix_lens> <runs> <gen_tokens>`: for each comma-separated prefix length, times a cold
//!   prefill, a two-stage cold prefill and a restore from disk, and checks that the restored
//!   greedy output matches both cold paths. Prints a Markdown table on stdout, details on stderr.
//! - `save <prefix_len> <tokens_file> <gen_tokens>`, `restore <prefix_len> <tokens_file> <gen_tokens>`
//!   and `nocache <prefix_len> <tokens_file> <gen_tokens>`: the same questions answered by a
//!   process that saves the snapshot, one that restores it, and one without a cache. The generated
//!   tokens go to `<tokens_file>`; running the three in separate processes shows that a restore
//!   works across processes. `restore` fails if it does not hit.
//! - `negative <prefix_len>`: corrupts or mismatches a saved snapshot ten ways and prints one
//!   `negative/<case>: miss <reason> (ok)` line each. Any restore of a bad file prints `HIT (BAD)`.
//! - `statehash <prefix_len>`: compares the state digest of a prefill with that of its restore and
//!   prints `MATCH (ok)` or `MISMATCH (BAD)`.
//! - `long <prefix_len> <prompts> <gen_tokens> <max_context>`: long-context run over real documents,
//!   uncached against restored, as a Markdown table. The length must be a multiple of 1,024.
//! - `factory <records.jsonl> <cache|cache-default-kernels|invariant> <granule>`: writes eval factory
//!   records (floor, bits, tokens, perplexity, speed) for one comparison. `cache` compares an
//!   uncached prefill with a restore at a multiple of `<granule>` tokens (batch-invariant kernels
//!   when the granule is below 1,024), `invariant` compares default with batch-invariant kernels,
//!   `cache-default-kernels` is `cache` on default kernels at any granule.
//! - `dumplogits <out_dir> <default|invariant|split64> <probes>`: writes the first-answer logits
//!   (f32, little endian) of fixed document probes. Run it with `UZU_BACKEND=cpu` and `default`
//!   for the common reference.
//! - `freezecorpus <corpus.jsonl>`: writes the document pool to a JSONL file for `PROBE_CORPUS`.
//!
//! Environment:
//!
//! - `PREFIX_ALIGN`: `exact` or a token count; sets the snapshot alignment (default: prefill batch).
//! - `PROBE_CORPUS`: frozen document pool (see `freezecorpus`). Without it, Markdown files from the
//!   local cargo registry are used.
//! - `COLD_RESTORE`: evict the snapshot files from the OS file cache before each restore.
//! - `SHARD=<k>/<n>`, `PROBE_MIN`, `PROBE_SPAN`: probe selection for `dumplogits`.
//! - `MODEL_ID`, `FACTORY_CONFIG_SHA`, `ENGINE_COMMIT`, `PROBE_CORPUS_SHA`: provenance recorded by `factory`.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use uzu_engine::{
    backends::{Backend, BackendSelection, select_backend},
    engine::{
        Engine,
        language_model::{
            LanguageModel,
            prefix_cache::{
                PrefixCache, PrefixCacheOutcome, RestoreOptions, SnapshotAlignment, SnapshotMiss, content_hash,
            },
            state::LanguageModelState,
            stream::SamplingMethod,
        },
    },
};

mod checks;
mod factory;
mod long;
mod timing;

use checks::*;
use factory::*;
use long::*;
use timing::*;

pub(crate) const MAX_CONTEXT: u32 = 65536;
pub(crate) const SEED: u64 = 1234;

pub(crate) const QUESTIONS: &[&str] = &[
    "Summarize the reference material in two sentences.",
    "What programming language is used here? Answer briefly.",
    "List three key concepts mentioned above.",
    "Write a one-line haiku about the material.",
    "What is the very first word of the reference material?",
];

pub(crate) fn corpus(repo: &Path) -> String {
    let files = [
        "README.md",
        "crates/uzu-engine/src/engine/language_model/stream/stream.rs",
        "crates/uzu-engine/src/encodable_block/mixer/delta_net.rs",
        "crates/uzu-engine/src/encodable_block/mixer/attention/state.rs",
        "crates/uzu-engine/src/encodable_block/transformer.rs",
    ];
    files
        .iter()
        .map(|file| std::fs::read_to_string(repo.join(file)).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub(crate) fn encode<B: Backend>(
    model: &LanguageModel<B>,
    text: &str,
) -> Vec<u64> {
    model.tokenizer().encode(text, false).unwrap().get_ids().iter().map(|id| *id as u64).collect()
}

pub(crate) fn prefix_tokens<B: Backend>(
    model: &LanguageModel<B>,
    length: usize,
) -> Vec<u64> {
    // Real documents, no repetition: uzu's own README and source, then Markdown documents from
    // the local cargo registry until the requested length is covered.
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let mut text = format!(
        "<|im_start|>system\nYou are a helpful assistant. Answer questions about this reference material.\n\n{}",
        corpus(&repo)
    );
    for (name, body) in document_pool() {
        if text.len() > length * 6 {
            break;
        }
        text.push_str(&format!("\n\n## Document: {name}\n\n{body}"));
    }
    let mut tokens = encode(model, &text);
    assert!(tokens.len() >= length, "corpus too short: {} < {}", tokens.len(), length);
    tokens.truncate(length);
    tokens
}

pub(crate) fn suffix_tokens<B: Backend>(
    model: &LanguageModel<B>,
    question: &str,
) -> Vec<u64> {
    encode(
        model,
        &format!("<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    )
}

/// Streams `input` into `state` and returns (time to first token, generated tokens).
pub(crate) async fn generate<B: Backend>(
    model: &LanguageModel<B>,
    state: &mut LanguageModelState<B>,
    input: &[u64],
    count: usize,
) -> (Duration, Vec<u64>) {
    let started = Instant::now();
    let mut options = model.default_stream_options();
    options.sampling_method = SamplingMethod::Greedy;
    let mut stream = model.stream(input, state, options).unwrap();
    let mut tokens = Vec::new();
    let mut ttft = Duration::ZERO;
    while tokens.len() < count {
        match stream.generate().await.unwrap() {
            Some(token) => {
                if tokens.is_empty() {
                    ttft = started.elapsed();
                }
                tokens.push(token);
            },
            None => break,
        }
    }
    stream.close().await.unwrap();
    (ttft, tokens)
}

pub(crate) fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

pub(crate) fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

pub(crate) struct Bench {
    model_dir: PathBuf,
    cache_dir: PathBuf,
    args: Vec<String>,
}

impl BackendSelection for Bench {
    type Output = ();
    type Error = String;

    async fn select<B: Backend>(self) -> Result<(), String> {
        let load_started = Instant::now();
        let engine = Engine::<B>::new().await.map_err(|error| error.to_string())?;
        let model = engine.load_language_model(&self.model_dir).await.map_err(|error| error.to_string())?;
        eprintln!("[bench] backend={} model loaded in {:.0} ms", B::NAME, ms(load_started.elapsed()));
        let mut cache = PrefixCache::new(&self.cache_dir);
        match std::env::var("PREFIX_ALIGN").as_deref() {
            Ok("exact") => cache.alignment = SnapshotAlignment::Exact,
            Ok(granule) if granule.parse::<usize>().is_ok() => {
                cache.alignment = SnapshotAlignment::Multiple(granule.parse().unwrap())
            },
            _ => {},
        }
        eprintln!("[bench] alignment={:?} max_context_length={:?}", cache.alignment, model.max_context_length());
        let mode = self.args[0].as_str();
        match mode {
            "bench" => bench(&model, &cache, &self.args[1..]).await,
            "save" | "restore" | "nocache" => cross_process(&model, &cache, mode, &self.args[1..]).await,
            "negative" => negative(&model, &cache, &self.args[1..]).await,
            "statehash" => statehash(&model, &cache, &self.args[1..]).await,
            "long" => long(&model, &cache, &self.args[1..]).await,
            "factory" => factory(&model, &self.cache_dir, &self.args[1..]).await,
            "dumplogits" => dump_logits(&model, &self.cache_dir, &self.args[1..]).await,
            "freezecorpus" => {
                let lines: Vec<String> = document_pool()
                    .into_iter()
                    .map(|(name, text)| serde_json::json!({ "name": name, "text": text }).to_string())
                    .collect();
                std::fs::write(&self.args[1], lines.join("\n") + "\n").map_err(|error| error.to_string())
            },
            _ => Err(format!("unknown mode {mode}")),
        }
    }
}

pub(crate) fn hex_prefix(bytes: &[u8; 32]) -> String {
    bytes[..4].iter().map(|byte| format!("{byte:02x}")).collect()
}

#[tokio::main]
pub(crate) async fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("usage: prefix_cache_bench <model_dir> <cache_dir> <mode> ...".to_string());
    }
    let bench = Bench {
        model_dir: PathBuf::from(&args[0]),
        cache_dir: PathBuf::from(&args[1]),
        args: args[2..].to_vec(),
    };
    select_backend(bench, "unknown backend".to_string()).await
}
