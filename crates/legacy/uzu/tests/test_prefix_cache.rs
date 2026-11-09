#![cfg(not(target_family = "wasm"))]

use std::path::PathBuf;

use uzu_engine::{
    backends::{Backend, BackendSelection, select_backend},
    engine::{
        Engine,
        language_model::{
            LanguageModel,
            prefix_cache::{PrefixCache, PrefixCacheOutcome, SnapshotMiss},
            state::LanguageModelState,
            stream::SamplingMethod,
        },
    },
};

const PREFIX_LENGTH: usize = 1536;
const SEED: u64 = 1234;

struct PrefixCacheScenario {
    model_path: PathBuf,
    cache_directory: PathBuf,
}

impl BackendSelection for PrefixCacheScenario {
    type Output = ();
    type Error = String;

    async fn select<B: Backend>(self) -> Result<(), String> {
        let engine = Engine::<B>::new().await.map_err(|error| error.to_string())?;
        let model = engine.load_language_model(&self.model_path).await.map_err(|error| error.to_string())?;
        run(&model, PrefixCache::new(&self.cache_directory)).await
    }
}

fn encode<B: Backend>(
    model: &LanguageModel<B>,
    text: &str,
) -> Vec<u64> {
    model.tokenizer().encode(text, false).unwrap().get_ids().iter().map(|id| *id as u64).collect()
}

async fn digest<B: Backend>(
    model: &LanguageModel<B>,
    state: &mut LanguageModelState<B>,
) -> String {
    model.state_digest(state).await.map_err(|error| error.to_string()).unwrap()
}

async fn continuation<B: Backend>(
    model: &LanguageModel<B>,
    state: &mut LanguageModelState<B>,
    input: &[u64],
) -> Vec<u64> {
    let mut options = model.default_stream_options();
    options.sampling_method = SamplingMethod::Greedy;
    let mut stream = model.stream(input, state, options).unwrap();
    let mut tokens = Vec::new();
    while tokens.len() < 16 {
        match stream.generate().await.unwrap() {
            Some(token) => tokens.push(token),
            None => break,
        }
    }
    stream.close().await.unwrap();
    tokens
}

async fn run<B: Backend>(
    model: &LanguageModel<B>,
    cache: PrefixCache,
) -> Result<(), String> {
    let text: String = (0..400).map(|i| format!("Entry {i}: the cache stores {} tokens.\n", i * 7 % 101)).collect();
    let mut prefix = encode(model, &text);
    assert!(prefix.len() >= PREFIX_LENGTH, "prefix too short: {}", prefix.len());
    prefix.truncate(PREFIX_LENGTH);
    let suffix = encode(model, "\nQuestion: what does the cache store?\nAnswer:");

    // Reference: an uncached prefill of the snapshot span, and an uncached run of the whole input.
    let snapshot = cache.alignment.snapshot_length(prefix.len());
    let mut fresh = model.create_empty_state(None, SEED).unwrap();
    model.prefill(&mut fresh, &prefix[..snapshot]).await.map_err(|e| e.to_string())?;
    let fresh_digest = digest(model, &mut fresh).await;
    let mut uncached = model.create_empty_state(None, SEED).unwrap();
    let expected = continuation(model, &mut uncached, &[&prefix[..], &suffix[..]].concat()).await;

    // First call misses and saves.
    let (mut state, outcome) = cache.load_or_prefill(model, &prefix, None, SEED).await.map_err(|e| e.to_string())?;
    assert!(
        matches!(
            outcome,
            PrefixCacheOutcome::Miss {
                reason: SnapshotMiss::NotFound,
                save: Some(_),
                ..
            }
        ),
        "expected a miss that saves, got {outcome:?}"
    );
    assert_eq!(digest(model, &mut state).await, fresh_digest);

    // Second call, with a fresh state, restores from disk.
    let (mut state, outcome) = cache.load_or_prefill(model, &prefix, None, SEED).await.map_err(|e| e.to_string())?;
    assert!(matches!(outcome, PrefixCacheOutcome::Hit(_)), "expected a hit, got {outcome:?}");
    assert_eq!(state.tokens().len(), snapshot);
    assert_eq!(digest(model, &mut state).await, fresh_digest);
    let rest = [&prefix[snapshot..], &suffix[..]].concat();
    assert_eq!(continuation(model, &mut state, &rest).await, expected);

    // A flipped payload byte is a checksum miss that falls back to prefill.
    let path = cache.path_for(model, &prefix[..snapshot]).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(&path, bytes).unwrap();
    let (mut state, outcome) = cache.load_or_prefill(model, &prefix, None, SEED).await.map_err(|e| e.to_string())?;
    assert!(
        matches!(
            outcome,
            PrefixCacheOutcome::Miss {
                reason: SnapshotMiss::PayloadChecksum,
                ..
            }
        ),
        "expected a checksum miss, got {outcome:?}"
    );
    assert_eq!(digest(model, &mut state).await, fresh_digest);
    Ok(())
}

#[ignore]
#[tokio::test]
async fn test_prefix_cache_round_trip() {
    let Ok(model_path) = std::env::var("UZU_TEST_MODEL_PATH") else {
        eprintln!("UZU_TEST_MODEL_PATH is not set, skipping");
        return;
    };
    let cache_directory = tempfile::tempdir().unwrap();
    let scenario = PrefixCacheScenario {
        model_path: PathBuf::from(model_path),
        cache_directory: cache_directory.path().to_path_buf(),
    };
    select_backend(scenario, "unknown backend".to_string()).await.unwrap();
}
