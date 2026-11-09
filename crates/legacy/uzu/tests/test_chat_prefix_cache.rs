#![cfg(not(target_family = "wasm"))]

use std::{path::Path, sync::Arc};

use hanashi::chat::{EncodingConfig, hanashi::config::HanashiConfig};
use shoji::traits::Backend;
use uzu::{
    engine::bridge::UzuLlmBackend,
    session::chat::{ChatInstance, ChatSession},
    types::{
        basic::{ReasoningEffort, SamplingMethod, Value},
        model::{Model, ModelAccessibility, ModelSource, ModelSpecialization},
        session::chat::{ChatConfig, ChatContentBlock, ChatMessage, ChatReplyConfig},
    },
};

struct Run {
    time_to_first_token: f64,
    tokens_count_input: u32,
    tokens_count_output: u32,
    content: Vec<ChatContentBlock>,
}

fn qwen_model(model_path: &str) -> Model {
    let backend = UzuLlmBackend::new();
    let encoding = serde_json::to_value(EncodingConfig::Hanashi {
        config: HanashiConfig::Qwen35,
    })
    .unwrap();
    Model::external(
        "qwen3.5-0.8b-local".to_string(),
        "local".to_string(),
        "Local".to_string(),
        shoji::traits::Backend::identifier(&backend),
        shoji::traits::Backend::identifier(&backend),
        shoji::traits::Backend::version(&backend),
        vec![ModelSpecialization::Chat {}],
        ModelAccessibility::OnDevice {
            source: ModelSource::Filesystem {
                path: model_path.to_string(),
            },
        },
        Some(Value::from(encoding)),
    )
}

fn cache_files(directory: &Path) -> Vec<String> {
    let mut files: Vec<_> = std::fs::read_dir(directory)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.file_name().to_string_lossy().to_string())
                .filter(|name| name.ends_with(".uzkv"))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

/// A prompt well past two prefill batches (1,024 tokens each), so the snapshot covers 2,048 tokens or more.
fn long_instructions() -> String {
    (0..300).map(|i| format!("Entry {i}: the archive stores {} records.\n", i * 7 % 101)).collect()
}

async fn reply(
    instance: &ChatInstance,
    breakpoint: bool,
) -> Run {
    let session = ChatSession::with_instance(instance).await.unwrap();
    let system = ChatMessage::system()
        .with_reasoning_effort(ReasoningEffort::Disabled)
        .with_text(format!("Answer from the archive below.\n{}", long_instructions()));
    let messages = vec![
        if breakpoint {
            system.with_kv_cache_breakpoint()
        } else {
            system
        },
        ChatMessage::user().with_text("How many records does entry 5 store? Answer in one sentence.".to_string()),
    ];
    let config = ChatReplyConfig::default().with_token_limit(Some(24)).with_sampling_method(SamplingMethod::Greedy {});
    let replies = session.reply(messages, config).await.unwrap();
    let last = replies.last().unwrap();
    Run {
        time_to_first_token: last.stats.time_to_first_token.expect("no first token"),
        tokens_count_input: last.stats.tokens_count_input.unwrap(),
        tokens_count_output: last.stats.tokens_count_output.unwrap(),
        content: last.message.content.clone(),
    }
}

#[ignore]
#[tokio::test]
async fn test_chat_kv_cache_breakpoint_round_trip() {
    let Ok(model_path) = std::env::var("UZU_TEST_MODEL_PATH") else {
        eprintln!("UZU_TEST_MODEL_PATH is not set, skipping");
        return;
    };
    let cache_directory = tempfile::tempdir().unwrap();
    let config =
        ChatConfig::default().with_kv_cache_directory(Some(cache_directory.path().to_string_lossy().to_string()));
    let backend: Arc<dyn Backend> = Arc::new(UzuLlmBackend::new());
    let instance = ChatInstance::new(backend, config, qwen_model(&model_path), Some(model_path)).await.unwrap();

    // Without a breakpoint nothing is cached; this run also warms the kernels up for the timings below.
    let uncached = reply(&instance, false).await;
    assert!(uncached.tokens_count_input >= 2048, "prompt too short: {}", uncached.tokens_count_input);
    assert!(cache_files(cache_directory.path()).is_empty(), "no breakpoint must not write a snapshot");

    let first = reply(&instance, true).await;
    let files = cache_files(cache_directory.path());
    assert_eq!(files.len(), 1, "the first run must save one snapshot, found {files:?}");

    let second = reply(&instance, true).await;
    assert_eq!(cache_files(cache_directory.path()), files, "the second run must hit the saved snapshot");

    println!(
        "TTFT: first (miss and save) {:.3}s, second (hit) {:.3}s, no breakpoint {:.3}s; {} input tokens",
        first.time_to_first_token, second.time_to_first_token, uncached.time_to_first_token, first.tokens_count_input
    );
    assert!(
        second.time_to_first_token < first.time_to_first_token,
        "a restored prefix must start faster than prefilling and saving it"
    );
    for run in [&first, &second] {
        assert_eq!(run.tokens_count_output, uncached.tokens_count_output);
        assert_eq!(run.content, uncached.content, "greedy output must match the uncached run");
    }
    println!("greedy output identical across the three runs: {} tokens", uncached.tokens_count_output);
}
