use std::{
    pin::Pin,
    sync::{Arc, Mutex as StdMutex},
};

use hanashi::chat::{EncodingConfig, hanashi::config::HanashiConfig};
use shoji::{
    traits::backend::{
        Error, Instance as BackendInstance, InstanceStream, NoMetricsStream, State,
        chat_token::{Instance as TokenInstance, StreamInput, StreamOutput, TokenStreamMetrics},
    },
    types::{
        basic::{ReasoningEffort, SamplingParameters},
        model::{ModelAccessibility, ModelSource},
    },
};
use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel};

use super::*;

struct EmptyState;
impl State for EmptyState {}

struct RecordedStream {
    input: Vec<u64>,
    prefix_length: Option<usize>,
}

// Scripts the sampled tokens and records what the token session hands to the backend.
struct Recorder {
    tokenizer: Arc<Tokenizer>,
    streams: StdMutex<Vec<RecordedStream>>,
}

impl Recorder {
    fn record<'a>(
        &'a self,
        input: &StreamInput,
        prefix_length: Option<usize>,
    ) -> Pin<
        Box<dyn InstanceStream<Item = Result<StreamOutput, Error>, Metrics = Option<TokenStreamMetrics>> + Send + 'a>,
    > {
        self.streams.lock().unwrap().push(RecordedStream {
            input: input.clone(),
            prefix_length,
        });
        let tokens = self.tokenizer.encode("Thought.\n</think>\n\nAnswer<|im_end|>", false).unwrap().get_ids().to_vec();
        Box::pin(NoMetricsStream::new(futures::stream::iter(
            tokens.into_iter().map(|id| Ok(StreamOutput::Token(u64::from(id)))),
        )))
    }
}

impl BackendInstance for Recorder {
    type StreamConfig = ChatReplyConfig;
    type StreamInput = StreamInput;
    type StreamOutput = StreamOutput;
    type StreamMetrics = Option<TokenStreamMetrics>;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn State>, Error>> + Send + '_>> {
        Box::pin(async { Ok(Box::new(EmptyState) as Box<dyn State>) })
    }

    fn stream<'a>(
        &'a self,
        input: &'a StreamInput,
        _state: &'a mut dyn State,
        _config: ChatReplyConfig,
        _cancel_token: CancellationToken,
    ) -> Pin<Box<dyn InstanceStream<Item = Result<StreamOutput, Error>, Metrics = Self::StreamMetrics> + Send + 'a>>
    {
        self.record(input, None)
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        None
    }
}

impl TokenInstance for Recorder {
    fn with_config(
        &self,
        _config: ChatConfig,
    ) -> Box<dyn TokenInstance> {
        Box::new(Self {
            tokenizer: self.tokenizer.clone(),
            streams: StdMutex::new(Vec::new()),
        })
    }

    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }

    fn max_context_length(&self) -> Option<usize> {
        None
    }

    fn stop_token_ids(&self) -> Option<Box<[u64]>> {
        Some(Box::new([u64::from(self.tokenizer.token_to_id("<|im_end|>").unwrap())]))
    }

    fn sampling_defaults(&self) -> SamplingParameters {
        SamplingParameters::default()
    }

    fn stream_with_prefix_cache<'a>(
        &'a self,
        input: &'a StreamInput,
        _state: &'a mut dyn State,
        _config: ChatReplyConfig,
        _cancel_token: CancellationToken,
        prefix_length: usize,
    ) -> Pin<Box<dyn InstanceStream<Item = Result<StreamOutput, Error>, Metrics = Self::StreamMetrics> + Send + 'a>>
    {
        self.record(input, Some(prefix_length))
    }
}

async fn session(kv_cache_enabled: bool) -> (ChatSession, Arc<Recorder>) {
    let mut config = HanashiConfig::Qwen38.resolve().unwrap();
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let vocabulary: [(String, u32); 256] = alphabet
        .into_iter()
        .enumerate()
        .map(|(index, value)| (value.to_string(), index as u32))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let mut tokenizer = Tokenizer::new(BPE::builder().vocab_and_merges(vocabulary, Vec::new()).build().unwrap());
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
    tokenizer.add_tokens(&["system", "user", "assistant"].map(|value| AddedToken::from(value, false)));
    let mut markers = config.parsing.framing_config().tokens;
    markers.extend(["<bos>".to_string(), "<eos>".to_string(), "<|im_end|>".to_string()]);
    tokenizer.add_special_tokens(&markers.into_iter().map(|value| AddedToken::from(value, true)).collect::<Vec<_>>());
    config.tokens.bos_token_id = tokenizer.token_to_id("<bos>");
    config.tokens.eos_token_id = tokenizer.token_to_id("<eos>");
    let backend = Arc::new(Recorder {
        tokenizer: Arc::new(tokenizer),
        streams: StdMutex::new(Vec::new()),
    });
    let mut encoding = serde_json::to_value(EncodingConfig::Hanashi {
        config: HanashiConfig::Custom {
            config,
        },
    })
    .unwrap();
    // JinjaConfig omits empty lists on serialization but requires this key when read.
    encoding["rendering"]["jinja"]["required_functions"] = serde_json::json!([]);
    let model = Model::external(
        "test".into(),
        "test".into(),
        "Test".into(),
        "test".into(),
        "Test".into(),
        "1".into(),
        vec![],
        ModelAccessibility::OnDevice {
            source: ModelSource::Filesystem {
                path: String::new(),
            },
        },
        Some(encoding.into()),
    );
    let instance =
        token::Session::with_instance(backend.clone(), String::new(), &model, kv_cache_enabled).await.unwrap();
    let session = ChatSession {
        instance: Arc::new(Mutex::new(Instance::Token(instance))),
        state: Arc::new(Mutex::new(ChatSessionState::Idle)),
        messages: Arc::new(Mutex::new(Vec::new())),
        tool_registry: None,
    };
    (session, backend)
}

fn system() -> ChatMessage {
    ChatMessage::system().with_reasoning_effort(ReasoningEffort::Default).with_text("Stable instructions.".to_string())
}

fn question() -> ChatMessage {
    ChatMessage::user().with_text("Question?".to_string())
}

async fn first_prefix_length(
    kv_cache_enabled: bool,
    messages: Vec<ChatMessage>,
) -> (Option<usize>, Vec<u64>, Arc<Recorder>) {
    let (session, backend) = session(kv_cache_enabled).await;
    session.reply(messages, ChatReplyConfig::default()).await.unwrap();
    let (prefix_length, input) = {
        let streams = backend.streams.lock().unwrap();
        (streams[0].prefix_length, streams[0].input.clone())
    };
    (prefix_length, input, backend)
}

#[tokio::test]
async fn breakpoint_with_a_cache_directory_sets_the_prefix_length() {
    let (prefix_length, input, backend) =
        first_prefix_length(true, vec![system().with_kv_cache_breakpoint(), question()]).await;
    let length = prefix_length.expect("a marked message must set the prefix length");
    assert!(length > 0 && length < input.len());
    let prefix = backend.tokenizer.decode(&input[..length].iter().map(|id| *id as u32).collect::<Vec<_>>(), false);
    // The prefix ends where the user turn opens, so it holds the system prompt but not the question.
    let prefix = prefix.unwrap();
    assert!(prefix.contains("Stable instructions."), "{prefix:?}");
    assert!(!prefix.contains("Question?"), "{prefix:?}");
}

#[tokio::test]
async fn prefix_length_needs_both_the_breakpoint_and_the_cache_directory() {
    let (without_directory, ..) =
        first_prefix_length(false, vec![system().with_kv_cache_breakpoint(), question()]).await;
    assert_eq!(without_directory, None);
    let (without_breakpoint, ..) = first_prefix_length(true, vec![system(), question()]).await;
    assert_eq!(without_breakpoint, None);
}

#[tokio::test]
async fn the_last_breakpoint_wins_and_later_turns_do_not_carry_one() {
    let (session, backend) = session(true).await;
    let messages = vec![
        system().with_kv_cache_breakpoint(),
        ChatMessage::user().with_text("Earlier question?".to_string()).with_kv_cache_breakpoint(),
        ChatMessage::assistant().with_text("Earlier answer.".to_string()),
        question(),
    ];
    session.reply(messages, ChatReplyConfig::default()).await.unwrap();
    let system_only = {
        let (single, ..) = first_prefix_length(true, vec![system().with_kv_cache_breakpoint(), question()]).await;
        single.unwrap()
    };
    let first = backend.streams.lock().unwrap()[0].prefix_length.unwrap();
    assert!(first > system_only, "the breakpoint on the second message is the later one");

    // The session now holds the prefix in its state, so there is nothing left to restore.
    session
        .reply(
            vec![ChatMessage::user().with_text("Follow up".to_string()).with_kv_cache_breakpoint()],
            ChatReplyConfig::default(),
        )
        .await
        .unwrap();
    assert_eq!(backend.streams.lock().unwrap()[1].prefix_length, None);
}
