use std::{pin::Pin, sync::Arc};

use tokenizers::Tokenizer;
use tokio_util::sync::CancellationToken;

use crate::{
    traits::backend::{Error, Instance as InstanceTrait, InstanceStream, State},
    types::{
        basic::SamplingParameters,
        session::chat::{ChatConfig, ChatReplyConfig},
    },
};

pub enum TokenStreamOutput {
    LimitReached, // This should be just end of stream
    Token(u64),
}

#[derive(Debug, Clone, Default)]
pub struct TokenStreamMetrics {
    pub num_prefill_forward_passes: usize,
    pub num_decode_forward_passes: usize,
    pub num_tokens_prefilled: usize,
    pub num_tokens_proposed: usize,
    pub num_tokens_accepted: usize,
    pub num_tokens_returned: usize,
}

pub type StreamInput = Vec<u64>;
pub type StreamOutput = TokenStreamOutput;
pub type StreamMetrics = Option<TokenStreamMetrics>;

pub trait Backend: Send + Sync {
    fn instance<'a>(
        &'a self,
        reference: String,
        config: ChatConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn Instance>, Error>> + Send + 'a>>;
}

pub trait Instance:
    InstanceTrait<
        StreamConfig = ChatReplyConfig,
        StreamInput = StreamInput,
        StreamOutput = StreamOutput,
        StreamMetrics = StreamMetrics,
    >
{
    fn with_config(
        &self,
        config: ChatConfig,
    ) -> Box<dyn Instance>;

    fn tokenizer(&self) -> Arc<Tokenizer>;

    fn max_context_length(&self) -> Option<usize>;

    fn stop_token_ids(&self) -> Option<Box<[u64]>>;

    fn sampling_defaults(&self) -> SamplingParameters;

    /// Like `stream`, for an `input` whose first `prefix_length` tokens end at a KV cache breakpoint.
    /// A backend with a KV cache directory restores those tokens from it, or prefills and saves them
    /// there, instead of prefilling them in the stream. The default streams the whole input.
    fn stream_with_prefix_cache<'a>(
        &'a self,
        input: &'a StreamInput,
        state: &'a mut dyn State,
        config: ChatReplyConfig,
        cancel_token: CancellationToken,
        _prefix_length: usize,
    ) -> Pin<Box<dyn InstanceStream<Item = Result<StreamOutput, Error>, Metrics = StreamMetrics> + Send + 'a>> {
        self.stream(input, state, config, cancel_token)
    }
}
