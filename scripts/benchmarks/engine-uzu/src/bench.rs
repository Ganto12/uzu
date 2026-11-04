use serde::{Deserialize, Serialize};
use serde_json::Value;
use uzu::types::basic::ReasoningEffort;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<Value>>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GreedySampling {
    pub greedy: monostate::MustBe!(true),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StochasticSampling {
    #[serde(deserialize_with = "Option::deserialize")]
    pub top_k: Option<i32>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub top_p: Option<f32>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub min_p: Option<f32>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub temp: Option<f32>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub repetition_penalty: Option<f32>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub suffix_repetition_length: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BenchSampling {
    Greedy(GreedySampling),
    Stochastic(StochasticSampling),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,

    pub max_output_tokens: Option<usize>,
    pub speculative_depth: Option<usize>,
    pub reasoning_effort: Option<ReasoningEffort>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResponse {
    pub text: String,
    pub finish_reason: FinishReason,
    pub prompt_tokens_count: usize,
    pub tokens_count: usize,
    pub time_to_first_token: f64,
    pub prompt_tps: f64,
    pub decode_tps: f64,
    pub tokens_per_forward_pass: Option<f64>,
    pub duration: f64,
    pub energy_gpu: f64,
    pub memory_phys_footprint: u64,
    pub memory_resident: u64,
    pub memory_graphics_total: u64,
}
