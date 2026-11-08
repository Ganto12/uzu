use uzu_engine_macros::uzu_config;

#[uzu_config]
pub struct ReasoningConfig {
    pub default_reasoning_effort: String,
    pub field_name: String,
    pub reasoning_effort_to_field_value: serde_json::Map<String, serde_json::Value>,
}

// TODO: These are not respected, we use our own configs (see
// `hanashi/configs`). They're just here to parse the format. This should be
// unified eventually.
#[uzu_config]
pub struct ToolCallsConfig {
    tags: Box<[String]>,
    function_pattern: Option<String>,
    parameter_pattern: Option<String>,
}

#[uzu_config]
pub struct ResponseFormatConfig {
    pub reasoning_pattern: Option<String>,
    pub prefixes: Box<[String]>,
    turn_pattern: Option<String>,
    tool_calls: Option<ToolCallsConfig>,
}

#[uzu_config(super::TokenCodecConfig)]
pub struct ChatCodecConfig {
    pub prompt_template: String,
    pub response_format: Option<ResponseFormatConfig>,
    pub system_role_name: String,
    pub user_role_name: String,
    pub assistant_role_name: String,
    pub eos_token: Option<String>,
    pub bos_token: Option<String>,
    pub default_system_prompt: Option<String>,
    #[serde(default)] // TODO: ??? inconsistency in converted models in registry workaround, remove when resolved
    pub reasoning_config: Option<ReasoningConfig>,
}

impl ChatCodecConfig {
    pub fn end_of_thinking_tag(&self) -> Option<&str> {
        self.response_format.as_ref()?.prefixes.last().map(String::as_str)
    }
}
