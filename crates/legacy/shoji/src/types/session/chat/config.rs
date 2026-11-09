use serde::{Deserialize, Serialize};

use crate::types::basic::{ContextLength, SamplingSeed};

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ChatConfig {
    pub context_length: ContextLength,
    pub sampling_seed: SamplingSeed,
    /// Directory for saved prompt prefixes. Messages marked with
    /// `ChatMessage::with_kv_cache_breakpoint` are saved there and restored by later sessions.
    #[serde(default)]
    pub kv_cache_directory: Option<String>,
}

#[bindings::export(Implementation)]
impl ChatConfig {
    #[bindings::export(Method(Factory))]
    pub fn create() -> Self {
        Self::default()
    }
}

#[bindings::export(Implementation)]
impl ChatConfig {
    #[bindings::export(Method)]
    pub fn with_context_length(
        &self,
        context_length: ContextLength,
    ) -> Self {
        Self {
            context_length,
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_sampling_seed(
        &self,
        sampling_seed: SamplingSeed,
    ) -> Self {
        Self {
            sampling_seed,
            ..self.clone()
        }
    }

    #[bindings::export(Method)]
    pub fn with_kv_cache_directory(
        &self,
        kv_cache_directory: Option<String>,
    ) -> Self {
        Self {
            kv_cache_directory,
            ..self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_directory_defaults_to_none_and_survives_serde() {
        let config = ChatConfig::create();
        assert_eq!(config.kv_cache_directory, None);

        let config = config.with_kv_cache_directory(Some("/tmp/kv".to_string()));
        let restored: ChatConfig = serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(restored, config);

        let mut old = serde_json::to_value(ChatConfig::create()).unwrap();
        old.as_object_mut().unwrap().remove("kv_cache_directory");
        let old: ChatConfig = serde_json::from_value(old).unwrap();
        assert_eq!(old.kv_cache_directory, None);
    }
}
