use std::path::PathBuf;

use clap::Parser;
use serde::Deserialize;

use crate::bench::BenchSampling;

#[derive(Debug, Parser)]
pub struct Args {
    pub model: PathBuf,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub model: String,
    pub sampling: BenchSampling,
}
