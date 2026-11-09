use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use walkdir::WalkDir;

/// Source directories (relative to the crate root) whose contents define the compiled kernels.
const IDENTITY_DIRECTORIES: &[&str] = &[
    "build",
    "src/backends/common/gpu_types",
    "src/backends/metal/gpu_types",
    "src/backends/metal/kernel",
    "src/backends/cpu/kernel",
];

/// Emits `UZU_KERNEL_IDENTITY`: a hash over the sorted contents of every kernel source file
/// (and the build scripts that generate from them). It changes whenever kernel code changes,
/// unlike `CARGO_PKG_VERSION`. Uses blake3 when the build has it, std's DefaultHasher otherwise.
pub fn emit_kernel_identity() -> anyhow::Result<()> {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?);
    let mut files: Vec<PathBuf> = Vec::new();
    for directory in IDENTITY_DIRECTORIES {
        let directory = crate_dir.join(directory);
        if !directory.exists() {
            continue;
        }
        println!("cargo::rerun-if-changed={}", directory.display());
        for entry in WalkDir::new(&directory) {
            let entry = entry.with_context(|| format!("cannot walk {}", directory.display()))?;
            if entry.file_type().is_file() {
                files.push(entry.into_path());
            }
        }
    }
    files.sort();
    files.dedup();

    let mut hasher = IdentityHasher::new();
    for file in &files {
        let relative = file.strip_prefix(&crate_dir).unwrap_or(Path::new(file));
        hasher.update(relative.as_os_str().as_encoded_bytes());
        hasher.update(b"\0");
        hasher.update(&fs::read(file).with_context(|| format!("cannot read {}", file.display()))?);
        hasher.update(b"\0");
    }
    println!("cargo::rustc-env=UZU_KERNEL_IDENTITY={}", hasher.finish_hex());
    Ok(())
}

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "webgpu"))]
struct IdentityHasher(blake3::Hasher);

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "webgpu"))]
impl IdentityHasher {
    fn new() -> Self {
        Self(blake3::Hasher::new())
    }

    fn update(
        &mut self,
        bytes: &[u8],
    ) {
        self.0.update(bytes);
    }

    fn finish_hex(self) -> String {
        self.0.finalize().to_hex().to_string()
    }
}

#[cfg(not(any(all(feature = "metal", target_os = "macos"), feature = "webgpu")))]
struct IdentityHasher(std::collections::hash_map::DefaultHasher);

#[cfg(not(any(all(feature = "metal", target_os = "macos"), feature = "webgpu")))]
impl IdentityHasher {
    fn new() -> Self {
        Self(std::collections::hash_map::DefaultHasher::new())
    }

    fn update(
        &mut self,
        bytes: &[u8],
    ) {
        use std::hash::Hasher;
        self.0.write(bytes);
    }

    fn finish_hex(self) -> String {
        use std::hash::Hasher;
        format!("{:016x}", self.0.finish())
    }
}
