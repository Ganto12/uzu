//! Disk-backed prefix cache.
//!
//! Persists a [`LanguageModelState`] (attention KV cache, DeltaNet / short-conv recurrent state and
//! accepted tokens) at a prompt-prefix boundary, so that a later session or process can restore it
//! instead of running prefill again.
//!
//! File layout (little endian): 8 magic bytes, u32 format version, u32 header length, the JSON
//! `SnapshotHeader`, the prefix tokens as u32, zero padding up to a 16 KiB boundary, then the
//! payload: the live leading bytes of every state buffer, each at a 256-byte aligned offset in
//! layer order. The file name is a hash of the cache key; the header repeats every component of
//! the key, so a lookup verifies the file instead of trusting its name.
//!
//! Any doubt (format version, model, backend, device, kernels, numerics mode, layer layout, prefix
//! tokens, payload size or checksum) is a [`SnapshotMiss`] and the caller prefills as usual. The
//! payload checksum (xxh3-128) guards against accidental corruption only: anyone who can write to
//! the cache directory can forge a snapshot, so keep it somewhere private.
//!
//! A snapshot holds the prompt token ids and the KV state derived from them. On unix the cache
//! directory is created with mode 0700 and every snapshot with mode 0600; an existing directory
//! keeps its mode.
//!
//! No uninitialised GPU memory reaches disk: padding is written as zeros, unwritten slots of the
//! last partly filled KV key block are zeroed from the KV layout geometry, and the buffers of a
//! never-written recurrent state are saved as zeros. A layout whose geometry is unknown is refused.
//!
//! Device identity is [`Context::device_fingerprint`]; kernel identity is a hash of the kernel
//! sources taken at build time (`UZU_KERNEL_IDENTITY`). When either is unknown, snapshots are
//! refused, because the key could be shared by builds or devices that compute differently.
//!
//! The first cache use of a model hashes all of its weights to build the model fingerprint, which
//! takes a few seconds for a multi-GB file.

use std::{
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Buffer, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
        Context, CpuBuffer, GlobalBufferCpuGpu,
    },
    encodable_block::transformer::TransformerLayerSnapshot,
    engine::language_model::{
        LanguageModel,
        state::{LanguageModelCreateEmptyStateError, LanguageModelState},
        stream::{LanguageModelStreamError, PREFILL_BATCH_SIZE, SamplingMethod},
    },
};

/// Bump on any change to the header schema, the payload order or the snapshot semantics of a mixer.
const SNAPSHOT_FORMAT_VERSION: u32 = 3;
/// Hash of the kernel sources this engine was built from (see `build/identity.rs`), or "unknown".
/// The package version is not enough, since a kernel change can change cached state bit for bit.
const KERNEL_IDENTITY: &str = match option_env!("UZU_KERNEL_IDENTITY") {
    Some(identity) => identity,
    None => "unknown",
};
const UNKNOWN_IDENTITY: &str = "unknown";
#[cfg(unix)]
const DIRECTORY_MODE: u32 = 0o700;
#[cfg(unix)]
const FILE_MODE: u32 = 0o600;
/// Leftover temp files older than this are removed by [`PrefixCache::load_or_prefill`].
const STALE_TEMP_AGE: Duration = Duration::from_secs(60 * 60);
const MAGIC: &[u8; 8] = b"UZUPFX\0\0";
/// Payload starts at a page boundary so it can be read with `F_NOCACHE` / mapped directly.
const PAYLOAD_ALIGNMENT: u64 = 16 * 1024;
/// Each buffer inside the payload starts at this alignment.
const BUFFER_ALIGNMENT: u64 = 256;
/// Largest header a reader accepts; bounds the allocation a corrupt length field can cause.
const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct SnapshotHeader {
    format_version: u32,
    engine_version: String,
    backend: String,
    /// [`Context::device_fingerprint`]: cached state is only valid for the numerics of the device
    /// that produced it.
    device_fingerprint: String,
    kernel_identity: String,
    /// [`Context::batch_invariant`] when the state was created.
    batch_invariant: bool,
    model_fingerprint: String,
    prefix_token_count: u64,
    prefix_tokens_blake3: String,
    context_length: u32,
    last_output_token: Option<u64>,
    layers: Vec<TransformerLayerSnapshot>,
    payload_offset: u64,
    payload_bytes: u64,
    payload_xxh3: String,
}

#[derive(Debug, Error)]
pub enum PrefixCacheError<B: Backend> {
    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("State error: {0}")]
    State(#[from] LanguageModelCreateEmptyStateError<B>),
    #[error("Stream error: {0}")]
    Stream(#[from] LanguageModelStreamError<B>),
    #[error("Unsupported: {0}")]
    Unsupported(String),
}

/// Why a lookup did not restore. Every variant means "prefill normally".
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SnapshotMiss {
    NotFound,
    /// The model or its state cannot be snapshotted, e.g. a DFlash speculator is loaded.
    Unsupported(String),
    BadMagic,
    /// The header is not valid JSON for this format version.
    BadHeader,
    /// The file ended before the header, tokens or payload were complete.
    Truncated,
    FormatVersion(u32),
    EngineVersion(String),
    Backend(String),
    /// The snapshot was produced on a different device (name, core count or MXU support).
    Device(String),
    /// The snapshot was produced by a build with different kernel sources.
    KernelIdentity(String),
    /// The snapshot was produced with a different batch-invariant setting.
    NumericsMode,
    ModelFingerprint,
    PrefixTokens,
    Layout(String),
    PayloadSize,
    PayloadChecksum,
    ContextTooSmall,
}

/// Live state bytes copied to the CPU, with the layout that describes them.
struct CapturedState {
    layers: Vec<TransformerLayerSnapshot>,
    entries: Vec<(u64, u64)>,
    payload_bytes: u64,
    payload: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RestoreTimings {
    /// Open + header parse + token verification.
    pub header: Duration,
    /// Empty state allocation (also paid by a cold prefill).
    pub allocate: Duration,
    /// File read of the payload into a CPU/GPU shared staging buffer.
    pub read: Duration,
    /// Optional payload checksum verification.
    pub verify: Duration,
    /// GPU blit from staging into the state buffers, including wait.
    pub blit: Duration,
    pub total: Duration,
}

#[derive(Clone, Debug)]
pub struct SaveInfo {
    pub path: PathBuf,
    pub file_bytes: u64,
    pub payload_bytes: u64,
    pub duration: Duration,
}

pub struct RestoreOptions {
    /// Verify the xxh3-128 checksum of the payload before using it.
    pub verify_checksum: bool,
}

impl Default for RestoreOptions {
    fn default() -> Self {
        Self {
            verify_checksum: true,
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Tokens as little-endian u32. `None` if any token id does not fit in 32 bits (the format
/// cannot represent it, and truncating would alias different prompts onto one snapshot).
fn tokens_bytes(tokens: &[u64]) -> Option<Vec<u8>> {
    let mut bytes = Vec::with_capacity(tokens.len().checked_mul(4)?);
    for &token in tokens {
        bytes.extend_from_slice(&u32::try_from(token).ok()?.to_le_bytes());
    }
    Some(bytes)
}

/// Header length as the u32 the format stores, or `None` if it exceeds [`MAX_HEADER_BYTES`].
fn header_len_to_u32(len: usize) -> Option<u32> {
    u32::try_from(len).ok().filter(|&len| u64::from(len) <= MAX_HEADER_BYTES)
}

/// Checks a header length read from a file against the cap and the file size.
fn check_header_length(
    header_len: u64,
    file_len: u64,
) -> Result<(), SnapshotMiss> {
    if header_len > MAX_HEADER_BYTES {
        return Err(SnapshotMiss::BadHeader);
    }
    if 16u64.checked_add(header_len).is_none_or(|end| end > file_len) {
        return Err(SnapshotMiss::Truncated);
    }
    Ok(())
}

/// Sets every payload byte outside the live buffer ranges `entries` to zero. The staging buffer
/// the payload is read back from is uninitialised GPU memory, so the gaps between buffers (and
/// anything past the last one) must be cleared before they are hashed or written to disk.
fn zero_payload_gaps(
    payload: &mut [u8],
    entries: &[(u64, u64)],
) {
    let mut cursor = 0usize;
    for &(offset, bytes) in entries {
        let start = (offset as usize).min(payload.len());
        let end = (offset.saturating_add(bytes) as usize).min(payload.len());
        if start > cursor {
            payload[cursor..start].fill(0);
        }
        cursor = cursor.max(end);
    }
    payload[cursor..].fill(0);
}

/// Splits `payload` into one mutable slice per `(offset, bytes)` entry. Entries must be ascending
/// and non-overlapping, which [`payload_layout`] guarantees; out-of-range parts are clamped.
fn split_entries_mut<'a>(
    payload: &'a mut [u8],
    entries: &[(u64, u64)],
) -> Vec<&'a mut [u8]> {
    let mut slices = Vec::with_capacity(entries.len());
    let mut rest = payload;
    let mut consumed = 0u64;
    for &(offset, bytes) in entries {
        let skip = (offset.saturating_sub(consumed) as usize).min(rest.len());
        let (_, tail) = std::mem::take(&mut rest).split_at_mut(skip);
        let take = (bytes as usize).min(tail.len());
        let (slice, tail) = tail.split_at_mut(take);
        slices.push(slice);
        rest = tail;
        consumed = offset.saturating_add(bytes);
    }
    slices
}

/// Checks, before any command buffer exists, that the saved buffers line up with the state's
/// buffers: same count, and every saved range fits its target.
fn validate_restore_ranges(
    entries: &[(u64, u64)],
    buffer_sizes: &[usize],
) -> Result<(), String> {
    if entries.len() != buffer_sizes.len() {
        return Err(format!("buffer count mismatch: saved {}, state {}", entries.len(), buffer_sizes.len()));
    }
    for (index, (&(_, bytes), &size)) in entries.iter().zip(buffer_sizes).enumerate() {
        if usize::try_from(bytes).map_or(true, |bytes| bytes > size) {
            return Err(format!("saved buffer {index} ({bytes} bytes) larger than target ({size} bytes)"));
        }
    }
    Ok(())
}

fn align_up(
    value: u64,
    alignment: u64,
) -> Option<u64> {
    value.div_ceil(alignment).checked_mul(alignment)
}

/// Layout of the payload: `(offset inside payload, bytes)` per buffer, plus the total payload
/// size. `None` if any size computation overflows, which only happens for a hostile or corrupt
/// header.
fn payload_layout(layers: &[TransformerLayerSnapshot]) -> Option<(Vec<(u64, u64)>, u64)> {
    let mut offset = 0u64;
    let mut entries = Vec::new();
    for layer in layers {
        if let TransformerLayerSnapshot::Owned(layout) = layer {
            for &bytes in &layout.buffer_bytes {
                offset = align_up(offset, BUFFER_ALIGNMENT)?;
                entries.push((offset, bytes));
                offset = offset.checked_add(bytes)?;
            }
        }
    }
    Some((entries, offset))
}

/// Streaming BLAKE3 of `config.json` and `model.safetensors`. Hashing every byte means a modified
/// or partially overwritten weight file can never alias an older snapshot.
fn compute_model_fingerprint(model_path: &Path) -> io::Result<[u8; 32]> {
    const CHUNK_BYTES: usize = 8 * 1024 * 1024;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"uzu-model-fingerprint-v2");
    let config = fs::read(model_path.join("config.json"))?;
    hasher.update(&(config.len() as u64).to_le_bytes());
    hasher.update(&config);
    let mut weights = File::open(model_path.join("model.safetensors"))?;
    hasher.update(&weights.metadata()?.len().to_le_bytes());
    let mut chunk = vec![0u8; CHUNK_BYTES];
    loop {
        let read = weights.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        hasher.update(&chunk[..read]);
    }
    Ok(*hasher.finalize().as_bytes())
}

/// BLAKE3 of raw bytes as hex; used to compare logits and states bit for bit.
pub fn content_hash(bytes: &[u8]) -> String {
    hex(blake3::hash(bytes).as_bytes())
}

impl<B: Backend> LanguageModel<B> {
    /// Batch-invariant numerics for this model's engine (see `Context::set_batch_invariant`).
    /// With it on, a snapshot at any 64-token boundary (`SnapshotAlignment::Multiple(64)`)
    /// restores bit-exactly, not only at prefill-batch boundaries.
    pub fn set_batch_invariant(
        &self,
        enabled: bool,
    ) {
        self.engine.context.set_batch_invariant(enabled);
    }

    pub fn batch_invariant(&self) -> bool {
        self.engine.context.batch_invariant()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Identity of the loaded weights and config, computed once on first use by hashing the whole
    /// of `config.json` and `model.safetensors`. This blocks the calling thread for a few seconds
    /// on a multi-GB model, once per loaded model (the first cache use pays it). `Err` carries the
    /// reason the model files could not be read, in which case snapshots are disabled.
    pub fn model_fingerprint(&self) -> Result<&[u8; 32], &str> {
        self.model_fingerprint
            .get_or_init(|| {
                compute_model_fingerprint(&self.model_path)
                    .map_err(|error| format!("model fingerprint unavailable: {error}"))
            })
            .as_ref()
            .map_err(String::as_str)
    }

    /// Prefills `tokens` into `state` without returning a sampled token to the caller.
    /// The sampled next token is kept as `last_output_token`, exactly as after a closed stream.
    pub async fn prefill(
        &self,
        state: &mut LanguageModelState<B>,
        tokens: &[u64],
    ) -> Result<(), LanguageModelStreamError<B>> {
        let mut options = self.default_stream_options();
        options.sampling_method = SamplingMethod::Greedy;
        let mut stream = self.stream(tokens, state, options)?;
        stream.close().await
    }

    fn device_fingerprint(&self) -> String {
        self.engine.context.device_fingerprint()
    }

    /// Model fingerprint if this model, device and build can be snapshotted, else the reason not.
    fn check_snapshot_supported(&self) -> Result<[u8; 32], String> {
        if self.speculator.is_some() {
            return Err("models with a DFlash speculator are not supported yet".to_string());
        }
        if self.device_fingerprint() == UNKNOWN_IDENTITY {
            return Err("device identity unknown".to_string());
        }
        if KERNEL_IDENTITY == UNKNOWN_IDENTITY {
            return Err("kernel identity unknown (build did not provide UZU_KERNEL_IDENTITY)".to_string());
        }
        self.model_fingerprint().copied().map_err(str::to_string)
    }

    /// Copies the live state bytes of `state` into one CPU vector. Shared by the save path and
    /// [`Self::state_digest`]. Errors if the state has unaccepted tokens. Bytes outside the live
    /// buffer ranges are zeroed, never left as staging memory.
    async fn capture_state(
        &self,
        state: &mut LanguageModelState<B>,
    ) -> Result<CapturedState, PrefixCacheError<B>> {
        let context = &self.engine.context;
        if state.transformer_state.context_length() as usize != state.tokens.len() {
            return Err(PrefixCacheError::Unsupported("state has unaccepted tokens".to_string()));
        }
        let layers = state.transformer_state.snapshot_layout().map_err(PrefixCacheError::Unsupported)?;
        let (entries, payload_bytes) = payload_layout(&layers)
            .ok_or_else(|| PrefixCacheError::Unsupported("snapshot payload size overflows".to_string()))?;

        // GPU -> one shared staging buffer, then a single CPU read.
        let mut staging =
            context.create_buffer_gpu(payload_bytes.max(1) as usize).map_err(PrefixCacheError::Backend)?;
        let pending = {
            let buffers = state.transformer_state.snapshot_buffers();
            if entries.len() != buffers.len() {
                return Err(PrefixCacheError::Unsupported(format!(
                    "snapshot buffer count mismatch: layout {}, state {}",
                    entries.len(),
                    buffers.len()
                )));
            }
            let mut command_buffer =
                context.create_command_buffer(Some("prefix cache save"), None).map_err(PrefixCacheError::Backend)?;
            for (buffer, (offset, bytes)) in buffers.iter().zip(&entries) {
                if *bytes == 0 {
                    continue;
                }
                let source = (*buffer).subrange(..*bytes as usize);
                let destination = BufferMut::subrange_mut(&mut staging, *offset as usize..(*offset + *bytes) as usize);
                command_buffer.encode_copy(source, destination);
            }
            command_buffer.end_encoding().submit()
        };
        pending.wait_until_completed().await.map_err(PrefixCacheError::Backend)?;
        let mut payload: Vec<u8> = context.copyout::<u8>(&staging).await.map_err(PrefixCacheError::Backend)?;
        payload.truncate(payload_bytes as usize);
        zero_payload_gaps(&mut payload, &entries);
        let mut buffers = split_entries_mut(&mut payload, &entries);
        if !state.transformer_state.zero_unwritten_snapshot_bytes(&mut buffers) {
            return Err(PrefixCacheError::Unsupported(
                "cannot prove all snapshot bytes are initialised for this KV layout".to_string(),
            ));
        }
        Ok(CapturedState {
            layers,
            entries,
            payload_bytes,
            payload,
        })
    }

    /// BLAKE3 (hex) over the layer layouts and every live snapshot byte of `state` (padding
    /// between buffers excluded). Two states with equal digests hold bit-identical KV caches and
    /// recurrent state; used to check a restore against the state it was saved from. It covers KV
    /// and recurrent state only, not the token list or `last_output_token`.
    pub async fn state_digest(
        &self,
        state: &mut LanguageModelState<B>,
    ) -> Result<String, PrefixCacheError<B>> {
        let captured = self.capture_state(state).await?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(&serde_json::to_vec(&captured.layers)?);
        for (offset, bytes) in &captured.entries {
            hasher.update(&bytes.to_le_bytes());
            hasher.update(&captured.payload[*offset as usize..(*offset + *bytes) as usize]);
        }
        Ok(hex(hasher.finalize().as_bytes()))
    }

    /// Checks that `header` was produced by this model, build and device under the same numerics.
    fn check_header(
        &self,
        header: &SnapshotHeader,
        fingerprint: &[u8; 32],
    ) -> Result<(), SnapshotMiss> {
        if header.format_version != SNAPSHOT_FORMAT_VERSION {
            return Err(SnapshotMiss::FormatVersion(header.format_version));
        }
        if header.engine_version != env!("CARGO_PKG_VERSION") {
            return Err(SnapshotMiss::EngineVersion(header.engine_version.clone()));
        }
        if header.backend != B::NAME {
            return Err(SnapshotMiss::Backend(header.backend.clone()));
        }
        if header.kernel_identity != KERNEL_IDENTITY {
            return Err(SnapshotMiss::KernelIdentity(header.kernel_identity.clone()));
        }
        if header.device_fingerprint != self.device_fingerprint() {
            return Err(SnapshotMiss::Device(header.device_fingerprint.clone()));
        }
        if header.batch_invariant != self.engine.context.batch_invariant() {
            return Err(SnapshotMiss::NumericsMode);
        }
        if header.model_fingerprint != hex(fingerprint) {
            return Err(SnapshotMiss::ModelFingerprint);
        }
        Ok(())
    }

    /// Persists the full state (all accepted tokens) to `path`, atomically via a temp file and a
    /// rename. On unix the parent directory is created with mode 0700 if missing and the file with
    /// mode 0600.
    ///
    /// Returns [`PrefixCacheError::Unsupported`] if the state has unaccepted tokens, a token id
    /// does not fit the file format, the device or kernel identity is unknown, the KV layout
    /// cannot be proven initialised, or the batch-invariant flag changed since the state was
    /// created.
    pub async fn save_state_snapshot(
        &self,
        state: &mut LanguageModelState<B>,
        path: &Path,
    ) -> Result<SaveInfo, PrefixCacheError<B>> {
        let started = Instant::now();
        let fingerprint = self.check_snapshot_supported().map_err(PrefixCacheError::Unsupported)?;
        if state.numerics_mode_changed || state.batch_invariant != self.engine.context.batch_invariant() {
            return Err(PrefixCacheError::Unsupported(
                "batch-invariant flag changed since the state was created".to_string(),
            ));
        }
        let token_bytes = tokens_bytes(&state.tokens)
            .ok_or_else(|| PrefixCacheError::Unsupported("token id above u32::MAX".to_string()))?;
        let context_length = state.transformer_state.context_length();
        let CapturedState {
            layers,
            entries: _,
            payload_bytes,
            payload,
        } = self.capture_state(state).await?;
        let payload = &payload[..];

        let mut header = SnapshotHeader {
            format_version: SNAPSHOT_FORMAT_VERSION,
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            backend: B::NAME.to_string(),
            device_fingerprint: self.device_fingerprint(),
            kernel_identity: KERNEL_IDENTITY.to_string(),
            batch_invariant: state.batch_invariant,
            model_fingerprint: hex(&fingerprint),
            prefix_token_count: state.tokens.len() as u64,
            prefix_tokens_blake3: hex(blake3::hash(&token_bytes).as_bytes()),
            context_length,
            last_output_token: state.last_output_token,
            layers,
            payload_offset: 0,
            payload_bytes,
            payload_xxh3: format!("{:032x}", xxhash_rust::xxh3::xxh3_128(payload)),
        };
        let header_json = encode_header(&mut header, token_bytes.len() as u64)?;
        write_snapshot_file(path, &header, &header_json, &token_bytes, payload)?;

        Ok(SaveInfo {
            path: path.to_path_buf(),
            file_bytes: header.payload_offset + payload_bytes,
            payload_bytes,
            duration: started.elapsed(),
        })
    }

    /// Restores a state saved by [`Self::save_state_snapshot`]. Any mismatch is returned as
    /// `Ok(Err(miss))`, so callers fall back to a normal prefill. A state is only returned fully
    /// restored.
    pub async fn restore_state_snapshot(
        &self,
        path: &Path,
        expected_tokens: &[u64],
        max_context_length: Option<u32>,
        sampling_seed: u64,
        options: &RestoreOptions,
    ) -> Result<Result<(LanguageModelState<B>, RestoreTimings), SnapshotMiss>, PrefixCacheError<B>> {
        let started = Instant::now();
        let mut timings = RestoreTimings::default();
        let fingerprint = match self.check_snapshot_supported() {
            Ok(fingerprint) => fingerprint,
            Err(reason) => return Ok(Err(SnapshotMiss::Unsupported(reason))),
        };

        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Err(SnapshotMiss::NotFound)),
            Err(error) => return Err(error.into()),
        };
        #[cfg(target_vendor = "apple")]
        unsafe {
            // The payload is copied once, so keep it out of the unified buffer cache.
            use std::os::fd::AsRawFd;
            disable_file_cache(file.as_raw_fd());
        }

        let mut fixed = [0u8; 16];
        if file.read_exact(&mut fixed).is_err() || &fixed[..8] != MAGIC {
            return Ok(Err(SnapshotMiss::BadMagic));
        }
        let version = u32::from_le_bytes(fixed[8..12].try_into().unwrap());
        if version != SNAPSHOT_FORMAT_VERSION {
            return Ok(Err(SnapshotMiss::FormatVersion(version)));
        }
        let file_len = file.metadata()?.len();
        let header_len = u32::from_le_bytes(fixed[12..16].try_into().unwrap()) as u64;
        if let Err(miss) = check_header_length(header_len, file_len) {
            return Ok(Err(miss));
        }
        let mut header_json = vec![0u8; header_len as usize];
        if file.read_exact(&mut header_json).is_err() {
            return Ok(Err(SnapshotMiss::Truncated));
        }
        let header: SnapshotHeader = match serde_json::from_slice(&header_json) {
            Ok(header) => header,
            Err(_) => return Ok(Err(SnapshotMiss::BadHeader)),
        };
        if let Err(miss) = self.check_header(&header, &fingerprint) {
            return Ok(Err(miss));
        }
        // Exact token comparison, not just the hash in the file name.
        if header.prefix_token_count != expected_tokens.len() as u64
            || header.context_length as u64 != header.prefix_token_count
        {
            return Ok(Err(SnapshotMiss::PrefixTokens));
        }
        let Some(expected_bytes) = tokens_bytes(expected_tokens) else {
            return Ok(Err(SnapshotMiss::Unsupported("token id above u32::MAX".to_string())));
        };
        let mut saved_tokens = vec![0u8; expected_bytes.len()];
        if file.read_exact(&mut saved_tokens).is_err() {
            return Ok(Err(SnapshotMiss::Truncated));
        }
        if saved_tokens != expected_bytes {
            return Ok(Err(SnapshotMiss::PrefixTokens));
        }
        if max_context_length.is_some_and(|max| (max as u64) < header.prefix_token_count + 1) {
            return Ok(Err(SnapshotMiss::ContextTooSmall));
        }
        let Some((entries, payload_bytes)) = payload_layout(&header.layers) else {
            return Ok(Err(SnapshotMiss::PayloadSize));
        };
        // Everything before the payload (fixed prefix, header, tokens) is already consumed.
        let consumed = 16 + header_len + expected_bytes.len() as u64;
        if payload_bytes != header.payload_bytes
            || header.payload_offset < consumed
            || header.payload_offset.checked_add(payload_bytes) != Some(file_len)
        {
            return Ok(Err(SnapshotMiss::PayloadSize));
        }
        timings.header = started.elapsed();

        let phase = Instant::now();
        let mut state = self.create_empty_state(max_context_length, sampling_seed)?;
        if let Err(reason) = state.transformer_state.restore_layout(&header.layers, header.context_length) {
            return Ok(Err(SnapshotMiss::Layout(reason)));
        }
        // Map sparse KV pages for the restored length (plus one decode slot).
        state.transformer_state.prepare(header.context_length, 1).map_err(PrefixCacheError::Backend)?;
        timings.allocate = phase.elapsed();

        let context = &self.engine.context;
        let phase = Instant::now();
        let mut staging =
            context.create_buffer_cpu_gpu(payload_bytes.max(1) as usize).map_err(PrefixCacheError::Backend)?;
        file.seek(SeekFrom::Start(header.payload_offset))?;
        if file.read_exact(&mut staging.as_slice_mut::<u8>()[..payload_bytes as usize]).is_err() {
            return Ok(Err(SnapshotMiss::Truncated));
        }
        timings.read = phase.elapsed();

        if options.verify_checksum {
            let phase = Instant::now();
            let digest = xxhash_rust::xxh3::xxh3_128(&staging.as_slice::<u8>()[..payload_bytes as usize]);
            timings.verify = phase.elapsed();
            if format!("{digest:032x}") != header.payload_xxh3 {
                return Ok(Err(SnapshotMiss::PayloadChecksum));
            }
        }

        let phase = Instant::now();
        let staging = staging.handoff();
        // All validation happens before the command buffer exists: once encoding starts it always
        // runs to submit and wait, so no Metal encoder is ever dropped unfinished.
        let buffer_sizes: Vec<usize> =
            state.transformer_state.snapshot_buffers().iter().map(|buffer| Buffer::size(*buffer)).collect();
        if let Err(reason) = validate_restore_ranges(&entries, &buffer_sizes) {
            return Ok(Err(SnapshotMiss::Layout(reason)));
        }
        let pending = {
            let mut buffers = state.transformer_state.snapshot_buffers_mut();
            let mut command_buffer =
                context.create_command_buffer(Some("prefix cache restore"), None).map_err(PrefixCacheError::Backend)?;
            for (buffer, (offset, bytes)) in buffers.iter_mut().zip(&entries) {
                if *bytes == 0 {
                    continue;
                }
                // `offset + bytes` cannot overflow: payload_layout was checked above. The target
                // size was validated in `validate_restore_ranges`.
                let source = (&staging).subrange(*offset as usize..(*offset + *bytes) as usize);
                let destination = BufferMut::subrange_mut(&mut **buffer, ..*bytes as usize);
                command_buffer.encode_copy(source, destination);
            }
            command_buffer.end_encoding().submit()
        };
        pending.wait_until_completed().await.map_err(PrefixCacheError::Backend)?;
        drop(staging);
        timings.blit = phase.elapsed();

        state.tokens = expected_tokens.to_vec();
        state.last_output_token = header.last_output_token;
        timings.total = started.elapsed();
        Ok(Ok((state, timings)))
    }
}

/// Serialises `header` with the `payload_offset` that follows from its own length. The offset is a
/// number inside the JSON, so the length is found by iterating to a fixed point.
fn encode_header<B: Backend>(
    header: &mut SnapshotHeader,
    token_bytes: u64,
) -> Result<Vec<u8>, PrefixCacheError<B>> {
    let mut json = serde_json::to_vec(header)?;
    loop {
        let offset = align_up(16 + json.len() as u64 + token_bytes, PAYLOAD_ALIGNMENT)
            .ok_or_else(|| PrefixCacheError::Unsupported("snapshot header size overflows".to_string()))?;
        if offset == header.payload_offset {
            return Ok(json);
        }
        header.payload_offset = offset;
        json = serde_json::to_vec(header)?;
    }
}

/// Writes a snapshot to a unique temp file next to `path` and renames it into place, so readers
/// see either no file or a complete one. A failed write removes its own temp file.
fn write_snapshot_file<B: Backend>(
    path: &Path,
    header: &SnapshotHeader,
    header_json: &[u8],
    token_bytes: &[u8],
    payload: &[u8],
) -> Result<(), PrefixCacheError<B>> {
    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
    let header_len = header_len_to_u32(header_json.len()).ok_or_else(|| {
        PrefixCacheError::Unsupported(format!("snapshot header of {} bytes is too large", header_json.len()))
    })?;
    if let Some(parent) = path.parent() {
        create_private_directory(parent)?;
    }
    let temp_path =
        path.with_extension(format!("tmp.{}.{}", std::process::id(), TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)));
    let write = || -> io::Result<()> {
        let mut file = create_private_file(&temp_path)?;
        file.write_all(MAGIC)?;
        file.write_all(&SNAPSHOT_FORMAT_VERSION.to_le_bytes())?;
        file.write_all(&header_len.to_le_bytes())?;
        file.write_all(header_json)?;
        file.write_all(token_bytes)?;
        let written = 16 + header_json.len() as u64 + token_bytes.len() as u64;
        file.write_all(&vec![0u8; (header.payload_offset - written) as usize])?;
        file.write_all(payload)?;
        file.sync_all()?;
        fs::rename(&temp_path, path)
    };
    if let Err(error) = write() {
        let _ = fs::remove_file(&temp_path);
        return Err(error.into());
    }
    Ok(())
}

/// Creates `directory` (and missing ancestors) with mode 0700 on unix. An existing directory is
/// not touched, so its mode is never loosened.
fn create_private_directory(directory: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(DIRECTORY_MODE);
    }
    builder.create(directory)
}

/// Creates a new file (never opens an existing one) with mode 0600 on unix.
fn create_private_file(path: &Path) -> io::Result<File> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(FILE_MODE);
    }
    options.open(path)
}

/// True for the names `save_state_snapshot` gives its temp files: `<32 hex>.tmp.<pid>.<counter>`.
fn is_temp_snapshot_name(name: &str) -> bool {
    let Some((stem, rest)) = name.split_once(".tmp.") else {
        return false;
    };
    let Some((pid, counter)) = rest.split_once('.') else {
        return false;
    };
    let digits = |text: &str| !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    stem.len() == 32 && stem.bytes().all(|byte| byte.is_ascii_hexdigit()) && digits(pid) && digits(counter)
}

/// Removes temp snapshot files in `directory` (only that directory, only regular files matching
/// our temp naming) that were last modified more than `max_age` ago: leftovers of a crashed save.
/// Best effort; returns how many were removed.
fn remove_stale_temp_files(
    directory: &Path,
    max_age: Duration,
) -> usize {
    let Ok(entries) = fs::read_dir(directory) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_str().is_some_and(is_temp_snapshot_name) {
            continue;
        }
        let Ok(metadata) = fs::symlink_metadata(entry.path()) else {
            continue;
        };
        let old_enough =
            metadata.modified().ok().and_then(|modified| modified.elapsed().ok()).is_some_and(|age| age > max_age);
        if metadata.is_file() && old_enough && fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

#[cfg(target_vendor = "apple")]
unsafe fn disable_file_cache(fd: i32) {
    unsafe extern "C" {
        fn fcntl(
            fd: i32,
            cmd: i32,
            ...
        ) -> i32;
    }
    const F_NOCACHE: i32 = 48;
    unsafe {
        fcntl(fd, F_NOCACHE, 1);
    }
}

/// Where a snapshot is taken relative to the requested breakpoint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SnapshotAlignment {
    /// Round the breakpoint down to a multiple of [`PREFILL_BATCH_SIZE`] (1,024). Prefill batches
    /// are counted from the start of each stream call, so a snapshot at an aligned length
    /// followed by one stream over the rest reproduces the batch shapes of a single uncached
    /// prefill. Such snapshots are exact by construction: greedy output matches the uncached path
    /// token for token. This is the only alignment with that guarantee on default kernels.
    #[default]
    PrefillBatch,
    /// Snapshot exactly at the breakpoint. Saves more prefill, but the remainder is prefilled in
    /// different batch shapes than a single uncached prefill would use, so output can differ at
    /// numerical near-ties. Exactness is not guaranteed on default kernels (the result is
    /// identical to sending the prefix as an earlier turn of the same session, which is what
    /// uzu's in-session reuse already does).
    Exact,
    /// Round down to a multiple of this many tokens. A diagnostic for batch invariance: with
    /// 64 (the DeltaNet chunk size) the chunk boundaries line up, so any remaining drift comes
    /// from kernels whose reduction plan depends on the batch length.
    Multiple(usize),
}

impl SnapshotAlignment {
    /// Number of leading tokens of a `breakpoint`-token prefix that a snapshot covers.
    pub fn snapshot_length(
        self,
        breakpoint: usize,
    ) -> usize {
        match self {
            SnapshotAlignment::PrefillBatch => breakpoint / PREFILL_BATCH_SIZE * PREFILL_BATCH_SIZE,
            SnapshotAlignment::Exact => breakpoint,
            SnapshotAlignment::Multiple(granule) => breakpoint / granule.max(1) * granule.max(1),
        }
    }
}

/// Outcome of [`PrefixCache::load_or_prefill`].
#[derive(Clone, Debug)]
pub enum PrefixCacheOutcome {
    Hit(RestoreTimings),
    Miss {
        reason: SnapshotMiss,
        prefill: Duration,
        save: Option<SaveInfo>,
        /// Why persisting the snapshot failed (full disk, unsupported state), if it did. A failed
        /// save never fails the request; it only means the next call misses again.
        save_error: Option<String>,
    },
    /// The aligned snapshot length is zero (prefix shorter than one prefill batch): nothing to
    /// cache, the returned state is empty.
    TooShort,
}

/// Directory of prefix snapshots, keyed by blake3(model fingerprint, format version, kernel
/// identity, backend, device fingerprint, batch-invariant flag, tokens). The header repeats each
/// component and is checked on restore, so a hash collision or a copied file is a miss, not a
/// wrong restore.
///
/// Snapshots contain prompt token ids and KV state, so on unix the directory is created with
/// mode 0700 and each file with mode 0600; an existing directory keeps whatever mode it has.
pub struct PrefixCache {
    pub directory: PathBuf,
    pub restore_options: RestoreOptions,
    pub alignment: SnapshotAlignment,
}

impl PrefixCache {
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
            restore_options: RestoreOptions::default(),
            alignment: SnapshotAlignment::default(),
        }
    }

    /// File for a snapshot of exactly `snapshot_tokens`. `None` if the model cannot be snapshotted
    /// (unfingerprintable weights, unknown device or kernel identity) or a token id does not fit
    /// the format.
    pub fn path_for<B: Backend>(
        &self,
        model: &LanguageModel<B>,
        snapshot_tokens: &[u64],
    ) -> Option<PathBuf> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&model.check_snapshot_supported().ok()?);
        hasher.update(&SNAPSHOT_FORMAT_VERSION.to_le_bytes());
        hasher.update(&(B::NAME.len() as u64).to_le_bytes());
        hasher.update(B::NAME.as_bytes());
        hasher.update(&(KERNEL_IDENTITY.len() as u64).to_le_bytes());
        hasher.update(KERNEL_IDENTITY.as_bytes());
        let device_fingerprint = model.device_fingerprint();
        hasher.update(&(device_fingerprint.len() as u64).to_le_bytes());
        hasher.update(device_fingerprint.as_bytes());
        hasher.update(&[model.engine.context.batch_invariant() as u8]);
        hasher.update(&tokens_bytes(snapshot_tokens)?);
        Some(self.directory.join(format!("{}.uzkv", &hex(hasher.finalize().as_bytes())[..32])))
    }

    /// Returns a state holding the first `self.alignment.snapshot_length(prefix_tokens.len())`
    /// tokens of `prefix_tokens`: restored from disk on a hit, otherwise prefilled and saved. The
    /// caller streams `input[state.tokens.len()..]` next, so the uncovered tail of the prefix and
    /// the suffix are prefilled together.
    ///
    /// A failed save is reported in [`PrefixCacheOutcome::Miss`] and never fails the call. With
    /// batch-invariant numerics and an alignment other than [`SnapshotAlignment::PrefillBatch`],
    /// stream at least 2 tokens after a restore: a 1-token stream takes the single-row decode path,
    /// whose numerics differ from batched prefill.
    pub async fn load_or_prefill<B: Backend>(
        &self,
        model: &LanguageModel<B>,
        prefix_tokens: &[u64],
        max_context_length: Option<u32>,
        sampling_seed: u64,
    ) -> Result<(LanguageModelState<B>, PrefixCacheOutcome), PrefixCacheError<B>> {
        // Drop leftovers of crashed saves first, so they never accumulate.
        remove_stale_temp_files(&self.directory, STALE_TEMP_AGE);
        let snapshot_tokens = &prefix_tokens[..self.alignment.snapshot_length(prefix_tokens.len())];
        if snapshot_tokens.is_empty() {
            return Ok((model.create_empty_state(max_context_length, sampling_seed)?, PrefixCacheOutcome::TooShort));
        }
        let path = self.path_for(model, snapshot_tokens);
        let reason = match &path {
            None => SnapshotMiss::Unsupported(match model.check_snapshot_supported() {
                Err(reason) => reason,
                Ok(_) => "cannot key snapshot: token id above u32::MAX".to_string(),
            }),
            Some(path) => match model
                .restore_state_snapshot(path, snapshot_tokens, max_context_length, sampling_seed, &self.restore_options)
                .await?
            {
                Ok((state, timings)) => return Ok((state, PrefixCacheOutcome::Hit(timings))),
                Err(reason) => reason,
            },
        };
        let started = Instant::now();
        let mut state = model.create_empty_state(max_context_length, sampling_seed)?;
        model.prefill(&mut state, snapshot_tokens).await?;
        let prefill = started.elapsed();
        let (save, save_error) = match (&reason, &path) {
            (SnapshotMiss::Unsupported(_), _) | (_, None) => (None, None),
            (_, Some(path)) => match model.save_state_snapshot(&mut state, path).await {
                Ok(info) => (Some(info), None),
                Err(error) => (None, Some(error.to_string())),
            },
        };
        Ok((
            state,
            PrefixCacheOutcome::Miss {
                reason,
                prefill,
                save,
                save_error,
            },
        ))
    }
}

#[cfg(test)]
#[path = "../../../unit/engine/language_model/prefix_cache_test.rs"]
mod tests;
