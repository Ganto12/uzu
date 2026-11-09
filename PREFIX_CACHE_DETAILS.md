# Persistent prefill state for uzu: full write-up

Prototype for [trymirai/uzu#792](https://github.com/trymirai/uzu/issues/792): prefill a long shared prefix (system
prompt, tool definitions, a document) once, save the state to disk, restore it in any new session or process.

**Result:** M1 Pro, Qwen 3.5 0.8B, 8K-token prefix: 3.66 s -> 82 ms to first token in a fresh process. Restored state
is byte-identical to an uncached prefill at prefill-batch boundaries; any mismatched or damaged file falls back to a
normal prefill.

Setup: Apple M1 Pro 16 GB, internal SSD (~6-7 GB/s), Metal without MXU; Qwen 3.5 0.8B and LFM2.5 230M in uzu's 4-bit
`mirai-m:4` format; KV cached in uzu's own int8 + bf16-scale format, so cached and uncached numerics are identical.
Greedy unless stated.

## Architecture

```
  ChatSession ── message.with_kv_cache_breakpoint()      config.with_kv_cache_directory()
       │
       │ prefix tokens (rounded down to the 1,024-token prefill batch)
       ▼
  PrefixCache::load_or_prefill
       ├── hit  ─▶ restore_state_snapshot: read file ─▶ check key + xxh3 ─▶ GPU copy
       ├── miss ─▶ prefill ─▶ save_state_snapshot (temp file + rename, 0600)
       ▼
  LanguageModelState            saved per layer through MixerState hooks
   ├─ attention    KV codes (int8) + scales (bf16); shared-KV layers by reference
   ├─ DeltaNet     recurrent + conv state
   ├─ ShortConv    conv state
   └─ tokens       prefix tokens, last output token
       │
       ▼
  stream(rest of prompt + new message) ─▶ first token


  file.uzkv   magic │ version │ JSON header │ prefix tokens │ state buffers
  layout      payload 16 KiB-aligned, each buffer 256 B-aligned
  header      key parts, layer layout, sizes, xxh3 of the payload
  key         format │ model blake3 │ device (GPU, cores, MXU) │ kernel source hash │ numerics mode │ exact tokens
```

- **Why the whole state:** Qwen 3.5 is hybrid (18 Gated DeltaNet + 6 attention layers); KV alone can't reproduce it.
  New mixer types opt in through the hooks without a format change.
- **Miss, never a wrong restore:** any key mismatch, bad size, layout or checksum is a miss and a normal prefill. All
  10 negative cases (corrupt payload, truncated file, bad header, other model, device, kernel build or numerics mode,
  wrong tokens, context too small, missing file) miss with the right reason on both models.
- **Hit rules:** a snapshot covers an exact token prefix, aligned down to uzu's 1,024-token prefill batch
  (`SnapshotAlignment::PrefillBatch`), where it reproduces the uncached run exactly. Prompts under 1,024 tokens aren't
  cached; a diverging prompt misses (recurrent state can't be trimmed, so longer snapshots are never partly reused).
  Restores go into a fresh state, same process or new.
- **Saving:** a miss saves a fresh snapshot, atomically replacing a stale or damaged file; nothing is saved when the
  device or kernel identity is unknown. Files are owner-only (dir 0700, file 0600); two saves of one prompt are
  byte-identical.

### Exactness

Both models, frozen probe set, cold restores: restored state hashes identical to a fresh prefill; first-answer logits
identical on 100/100 probes; greedy, 1,024-token and seeded continuations identical.

## Results

Time to first token, uncached vs restored, ~25-token question after the prefix; median of 15 interleaved runs, snapshot
evicted from the OS page cache before every restore. Brackets: bootstrap 95% interval of the speedup. Large ratios
mostly reflect a long uncached prefill; compare the restored ms.

| Prefix tokens | Qwen 3.5 0.8B uncached | Restored | Speedup | LFM2.5 230M uncached | Restored | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 415 ms | 64 ms | 6.5x [5.3, 7.4] | 129 ms | 22 ms | 6.0x [5.8, 6.2] |
| 2,048 | 821 ms | 57 ms | 14.3x [13.3, 14.9] | 250 ms | 20 ms | 12.6x [12.4, 12.7] |
| 4,096 | 1,801 ms | 68 ms | 26.6x [23.0, 27.9] | 520 ms | 41 ms | 12.7x [12.4, 13.5] |
| 8,192 | 3,665 ms | 82 ms | 45.0x [43.5, 48.9] | 1,155 ms | 52 ms | 22.3x [21.3, 23.4] |
| 1,499 (unaligned: restores 1,024, prefills 475) | 592 ms | 220 ms | 2.7x [2.6, 2.7] | 183 ms | 73 ms | 2.5x [2.5, 2.7] |

### Where restore time goes

Qwen 3.5 0.8B, median of 9 cold restores (snapshot rewritten through `F_NOCACHE` so it is not in the OS page cache):

| Prefix | Snapshot | Save (once) | Disk read | Read speed | Checksum | GPU copy | Restore total | Question prefill + first token |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 25.3 MiB | 59 ms | 4.2 ms | 6.0 GB/s | 1.0 ms | 3.0 ms | 9.7 ms | 23-27 ms |
| 2,048 | 31.4 MiB | 56 ms | 5.5 ms | 5.7 GB/s | 1.3 ms | 3.2 ms | 11.4 ms | 29-42 ms |
| 8,192 | 67.7 MiB | 198 ms | 10.8 ms | 6.3 GB/s | 2.8 ms | 5.9 ms | 21.1 ms | 49-56 ms |

The read runs at SSD speed; most of what's left is prefilling the new question, which costs the same with or without a
cache. About 19 MiB of each snapshot is fixed-size recurrent state; attention KV adds ~6 MiB per 1,024 tokens.

### Same laptop: MLX and llama.cpp prompt caching

Both also restore Qwen 3.5's hybrid state from disk. Same session, same prompts, cold restores, 15 runs; each engine
on its own 4-bit build, so context, not a ranking. MLX and llama.cpp matched their own uncached output on the first 16
generated tokens at every size.

The uzu numbers here come from that same session, on a build before the final key and permission changes; the
per-size table above is the current build. Uncached times vary up to ~30% between sessions on this laptop, so compare
restored times within one table.

| Prefix tokens | uzu | MLX (mlx-lm 0.32.2) | llama.cpp (0.9.0, Q4_K_M) |
|---:|---:|---:|---:|
| 1,024 | 561 -> 49 ms | 754 -> 75 ms | 685 -> 102 ms |
| 2,048 | 1,065 -> 49 ms | 1,468 -> 85 ms | 1,462 -> 113 ms |
| 4,096 | 1,919 -> 52 ms | 3,047 -> 98 ms | 2,908 -> 116 ms |
| 8,192 | 5,025 -> 89 ms | 6,261 -> 120 ms | 7,201 -> 146 ms |
| Snapshot at 8,192 | 68 MiB | 120 MB | 162 MB |

## Chat API

```rust
let config = ChatConfig::default().with_kv_cache_directory(Some(cache_dir));
let system = ChatMessage::system().with_text(long_instructions).with_kv_cache_breakpoint();
```

- The breakpoint is a metadata flag on `ChatMessage`; chat template and serialized format are untouched.
- On reply, the messages up to the breakpoint are rendered on their own; if their tokens are an exact prefix of the
  full prompt, that prefix is restored from disk instead of prefilled. Otherwise nothing changes.
- The last marked message wins; the cache is used only when the session starts from an empty state.

Qwen 3.5 0.8B, 4,400-token system prompt, new `ChatSession` each time: 2.2-2.5 s to first token without the
breakpoint, 0.19 s on a hit, identical output across all three sessions; the first session pays a one-off save.

## Opt-in experiment: batch-invariant prefill

Restores are exact only at 1,024-token boundaries because kernels pick different plans for different batch sizes. An
opt-in mode pins those plans (matmul, attention, chunked DeltaNet), so any 64-token boundary restores exactly: the
1,499-token prompt goes from 2.7x to 8.6x. Off by default, in its own commit.

It reorders some reductions, so outputs differ at the bit level from the default kernels. Both kernel sets were
compared with uzu's CPU backend as a reference (fixed reduction order, f32 accumulation) on the first answer token of
48 frozen probes (100-250 tokens):

| Qwen 3.5 0.8B, 48 probes | Default kernels | Batch-invariant |
|---|---:|---:|
| Mean KL from the CPU reference | 5.09e-3 | 3.74e-3 |
| Top-1 token = reference | 48/48 | 48/48 |
| Top-1 flips vs default kernels | | 0 |
| Bit-identical logits vs default kernels | | 5/48 |
| Noise floor: default kernels, same prompt split at another boundary (mean KL vs unsplit) | 5.74e-3 | |

Paired difference in KL from the reference (batch-invariant minus default): mean -1.35e-3, 95% bootstrap interval
[-4.04e-3, +1.08e-3]. Not significant either way, and smaller than the default kernels' own drift from a different
split. The pre-set non-inferiority margin, 20% of the default kernels' KL (1.02e-3), is missed narrowly by the 1.08e-3
upper bound; more probes would tighten it. Over 3 documents (1,536 teacher-forced tokens) perplexity is 2.8328 default
vs 2.8343 batch-invariant (+0.05%), 99.4% top-1 agreement.

## Data and how it was captured

Every number comes from the bench example on this branch, run through a fixed evaluation procedure. Raw records are in
[`prefix-cache-results/`](prefix-cache-results):

| File | Holds |
|---|---|
| `qwen3.5-0.8b_cache_1024_cold.jsonl` | Qwen cache verdict and timing: state hashes, 100 logit comparisons, 28 token runs, perplexity, negative cases, 15 timed runs per size |
| `lfm2.5-230m_cache_1024_cold.jsonl` | Same for LFM2.5 230M |
| `qwen3.5-0.8b_batch_invariant_64_cold.jsonl` | Same with batch-invariant kernels, 64-token snapshots |
| `qwen3.5-0.8b_batch_invariant_vs_default.jsonl` | Batch-invariant vs default kernels: bits, tokens, perplexity |
| `qwen3.5-0.8b_batch_invariant_vs_cpu_reference.jsonl` | Per-probe KL and top-1 against the CPU reference (table above) |
| `restore_breakdown_bench.log` | Per-run stage timings (read, checksum, GPU copy, question prefill) |
| `mlx_llamacpp_baselines.jsonl` | Every MLX and llama.cpp timing, file sizes, token checks |

- **Inputs.** Probes come from a frozen set of 519 source files (local Cargo registry) with a recorded SHA-256; every
  record file names that hash and the engine commit, so a run on different inputs or code is detectable.
- **Exactness.** Uncached and restored run in the same process: blake3 over every state buffer, logits compared bit
  for bit, continuations token for token. Repeated uncached runs first confirm the engine is deterministic.
- **Safety.** Each negative case damages or mismatches a real snapshot and checks the miss reason.
- **Timing.** Uncached and restored runs alternate (15 each per size, after a warm-up); before each timed restore the
  snapshot is rewritten through `F_NOCACHE` so it is read from the SSD. Power and thermal state are logged per run; speedup intervals are bootstrap 95%.
- **Verdicts.** A small checker applies fixed rules (every exactness check must pass for "lossless"; reference margins
  are set before the run), has a 23-case known-answer self-test, and logs every run.

## Branch layout

1. `engine: save and restore the prefilled language model state` (core, unit tests, end-to-end test)
2. `metal: opt-in batch-invariant prefill` (experiment, off by default)
3. `chat: KV cache breakpoint`
4. `examples: prefix cache bench`
5. `docs: prefix cache write-up and raw records` (this file; fork only, not for upstream)

## Reproduce

```bash
cargo +nightly build --release -p uzu --example prefix_cache_bench
B=target/release/examples/prefix_cache_bench
COLD_RESTORE=1 $B <model dir> <cache dir> bench 1024,2048,8192 9 2   # timing + stage breakdown
$B <model dir> <cache dir> negative 2048                              # all miss cases
$B <model dir> <cache dir> statehash 2048                             # restored vs fresh state hash
UZU_TEST_MODEL_PATH=<model dir> cargo +nightly test --release -p uzu --test test_prefix_cache -- --ignored
UZU_TEST_MODEL_PATH=<model dir> cargo +nightly test --release -p uzu --test test_chat_prefix_cache -- --ignored
```

## Limits

- One machine (M1 Pro, no MXU); MXU paths on newer chips untested.
- Two small models (0.8B, 230M); greedy and seeded sampling only.
- Chat API built, server tier not yet; Python, Swift and TypeScript bindings not regenerated (the new `ChatConfig` field adds a
  constructor parameter there).
- No eviction policy; blocking file IO.

## Open questions for maintainers

1. **Engine first?** Want a PR with just the engine part (commit 1)? The chat part is built too and can follow once
   that looks good.
2. **Breakpoint shape.** `with_kv_cache_breakpoint()` is a metadata flag on `ChatMessage`. Or would you rather a
   `ChatContentBlock`? Either is a small change.
3. **Cache settings.** Where should the cache folder and clean-up of old files be set: `ChatConfig`, `EngineConfig`,
   or only the server?
