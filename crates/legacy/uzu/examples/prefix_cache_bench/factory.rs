//! Eval-factory producer: `factory` records, `dumplogits` for the CPU reference, shared arm runner.

use super::*;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Arm {
    Uncached {
        invariant: bool,
    },
    Restored {
        invariant: bool,
        granule: usize,
    },
}

pub(crate) fn logits_to_f32(
    bytes: &[u8],
    vocab: usize,
) -> Vec<f32> {
    let element = bytes.len() / vocab.max(1);
    let row = &bytes[bytes.len() - vocab * element..];
    match element {
        4 => row.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        2 => row.as_chunks::<2>().0.iter().map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16)).collect(),
        _ => panic!("unexpected logits element size {element}"),
    }
}

pub(crate) fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|x| (*x as f64 - max).exp()).sum();
    let log_sum = sum.ln() + max;
    logits.iter().map(|x| *x as f64 - log_sum).collect()
}

/// Prefill `prefix + rest` the way `arm` says, capture the logits of the first sampled position,
/// then generate `count` tokens. Returns (logits bytes, tokens, time to first token, state).
pub(crate) async fn run_arm<B: Backend>(
    model: &LanguageModel<B>,
    cache_dir: &Path,
    arm: Arm,
    prefix: &[u64],
    suffix: &[u64],
    sampling: Option<SamplingMethod>,
    count: usize,
    max_context: u32,
) -> Result<(Vec<u8>, Vec<u64>, Duration, LanguageModelState<B>), String> {
    let started = Instant::now();
    let (mut state, rest) = match arm {
        Arm::Uncached {
            invariant,
        } => {
            model.set_batch_invariant(invariant);
            let state = model.create_empty_state(Some(max_context), SEED).map_err(|e| e.to_string())?;
            (state, [prefix, suffix].concat())
        },
        Arm::Restored {
            invariant,
            granule,
        } => {
            model.set_batch_invariant(invariant);
            let mut cache = PrefixCache::new(cache_dir);
            cache.alignment = SnapshotAlignment::Multiple(granule);
            // Snapshot on an aligned boundary inside the prefix; the suffix is never empty, so the
            // restored stream still prefills at least one token and yields the first answer logits.
            let snapshot = cache.alignment.snapshot_length(prefix.len());
            if snapshot == 0 {
                return Err("prefix shorter than one granule".to_string());
            }
            let path = cache.path_for(model, &prefix[..snapshot]).ok_or("no model fingerprint")?;
            let state = match model
                .restore_state_snapshot(&path, &prefix[..snapshot], Some(max_context), SEED, &RestoreOptions::default())
                .await
                .map_err(|e| e.to_string())?
            {
                Ok((state, _)) => state,
                Err(_) => {
                    // Not cached yet: build it (untimed by the caller's checked run), then restore.
                    let (built, _) = cache
                        .load_or_prefill(model, &prefix[..snapshot], Some(max_context), SEED)
                        .await
                        .map_err(|e| e.to_string())?;
                    drop(built);
                    model
                        .restore_state_snapshot(
                            &path,
                            &prefix[..snapshot],
                            Some(max_context),
                            SEED,
                            &RestoreOptions::default(),
                        )
                        .await
                        .map_err(|e| e.to_string())?
                        .map_err(|miss| format!("miss after save: {miss:?}"))?
                        .0
                },
            };
            let rest = [&prefix[snapshot..], suffix].concat();
            // Batch-invariant mode: a 1-token stream takes the single-row decode path, which is not
            // bit-identical to batched prefill. Callers must stream >= 2 tokens after a restore.
            if rest.len() < 2 {
                return Err(format!("restored stream input is {} token(s); need >= 2", rest.len()));
            }
            (state, rest)
        },
    };
    let mut options = model.default_stream_options();
    if let Some(method) = sampling {
        options.sampling_method = method;
    }
    options.capture_logits = true;
    let mut stream = model.stream(&rest, &mut state, options).map_err(|e| e.to_string())?;
    let mut tokens = Vec::new();
    let mut ttft = Duration::ZERO;
    while tokens.len() < count {
        match stream.generate().await.map_err(|e| e.to_string())? {
            Some(token) => {
                if tokens.is_empty() {
                    ttft = started.elapsed();
                }
                tokens.push(token);
            },
            None => break,
        }
    }
    // close() finishes the prefill (and any in-flight decode) before the captured logits are read.
    stream.close().await.map_err(|e| e.to_string())?;
    let logits = stream.captured_logits().await.map_err(|e| e.to_string())?.ok_or("no logits captured")?;
    drop(stream);
    Ok((logits, tokens, ttft, state))
}

/// One teacher-forced step: feed `token`, return the logits for the next position.
pub(crate) async fn force_step<B: Backend>(
    model: &LanguageModel<B>,
    state: &mut LanguageModelState<B>,
    token: u64,
) -> Result<Vec<f32>, String> {
    let mut options = model.default_stream_options();
    options.sampling_method = SamplingMethod::Greedy;
    options.capture_logits = true;
    // No generate(): close() completes the one-token prefill without accepting a sampled token,
    // so the state holds exactly the teacher-forced tokens.
    let mut stream = model.stream(&[token], state, options).map_err(|e| e.to_string())?;
    stream.close().await.map_err(|e| e.to_string())?;
    let logits = stream.captured_logits().await.map_err(|e| e.to_string())?.ok_or("no logits captured")?;
    Ok(logits_to_f32(&logits, model.vocab_size()))
}

/// Drop the snapshot files from the OS file cache without root: rewrite each through a descriptor with
/// F_NOCACHE (so the new pages are not kept in the unified buffer cache) and replace the original.
pub(crate) fn evict_snapshots(cache_dir: &Path) -> Result<(), String> {
    use std::{io::Write, os::fd::AsRawFd};
    unsafe extern "C" {
        fn fcntl(
            fd: i32,
            cmd: i32,
            ...
        ) -> i32;
    }
    const F_NOCACHE: i32 = 48;
    for entry in std::fs::read_dir(cache_dir).map_err(|e| e.to_string())?.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|extension| extension != "uzkv") {
            continue;
        }
        let bytes = {
            let file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
            unsafe { fcntl(file.as_raw_fd(), F_NOCACHE, 1) };
            let mut reader = std::io::BufReader::new(file);
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut reader, &mut bytes).map_err(|e| e.to_string())?;
            bytes
        };
        let temp = path.with_extension("cold");
        {
            let mut file = std::fs::File::create(&temp).map_err(|e| e.to_string())?;
            unsafe { fcntl(file.as_raw_fd(), F_NOCACHE, 1) };
            file.write_all(&bytes).map_err(|e| e.to_string())?;
            file.sync_all().map_err(|e| e.to_string())?;
        }
        std::fs::rename(&temp, &path).map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub(crate) fn remove_cache_files(cache_dir: &Path) {
    // Our own snapshot files in the bench cache dir only.
    if let Ok(entries) = std::fs::read_dir(cache_dir) {
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|extension| extension == "uzkv") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

pub(crate) fn cand_invariant(arm: Arm) -> bool {
    match arm {
        Arm::Uncached {
            invariant,
        }
        | Arm::Restored {
            invariant,
            ..
        } => invariant,
    }
}

pub(crate) fn command_first_line(
    program: &str,
    args: &[&str],
    summary_lines: usize,
) -> String {
    match std::process::Command::new(program).args(args).output() {
        Ok(output) if output.status.success() => {
            let text = String::from_utf8_lossy(&output.stdout);
            let joined = text.lines().filter(|line| !line.trim().is_empty()).take(summary_lines).collect::<Vec<_>>();
            if joined.is_empty() {
                "unknown".to_string()
            } else {
                joined.join(" | ")
            }
        },
        _ => "unknown".to_string(),
    }
}

/// State identity for one probe: digest of a fresh prefill of the snapshot span vs the digest of the
/// state restored from that snapshot.
pub(crate) async fn state_pair<B: Backend>(
    model: &LanguageModel<B>,
    cache_dir: &Path,
    invariant: bool,
    granule: usize,
    prefix: &[u64],
) -> Result<Option<(String, String)>, String> {
    model.set_batch_invariant(invariant);
    let mut cache = PrefixCache::new(cache_dir);
    cache.alignment = SnapshotAlignment::Multiple(granule);
    let snapshot = cache.alignment.snapshot_length(prefix.len());
    if snapshot == 0 {
        return Ok(None);
    }
    let tokens = &prefix[..snapshot];
    let path = cache.path_for(model, tokens).ok_or("no model fingerprint")?;
    let mut fresh = model.create_empty_state(Some(65536), SEED).map_err(|e| e.to_string())?;
    model.prefill(&mut fresh, tokens).await.map_err(|e| e.to_string())?;
    model.save_state_snapshot(&mut fresh, &path).await.map_err(|e| e.to_string())?;
    let (mut restored, _) = model
        .restore_state_snapshot(&path, tokens, Some(65536), SEED, &cache.restore_options)
        .await
        .map_err(|e| e.to_string())?
        .map_err(|miss| format!("restore missed: {miss:?}"))?;
    let base = model.state_digest(&mut fresh).await.map_err(|e| e.to_string())?;
    let cand = model.state_digest(&mut restored).await.map_err(|e| e.to_string())?;
    Ok(Some((base, cand)))
}

/// Kernel eval factory records for one comparison (see the usage at the top of this file).
pub(crate) async fn factory<B: Backend>(
    model: &LanguageModel<B>,
    cache_dir: &Path,
    args: &[String],
) -> Result<(), String> {
    use std::io::Write;
    let out_path = PathBuf::from(&args[0]);
    let compare = args[1].as_str();
    let granule: usize = args[2].parse().unwrap();
    let (base, cand) = match compare {
        "cache" => {
            let invariant = granule < 1024;
            (
                Arm::Uncached {
                    invariant,
                },
                Arm::Restored {
                    invariant,
                    granule,
                },
            )
        },
        "invariant" => (
            Arm::Uncached {
                invariant: false,
            },
            Arm::Uncached {
                invariant: true,
            },
        ),
        // The red half of red/green: fine-grained snapshots on the default kernels.
        "cache-default-kernels" => (
            Arm::Uncached {
                invariant: false,
            },
            Arm::Restored {
                invariant: false,
                granule,
            },
        ),
        _ => return Err(format!("unknown comparison {compare}")),
    };
    let max_context: u32 = 65536;
    let vocab = model.vocab_size();
    let mut out = std::fs::File::create(&out_path).map_err(|e| e.to_string())?;
    let mut emit = |value: serde_json::Value| -> Result<(), String> {
        writeln!(out, "{value}").map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())
    };
    let restores_state = matches!(cand, Arm::Restored { .. });
    let profile = if compare == "invariant" {
        "numerics"
    } else {
        "state"
    };
    let config_sha = std::env::var("FACTORY_CONFIG_SHA")
        .map_err(|_| "FACTORY_CONFIG_SHA is not set (sha256 of the kernel eval factory config)".to_string())?;
    let model_id = std::env::var("MODEL_ID").unwrap_or_else(|_| "alibaba:qwen3.5:0.8b:mirai:mirai-m:4".to_string());
    let experiment = format!("uzu-prefix-cache/{compare}/{granule}");
    emit(serde_json::json!({
        "kind": "meta", "hash": "blake3",
        "experiment": experiment, "profile": profile, "restores_state": restores_state,
        "config_sha": config_sha, "models": [model_id.clone()],
        "device_paths": [format!("metal{}", if std::env::var("UZU_MXU").is_ok() { "-mxu" } else { "-non-mxu (M1 Pro)" })],
        "comparison": compare, "granule": granule, "base": format!("{base:?}"), "cand": format!("{cand:?}"),
        "engine_commit": std::env::var("ENGINE_COMMIT").unwrap_or_default(),
        "model": model_id,
        "subset_sha256": std::env::var("PROBE_CORPUS_SHA").unwrap_or_default(),
    }))?;

    // Real documents, deterministic probe lengths: odd, prime and round, 300..6000 tokens.
    let pool = document_pool();
    let mut document_tokens = Vec::new();
    for (index, (name, body)) in pool.iter().enumerate().step_by(7) {
        let tokens = encode(model, &format!("<|im_start|>system\nDocument: {name}\n\n{body}"));
        if tokens.len() >= 6600 {
            document_tokens.push(tokens);
        }
        if document_tokens.len() >= 24 || index > pool.len() {
            break;
        }
    }
    if document_tokens.len() < 4 {
        return Err("not enough long documents".to_string());
    }
    // Probes must be longer than one snapshot granule, or the cached arm would silently be the uncached one.
    let shortest = if matches!(cand, Arm::Restored { .. }) {
        (granule + 1).max(300)
    } else {
        300
    };
    let primes = [307usize, 557, 911, 1031, 1499, 2053, 2477, 3001, 3571, 4099, 4513, 5003, 5521];
    let mut lengths: Vec<usize> = primes.iter().copied().filter(|length| *length >= shortest).collect();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    while lengths.len() < 100 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        lengths.push(shortest + (seed % (6000 - shortest as u64)) as usize);
    }
    let questions = LONG_QUESTIONS;
    let probe = |index: usize| -> (Vec<u64>, Vec<u64>) {
        let document = &document_tokens[index % document_tokens.len()];
        let prefix = document[..lengths[index]].to_vec();
        let suffix = suffix_tokens(model, questions[index % questions.len()]);
        (prefix, suffix)
    };

    // Warm-up both arms.
    {
        let (prefix, suffix) = probe(0);
        let _ = run_arm(model, cache_dir, base, &prefix, &suffix, Some(SamplingMethod::Greedy), 2, max_context).await?;
        let _ = run_arm(model, cache_dir, cand, &prefix, &suffix, Some(SamplingMethod::Greedy), 2, max_context).await?;
        remove_cache_files(cache_dir);
    }

    // Steps 0-2: floor, bits, greedy tokens (first 20 probes carry 64 tokens, the rest 1).
    for index in 0..100 {
        let (prefix, suffix) = probe(index);
        let count = if index < 20 {
            64
        } else {
            1
        };
        let (base_logits, base_tokens, _, _) =
            run_arm(model, cache_dir, base, &prefix, &suffix, Some(SamplingMethod::Greedy), count, max_context).await?;
        let (cand_logits, cand_tokens, _, _) =
            run_arm(model, cache_dir, cand, &prefix, &suffix, Some(SamplingMethod::Greedy), count, max_context).await?;
        let id = format!("p{index}_len{}", prefix.len());
        emit(
            serde_json::json!({"kind": "bits", "id": id, "base_hash": content_hash(&base_logits), "cand_hash": content_hash(&cand_logits)}),
        )?;
        if restores_state
            && index < 2
            && let Some((state_base, state_cand)) =
                state_pair(model, cache_dir, cand_invariant(cand), granule, &prefix).await?
        {
            emit(serde_json::json!({"kind": "state", "id": id, "base_hash": state_base, "cand_hash": state_cand}))?;
        }
        if index < 20 {
            let (repeat_logits, _, _, _) =
                run_arm(model, cache_dir, base, &prefix, &suffix, Some(SamplingMethod::Greedy), 1, max_context).await?;
            emit(
                serde_json::json!({"kind": "floor", "id": id, "base_hash": content_hash(&base_logits), "base_repeat_hash": content_hash(&repeat_logits)}),
            )?;
            emit(
                serde_json::json!({"kind": "tokens", "mode": "greedy", "id": id, "base": base_tokens, "cand": cand_tokens}),
            )?;
        }
        remove_cache_files(cache_dir);
        eprintln!(
            "[factory] probe {index} len {} bits {}",
            prefix.len(),
            content_hash(&base_logits) == content_hash(&cand_logits)
        );
    }
    // Long decode and seeded sampling.
    for (mode, probes, count, sampling) in
        [("long", 3usize, 1024usize, Some(SamplingMethod::Greedy)), ("seeded", 5, 64, None)]
    {
        for index in 0..probes {
            let (prefix, suffix) = probe(20 + index);
            let (_, base_tokens, _, _) =
                run_arm(model, cache_dir, base, &prefix, &suffix, sampling.clone(), count, max_context).await?;
            let (_, cand_tokens, _, _) =
                run_arm(model, cache_dir, cand, &prefix, &suffix, sampling.clone(), count, max_context).await?;
            emit(
                serde_json::json!({"kind": "tokens", "mode": mode, "id": format!("{mode}{index}"), "base": base_tokens, "cand": cand_tokens}),
            )?;
            remove_cache_files(cache_dir);
        }
    }
    // Step 3: teacher-forced NLL, KL and top-1 over 512 real continuation tokens per document.
    for document_index in 0..3 {
        let document = &document_tokens[document_index * 5 % document_tokens.len()];
        let prefix_length = [2053usize, 3001, 4513][document_index];
        let (prefix, continuation) = (&document[..prefix_length], &document[prefix_length..prefix_length + 513]);
        // The first scored position comes from the prefill itself: split off the prefix's last token as the "suffix".
        let (head, last) = prefix.split_at(prefix.len() - 1);
        // count = 0: prefill only, no sampled token accepted, so both states hold exactly the prefix.
        let (base_logits, _, _, mut base_state) =
            run_arm(model, cache_dir, base, head, last, Some(SamplingMethod::Greedy), 0, max_context).await?;
        let (cand_logits, _, _, mut cand_state) =
            run_arm(model, cache_dir, cand, head, last, Some(SamplingMethod::Greedy), 0, max_context).await?;
        let (mut base_nll, mut cand_nll, mut kl_sum, mut kl_values, mut top1) = (0f64, 0f64, 0f64, Vec::new(), 0usize);
        let mut base_row = logits_to_f32(&base_logits, vocab);
        let mut cand_row = logits_to_f32(&cand_logits, vocab);
        let scored = 512;
        for position in 0..scored {
            let target = continuation[position] as usize;
            let (lb, lc) = (log_softmax(&base_row), log_softmax(&cand_row));
            base_nll -= lb[target];
            cand_nll -= lc[target];
            let kl: f64 = lb.iter().zip(&lc).map(|(b, c)| b.exp() * (b - c)).sum();
            kl_sum += kl;
            kl_values.push(kl);
            let argmax = |row: &[f64]| row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if argmax(&lb) == argmax(&lc) {
                top1 += 1;
            }
            if position + 1 < scored {
                if let Arm::Uncached {
                    invariant,
                } = base
                {
                    model.set_batch_invariant(invariant);
                }
                base_row = force_step(model, &mut base_state, continuation[position]).await?;
                match cand {
                    Arm::Uncached {
                        invariant,
                    }
                    | Arm::Restored {
                        invariant,
                        ..
                    } => model.set_batch_invariant(invariant),
                }
                cand_row = force_step(model, &mut cand_state, continuation[position]).await?;
            }
        }
        kl_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        emit(serde_json::json!({
            "kind": "perplexity", "id": format!("doc{document_index}_prefix{prefix_length}"), "scored_tokens": scored,
            "base_nll": base_nll, "cand_nll": cand_nll, "kl_mean": kl_sum / scored as f64,
            "kl_p999": kl_values[((scored as f64) * 0.999) as usize - 1], "top1_agreement": top1 as f64 / scored as f64,
        }))?;
        remove_cache_files(cache_dir);
    }
    // Safety gate: the same negative cases as the `negative` mode, under the factory's case names.
    if restores_state {
        model.set_batch_invariant(false);
        let mut cache = PrefixCache::new(cache_dir);
        cache.alignment = SnapshotAlignment::Multiple(granule);
        let length = 2048usize.max(granule * 2);
        for (case, result) in negative_cases(model, &cache, length).await? {
            let name = match case {
                "payload_bitflip" => "corrupt_payload",
                "truncated" | "format_version" | "context_too_small" => case,
                "model_fingerprint" => "model_identity",
                _ => continue,
            };
            let verdict = if result.is_err() {
                "miss"
            } else {
                "hit"
            };
            emit(serde_json::json!({"kind": "negative", "case": name, "result": verdict}))?;
        }
        remove_cache_files(cache_dir);
    }
    let power = command_first_line("pmset", &["-g", "batt"], 1);
    let thermal = command_first_line("pmset", &["-g", "therm"], 3);
    // Step 5: speed. Interleaved A/B, 15 timed runs per size after the checked run; timed outputs must match.
    for size in [1024usize, 1499, 2048, 4096, 8192] {
        let document =
            document_tokens.iter().find(|tokens| tokens.len() > size + 8).ok_or("no document long enough")?;
        let (prefix, suffix) = (document[..size].to_vec(), suffix_tokens(model, questions[0]));
        let (_, checked, _, _) =
            run_arm(model, cache_dir, base, &prefix, &suffix, Some(SamplingMethod::Greedy), 8, max_context).await?;
        let cold = std::env::var("COLD_RESTORE").is_ok();
        // Make sure the candidate's snapshot exists before timing (saving is a one-off cost, reported elsewhere).
        let _ = run_arm(model, cache_dir, cand, &prefix, &suffix, Some(SamplingMethod::Greedy), 1, max_context).await?;
        let (mut base_ms, mut cand_ms, mut timed_match) = (Vec::new(), Vec::new(), true);
        for _ in 0..15 {
            let (_, tokens, ttft, _) =
                run_arm(model, cache_dir, base, &prefix, &suffix, Some(SamplingMethod::Greedy), 8, max_context).await?;
            timed_match &= tokens == checked;
            base_ms.push(ms(ttft));
            if cold {
                evict_snapshots(cache_dir)?;
            }
            let (_, tokens, ttft, _) =
                run_arm(model, cache_dir, cand, &prefix, &suffix, Some(SamplingMethod::Greedy), 8, max_context).await?;
            timed_match &= tokens == checked;
            cand_ms.push(ms(ttft));
        }
        emit(serde_json::json!({
            "kind": "speed", "size": size, "base_ms": base_ms, "cand_ms": cand_ms,
            "cache_state": if cold { "cold (snapshot rewritten with F_NOCACHE before every timed restore)" } else { "warm (file just written)" },
            "hardware": "Apple M1 Pro, 16 GB", "timed_outputs_match": timed_match,
            "power": power, "thermal": thermal, "other_jobs": "none checked",
        }))?;
        remove_cache_files(cache_dir);
        eprintln!("[factory] speed {size}: base {base_ms:?} cand {cand_ms:?} match {timed_match}");
    }
    model.set_batch_invariant(false);
    Ok(())
}

/// Fixed probes for the reference step: real-document prefixes of PROBE_MIN..PROBE_MIN+PROBE_SPAN tokens
/// (default 150..400, short enough for the CPU backend), one question each. Same probes in every process
/// and on every backend; SHARD=k/n runs every n-th probe starting at k so CPU processes can run in parallel.
pub(crate) async fn dump_logits<B: Backend>(
    model: &LanguageModel<B>,
    cache_dir: &Path,
    args: &[String],
) -> Result<(), String> {
    let out_dir = PathBuf::from(&args[0]).join(&args[1]);
    let arm = match args[1].as_str() {
        "default" => Arm::Uncached {
            invariant: false,
        },
        "invariant" => Arm::Uncached {
            invariant: true,
        },
        "split64" => Arm::Restored {
            invariant: false,
            granule: 64,
        },
        other => return Err(format!("unknown arm {other}")),
    };
    let probes: usize = args[2].parse().unwrap();
    let env = |key: &str, default: usize| std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
    let (minimum, span) = (env("PROBE_MIN", 150), env("PROBE_SPAN", 250));
    let (shard, shards) = std::env::var("SHARD")
        .ok()
        .and_then(|v| v.split_once('/').map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap())))
        .unwrap_or((0usize, 1usize));
    let cache_dir = cache_dir.join(format!("shard{shard}"));
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let pool = document_pool();
    let vocab = model.vocab_size();
    // Build the probe list first so it is identical whatever the sharding.
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut list = Vec::new();
    for (index, (name, body)) in pool.iter().enumerate().step_by(11) {
        if list.len() == probes {
            break;
        }
        let tokens = encode(model, &format!("<|im_start|>system\nDocument: {name}\n\n{body}"));
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let length = minimum + (seed % span as u64) as usize;
        if tokens.len() >= length {
            list.push((tokens[..length].to_vec(), suffix_tokens(model, LONG_QUESTIONS[index % LONG_QUESTIONS.len()])));
        }
    }
    for (probe, (prefix, suffix)) in list.iter().enumerate().skip(shard).step_by(shards) {
        let started = Instant::now();
        let (logits, _, _, _) =
            run_arm(model, &cache_dir, arm, prefix, suffix, Some(SamplingMethod::Greedy), 0, 8192).await?;
        let row = logits_to_f32(&logits, vocab);
        let bytes: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(out_dir.join(format!("{probe:03}_len{}.f32", prefix.len())), bytes)
            .map_err(|e| e.to_string())?;
        remove_cache_files(&cache_dir);
        eprintln!(
            "[dumplogits] {} probe {probe} len {} in {:.1} s",
            args[1],
            prefix.len(),
            started.elapsed().as_secs_f64()
        );
    }
    model.set_batch_invariant(false);
    Ok(())
}
