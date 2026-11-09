//! Timing modes: `bench` (cold vs restored time to first token, restore stage breakdown) and the
//! cross-process `save` / `restore` / `nocache` check.

use super::*;

pub(crate) async fn bench<B: Backend>(
    model: &LanguageModel<B>,
    cache: &PrefixCache,
    args: &[String],
) -> Result<(), String> {
    let lengths: Vec<usize> = args[0].split(',').map(|value| value.parse().unwrap()).collect();
    let runs: usize = args[1].parse().unwrap();
    let gen_tokens: usize = args[2].parse().unwrap();

    // Warm-up: pipelines / kernels compile on first use.
    {
        let warm = prefix_tokens(model, 256);
        let mut state = model.create_empty_state(Some(MAX_CONTEXT), SEED).unwrap();
        let _ = generate(model, &mut state, &[warm, suffix_tokens(model, QUESTIONS[0])].concat(), 4).await;
    }

    println!(
        "| prefix tokens | cold TTFT (ms) | cold 2-stage TTFT (ms) | restored TTFT (ms) | restored TTFT, no checksum (ms) | restore total (ms) | header | allocate | read | verify | blit | save (ms) | file (MiB) | speedup | exact vs 2-stage | exact vs monolithic |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for length in lengths {
        let prefix = prefix_tokens(model, length);
        // The snapshot covers `snapshot` tokens; the rest of the prefix is streamed with the suffix.
        let snapshot = cache.alignment.snapshot_length(length);
        if snapshot == 0 {
            eprintln!("[bench] prefix {length}: shorter than one prefill batch, nothing to cache");
            continue;
        }
        let (cached, tail) = prefix.split_at(snapshot);
        let path = cache.path_for(model, cached).ok_or("no model fingerprint")?;
        let _ = std::fs::remove_file(&path); // only our own cache file, inside the cache dir

        // Populate the cache (miss path) once.
        let (_, outcome) =
            cache.load_or_prefill(model, &prefix, Some(MAX_CONTEXT), SEED).await.map_err(|e| e.to_string())?;
        let save = match outcome {
            PrefixCacheOutcome::Miss {
                save: Some(save),
                reason: SnapshotMiss::NotFound,
                ..
            } => save,
            other => return Err(format!("expected a NotFound miss with a save, got {other:?}")),
        };

        let mut m: std::collections::BTreeMap<&str, Vec<f64>> = Default::default();
        let (mut exact2, mut exact_mono, mut total) = (0, 0, 0);
        for run in 0..runs {
            let question = QUESTIONS[run % QUESTIONS.len()];
            let suffix = suffix_tokens(model, question);
            let full = [prefix.clone(), suffix.clone()].concat();

            // A. cold, monolithic: one stream over prefix + suffix (what uzu does today).
            let started = Instant::now();
            let mut state = model.create_empty_state(Some(MAX_CONTEXT), SEED).unwrap();
            let alloc = started.elapsed();
            let stream_started = Instant::now();
            let (ttft, mono_tokens) = generate(model, &mut state, &full, gen_tokens).await;
            let decode = stream_started.elapsed() - ttft;
            m.entry("cold").or_default().push(ms(alloc + ttft));
            m.entry("decode_tps").or_default().push((mono_tokens.len() - 1) as f64 / decode.as_secs_f64());
            drop(state);

            // B. cold, two-stage: prefill the snapshot part, then stream the rest (same batch boundaries as a restore).
            let rest = [tail, &suffix[..]].concat();
            let started = Instant::now();
            let mut state = model.create_empty_state(Some(MAX_CONTEXT), SEED).unwrap();
            model.prefill(&mut state, cached).await.map_err(|e| e.to_string())?;
            let prefilled = started.elapsed();
            let (ttft, two_stage_tokens) = generate(model, &mut state, &rest, gen_tokens).await;
            m.entry("cold2").or_default().push(ms(prefilled + ttft));
            m.entry("prefix_prefill").or_default().push(ms(prefilled));
            drop(state);

            // C. restored from disk (checksum verified), then stream suffix.
            for verify in [true, false] {
                if std::env::var("COLD_RESTORE").is_ok() {
                    evict_snapshots(&cache.directory)?;
                }
                let started = Instant::now();
                let (mut state, timings) = match model
                    .restore_state_snapshot(
                        &path,
                        cached,
                        Some(MAX_CONTEXT),
                        SEED,
                        &RestoreOptions {
                            verify_checksum: verify,
                        },
                    )
                    .await
                    .map_err(|e| e.to_string())?
                {
                    Ok(hit) => hit,
                    Err(miss) => return Err(format!("unexpected miss {miss:?}")),
                };
                let restored_in = started.elapsed();
                let (ttft, restored_tokens) = generate(model, &mut state, &rest, gen_tokens).await;
                drop(state);
                let key = if verify {
                    "restored"
                } else {
                    "restored_noverify"
                };
                m.entry(key).or_default().push(ms(restored_in + ttft));
                if verify {
                    m.entry("restore_wall").or_default().push(ms(restored_in));
                    m.entry("suffix_ttft").or_default().push(ms(ttft));
                    m.entry("restore").or_default().push(ms(timings.total));
                    m.entry("header").or_default().push(ms(timings.header));
                    m.entry("allocate").or_default().push(ms(timings.allocate));
                    m.entry("read").or_default().push(ms(timings.read));
                    m.entry("verify").or_default().push(ms(timings.verify));
                    m.entry("blit").or_default().push(ms(timings.blit));
                } else {
                    m.entry("restore_noverify").or_default().push(ms(timings.total));
                }
                let decode = |tokens: &[u64]| {
                    model
                        .tokenizer()
                        .decode(&tokens.iter().map(|t| *t as u32).collect::<Vec<_>>(), false)
                        .unwrap_or_default()
                };
                total += 1;
                if restored_tokens == two_stage_tokens {
                    exact2 += 1;
                } else {
                    eprintln!("[bench] MISMATCH vs two-stage at prefix {length} run {run} verify={verify}");
                    eprintln!("  two-stage: {:?}", decode(&two_stage_tokens));
                    eprintln!("  restored : {:?}", decode(&restored_tokens));
                }
                if restored_tokens == mono_tokens {
                    exact_mono += 1;
                } else {
                    let first_diff = restored_tokens.iter().zip(&mono_tokens).position(|(a, b)| a != b);
                    eprintln!(
                        "[bench] differs from monolithic at prefix {length} run {run}: first diff at token {first_diff:?}"
                    );
                    eprintln!("  monolithic: {:?}", decode(&mono_tokens));
                    eprintln!("  restored  : {:?}", decode(&restored_tokens));
                }
                if run < QUESTIONS.len() && verify {
                    eprintln!("[bench] prefix {length} q={question:?} -> {:?}", decode(&restored_tokens));
                }
            }
        }
        let mut med = |key: &str| median(m.get_mut(key).unwrap());
        let (cold_m, cold2_m, restored_m, restored_nv) =
            (med("cold"), med("cold2"), med("restored"), med("restored_noverify"));
        println!(
            "| {length} | {cold_m:.1} | {cold2_m:.1} | {restored_m:.1} | {restored_nv:.1} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.0} | {:.2} | {:.1}x | {exact2}/{total} | {exact_mono}/{total} |",
            med("restore"),
            med("header"),
            med("allocate"),
            med("read"),
            med("verify"),
            med("blit"),
            ms(save.duration),
            save.file_bytes as f64 / (1024.0 * 1024.0),
            cold_m / restored_m,
        );
        eprintln!(
            "[bench] prefix {length}: decode {:.1} tok/s, prefix-only prefill {:.1} ms ({:.0} tok/s), restore no-verify {:.2} ms",
            med("decode_tps"),
            med("prefix_prefill"),
            length as f64 / (med("prefix_prefill") / 1000.0),
            med("restore_noverify")
        );
        for (key, values) in &m {
            eprintln!(
                "[bench] raw {length} {key}: {:?}",
                values.iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>()
            );
        }
    }
    Ok(())
}

pub(crate) async fn cross_process<B: Backend>(
    model: &LanguageModel<B>,
    cache: &PrefixCache,
    mode: &str,
    args: &[String],
) -> Result<(), String> {
    let length: usize = args[0].parse().unwrap();
    let out = PathBuf::from(&args[1]);
    let gen_tokens: usize = args[2].parse().unwrap();
    let prefix = prefix_tokens(model, length);
    let mut lines = Vec::new();
    let snapshot = cache.alignment.snapshot_length(length);
    if mode == "save"
        && let Some(path) = cache.path_for(model, &prefix[..snapshot])
    {
        let _ = std::fs::remove_file(path);
    }
    for question in QUESTIONS {
        let suffix = suffix_tokens(model, question);
        let started = Instant::now();
        if mode == "nocache" {
            // Uncached reference: one fresh state, one stream over prefix + suffix.
            let mut state = model.create_empty_state(Some(MAX_CONTEXT), SEED).unwrap();
            let (ttft, tokens) = generate(model, &mut state, &[prefix.clone(), suffix].concat(), gen_tokens).await;
            eprintln!("[nocache] prefix={length} ttft {:.0} ms q={question:?}", ms(ttft));
            lines.push(tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "));
            continue;
        }
        let (mut state, outcome) =
            cache.load_or_prefill(model, &prefix, Some(MAX_CONTEXT), SEED).await.map_err(|e| e.to_string())?;
        let restore_ms = match &outcome {
            PrefixCacheOutcome::Hit(timings) => ms(timings.total),
            _ => 0.0,
        };
        let suffix_started = Instant::now();
        let rest = [&prefix[state.tokens().len()..], &suffix[..]].concat();
        let (ttft, tokens) = generate(model, &mut state, &rest, gen_tokens).await;
        eprintln!(
            "[{mode}] restore {restore_ms:.1} ms, suffix ttft {:.1} ms, prefix-to-first-token {:.1} ms",
            ms(ttft),
            ms(suffix_started - started + ttft)
        );
        let kind = match &outcome {
            PrefixCacheOutcome::Hit(_) => "hit".to_string(),
            PrefixCacheOutcome::Miss {
                reason,
                ..
            } => format!("miss({reason:?})"),
            PrefixCacheOutcome::TooShort => "too-short".to_string(),
        };
        eprintln!("[{mode}] {kind} prefix={length} total {:.0} ms q={question:?}", ms(started.elapsed()));
        if mode == "restore" && !matches!(outcome, PrefixCacheOutcome::Hit(_)) {
            return Err(format!("expected hit in restore process, got {kind}"));
        }
        lines.push(tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "));
    }
    std::fs::write(&out, lines.join("\n") + "\n").map_err(|e| e.to_string())?;
    Ok(())
}
