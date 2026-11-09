//! `long`: exactness at long contexts on real documents.

use super::*;

/// Real documents for the long-context runs and the logit probes, as (name, text). With
/// `PROBE_CORPUS` set, reads that frozen JSONL file of `{"name", "text"}` lines, so every run sees
/// the same documents. Otherwise scans the Markdown files over 4 KiB in the local cargo registry,
/// which changes whenever crates are unpacked; use that only to create a corpus with `freezecorpus`.
pub(crate) fn document_pool() -> Vec<(String, String)> {
    if let Ok(path) = std::env::var("PROBE_CORPUS") {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("PROBE_CORPUS {path}: {error}"));
        return text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let value: serde_json::Value = serde_json::from_str(line).expect("corpus line");
                (value["name"].as_str().unwrap().to_string(), value["text"].as_str().unwrap().to_string())
            })
            .collect();
    }
    let root = PathBuf::from(std::env::var("HOME").unwrap()).join(".cargo/registry/src");
    let mut files = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(directory) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|extension| extension == "md")
                && entry.metadata().is_ok_and(|metadata| metadata.len() > 4096)
            {
                files.push(path);
            }
        }
    }
    files.sort();
    files
        .into_iter()
        .filter_map(|path| {
            let text = std::fs::read_to_string(&path).ok()?;
            let name = path.strip_prefix(&root).unwrap_or(&path).to_string_lossy().to_string();
            Some((name, text))
        })
        .collect()
}

pub(crate) const LONG_QUESTIONS: &[&str] = &[
    "Which crate is documented first in the material above, and what does it do? Answer in two sentences.",
    "Name three different crates from the material above and say what each one is for.",
    "What is the last document in the material above about? Answer briefly.",
    "Summarize the main themes of the material above in three bullet points.",
];

/// Long-context gate: for each prompt, a different set of real documents of exactly
/// `length` tokens. Compares a fresh uncached prefill against a disk restore of the same prefix.
pub(crate) async fn long<B: Backend>(
    model: &LanguageModel<B>,
    cache: &PrefixCache,
    args: &[String],
) -> Result<(), String> {
    let length: usize = args[0].parse().unwrap();
    let prompts: usize = args[1].parse().unwrap();
    let gen_tokens: usize = args[2].parse().unwrap();
    let max_context: u32 = args[3].parse().unwrap();
    if cache.alignment.snapshot_length(length) != length {
        return Err(format!("long mode needs a prefill-batch-aligned length, got {length}"));
    }
    let pool = document_pool();
    eprintln!("[long] document pool: {} files", pool.len());

    // Warm-up: pipelines / kernels compile on first use.
    {
        let warm = prefix_tokens(model, 256);
        let mut state = model.create_empty_state(Some(max_context), SEED).unwrap();
        let _ = generate(model, &mut state, &[warm, suffix_tokens(model, QUESTIONS[0])].concat(), 4).await;
    }

    println!(
        "| prefix tokens | prompt | first document | cold TTFT (ms) | save (ms) | restored TTFT (ms) | restore (ms) | read (ms) | file (MiB) | speedup | exact tokens |"
    );
    println!("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|");
    for prompt in 0..prompts {
        // Each prompt starts at a different point of the pool, so the documents differ.
        let start = prompt * pool.len() / prompts.max(1);
        let mut text = String::from(
            "<|im_start|>system\nYou are a helpful assistant. Answer questions about the documents below.\n\n",
        );
        let first = pool[start].0.clone();
        let mut tokens = Vec::new();
        for (name, body) in pool.iter().cycle().skip(start).take(pool.len()) {
            text.push_str(&format!("## Document: {name}\n\n{body}\n\n"));
            if text.len() > length * 6 {
                tokens = encode(model, &text);
                if tokens.len() >= length {
                    break;
                }
            }
        }
        if tokens.len() < length {
            tokens = encode(model, &text);
        }
        if tokens.len() < length {
            return Err(format!("document pool too short: {} < {length}", tokens.len()));
        }
        tokens.truncate(length);
        let prefix = tokens;
        let suffix = suffix_tokens(model, LONG_QUESTIONS[prompt % LONG_QUESTIONS.len()]);
        let full = [prefix.clone(), suffix.clone()].concat();

        // A. fresh uncached prefill of prefix + suffix (today's path).
        let started = Instant::now();
        let mut state = model.create_empty_state(Some(max_context), SEED).map_err(|e| e.to_string())?;
        let allocated = started.elapsed();
        let (ttft, fresh_tokens) = generate(model, &mut state, &full, gen_tokens).await;
        let cold_ms = ms(allocated + ttft);
        drop(state);

        // B. populate the cache (miss: prefill the prefix, save).
        let path = cache.path_for(model, &prefix).ok_or("no model fingerprint")?;
        let _ = std::fs::remove_file(&path);
        let (state, outcome) =
            cache.load_or_prefill(model, &prefix, Some(max_context), SEED).await.map_err(|e| e.to_string())?;
        drop(state);
        let save = match outcome {
            PrefixCacheOutcome::Miss {
                save: Some(save),
                ..
            } => save,
            other => return Err(format!("expected a miss with a save, got {other:?}")),
        };

        // C. restore from disk, stream the suffix.
        let started = Instant::now();
        let (mut state, timings) = match model
            .restore_state_snapshot(&path, &prefix, Some(max_context), SEED, &RestoreOptions::default())
            .await
            .map_err(|e| e.to_string())?
        {
            Ok(hit) => hit,
            Err(miss) => return Err(format!("unexpected miss {miss:?}")),
        };
        let restored_in = started.elapsed();
        let (ttft, restored_tokens) = generate(model, &mut state, &suffix, gen_tokens).await;
        drop(state);
        let restored_ms = ms(restored_in + ttft);
        // Our own snapshot file in the bench cache dir; free the space before the next prompt.
        let _ = std::fs::remove_file(&path);

        let exact = if restored_tokens == fresh_tokens {
            format!("{0}/{0} identical", fresh_tokens.len())
        } else {
            let first_diff = restored_tokens.iter().zip(&fresh_tokens).position(|(a, b)| a != b);
            format!("DIFFERS at {first_diff:?} of {}", fresh_tokens.len())
        };
        let decode = |tokens: &[u64]| {
            model.tokenizer().decode(&tokens.iter().map(|t| *t as u32).collect::<Vec<_>>(), false).unwrap_or_default()
        };
        eprintln!("[long] {length} prompt {prompt} first={first} answer: {:?}", decode(&restored_tokens));
        if restored_tokens != fresh_tokens {
            eprintln!("[long]   fresh   : {:?}", decode(&fresh_tokens));
        }
        println!(
            "| {length} | {prompt} | {first} | {cold_ms:.0} | {:.0} | {restored_ms:.1} | {:.1} | {:.1} | {:.1} | {:.1}x | {exact} |",
            ms(save.duration),
            ms(timings.total),
            ms(timings.read),
            save.file_bytes as f64 / (1024.0 * 1024.0),
            cold_ms / restored_ms,
        );
    }
    Ok(())
}
