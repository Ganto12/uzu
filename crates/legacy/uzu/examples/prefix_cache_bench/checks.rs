//! Safety and exactness checks: `negative` (every miss case) and `statehash` (restored vs fresh state).

use super::*;

/// Corrupt or mismatched snapshots must be misses.
pub(crate) async fn negative<B: Backend>(
    model: &LanguageModel<B>,
    cache: &PrefixCache,
    args: &[String],
) -> Result<(), String> {
    let length: usize = args[0].parse().unwrap();
    for (name, result) in negative_cases(model, cache, length).await? {
        println!(
            "negative/{name}: {}",
            match result {
                Ok(()) => "HIT (BAD)".to_string(),
                Err(miss) => format!("miss {miss:?} (ok)"),
            }
        );
    }
    Ok(())
}

/// Runs every negative case and returns (case name, the miss, or `Ok(())` if it was wrongly restored).
pub(crate) async fn negative_cases<B: Backend>(
    model: &LanguageModel<B>,
    cache: &PrefixCache,
    length: usize,
) -> Result<Vec<(&'static str, Result<(), SnapshotMiss>)>, String> {
    let mut results = Vec::new();
    let prefix = prefix_tokens(model, cache.alignment.snapshot_length(length));
    let path = cache.path_for(model, &prefix).ok_or("no model fingerprint")?;
    let _ = cache.load_or_prefill(model, &prefix, Some(MAX_CONTEXT), SEED).await.map_err(|e| e.to_string())?;
    let options = RestoreOptions::default();
    let mut check = |name: &'static str, result: Result<(), SnapshotMiss>| results.push((name, result));
    // Different prefix tokens, same file.
    let mut other = prefix.clone();
    *other.last_mut().unwrap() ^= 1;
    let r = model
        .restore_state_snapshot(&path, &other, Some(MAX_CONTEXT), SEED, &options)
        .await
        .map_err(|e| e.to_string())?;
    check("wrong_tokens", r.map(|_| ()));
    // Payload bit flip.
    let corrupt = path.with_extension("corrupt");
    let mut bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let last = bytes.len() - 7;
    bytes[last] ^= 0x40;
    std::fs::write(&corrupt, &bytes).map_err(|e| e.to_string())?;
    let r = model
        .restore_state_snapshot(&corrupt, &prefix, Some(MAX_CONTEXT), SEED, &options)
        .await
        .map_err(|e| e.to_string())?;
    check("payload_bitflip", r.map(|_| ()));
    // Truncated file.
    std::fs::write(&corrupt, &bytes[..bytes.len() - 4096]).map_err(|e| e.to_string())?;
    let r = model
        .restore_state_snapshot(&corrupt, &prefix, Some(MAX_CONTEXT), SEED, &options)
        .await
        .map_err(|e| e.to_string())?;
    check("truncated", r.map(|_| ()));
    // Model fingerprint mismatch (edit header in place, same length).
    let mut bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let header = String::from_utf8_lossy(&bytes[..4096]).to_string();
    let needle = format!("\"model_fingerprint\":\"{}", hex_prefix(model.model_fingerprint().unwrap()));
    if let Some(position) = header.find(&needle) {
        let at = position + needle.len() - 1;
        bytes[at] = if bytes[at] == b'0' {
            b'1'
        } else {
            b'0'
        };
    }
    std::fs::write(&corrupt, &bytes).map_err(|e| e.to_string())?;
    let r = model
        .restore_state_snapshot(&corrupt, &prefix, Some(MAX_CONTEXT), SEED, &options)
        .await
        .map_err(|e| e.to_string())?;
    check("model_fingerprint", r.map(|_| ()));
    // Format version bump.
    let mut bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    bytes[8] = bytes[8].wrapping_add(1);
    std::fs::write(&corrupt, &bytes).map_err(|e| e.to_string())?;
    let r = model
        .restore_state_snapshot(&corrupt, &prefix, Some(MAX_CONTEXT), SEED, &options)
        .await
        .map_err(|e| e.to_string())?;
    check("format_version", r.map(|_| ()));
    // Context too small for the prefix.
    let r = model
        .restore_state_snapshot(&path, &prefix, Some(prefix.len() as u32), SEED, &options)
        .await
        .map_err(|e| e.to_string())?;
    check("context_too_small", r.map(|_| ()));
    // File cut inside the header (must be a miss, not an I/O error). Start from the
    // unmodified file so the version check above cannot mask this case.
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    std::fs::write(&corrupt, &bytes[..64]).map_err(|e| e.to_string())?;
    let r = model
        .restore_state_snapshot(&corrupt, &prefix, Some(MAX_CONTEXT), SEED, &options)
        .await
        .map_err(|e| e.to_string())?;
    check("truncated_header", r.map(|_| ()));
    // Device fingerprint edited in the header (same length).
    let mut bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let header = String::from_utf8_lossy(&bytes[..4096]).to_string();
    let needle = "\"device_fingerprint\":\"";
    let position = header.find(needle).ok_or("no device_fingerprint in header")?;
    bytes[position + needle.len()] ^= 1;
    std::fs::write(&corrupt, &bytes).map_err(|e| e.to_string())?;
    let r = model
        .restore_state_snapshot(&corrupt, &prefix, Some(MAX_CONTEXT), SEED, &options)
        .await
        .map_err(|e| e.to_string())?;
    check("device_fingerprint", r.map(|_| ()));
    // Kernel identity edited in the header (same length): a build with different kernels.
    let mut bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let header = String::from_utf8_lossy(&bytes[..4096]).to_string();
    let needle = "\"kernel_identity\":\"";
    let position = header.find(needle).ok_or("no kernel_identity in header")?;
    bytes[position + needle.len()] ^= 1;
    std::fs::write(&corrupt, &bytes).map_err(|e| e.to_string())?;
    let r = model
        .restore_state_snapshot(&corrupt, &prefix, Some(MAX_CONTEXT), SEED, &options)
        .await
        .map_err(|e| e.to_string())?;
    check("kernel_identity", r.map(|_| ()));
    // Different numerics mode (batch-invariant flag flipped on the model's context).
    let original = model.batch_invariant();
    model.set_batch_invariant(!original);
    let r = model
        .restore_state_snapshot(&path, &prefix, Some(MAX_CONTEXT), SEED, &options)
        .await
        .map_err(|e| e.to_string());
    model.set_batch_invariant(original);
    check("numerics_mode", r?.map(|_| ()));
    let _ = std::fs::remove_file(&corrupt);
    Ok(results)
}

/// Direct state check: prefill `prefix_len` tokens, save, restore into a fresh state, then
/// compare BLAKE3 digests of every live snapshot byte of both states.
pub(crate) async fn statehash<B: Backend>(
    model: &LanguageModel<B>,
    cache: &PrefixCache,
    args: &[String],
) -> Result<(), String> {
    let length: usize = args[0].parse().map_err(|_| "prefix_len must be a number".to_string())?;
    let prefix = prefix_tokens(model, length);
    let path = cache.path_for(model, &prefix).ok_or("no model fingerprint")?;
    let mut state = model.create_empty_state(Some(MAX_CONTEXT), SEED).map_err(|e| e.to_string())?;
    model.prefill(&mut state, &prefix).await.map_err(|e| e.to_string())?;
    model.save_state_snapshot(&mut state, &path).await.map_err(|e| e.to_string())?;
    let (mut restored, _) = model
        .restore_state_snapshot(&path, &prefix, Some(MAX_CONTEXT), SEED, &cache.restore_options)
        .await
        .map_err(|e| e.to_string())?
        .map_err(|miss| format!("restore missed: {miss:?}"))?;
    let saved_digest = model.state_digest(&mut state).await.map_err(|e| e.to_string())?;
    let restored_digest = model.state_digest(&mut restored).await.map_err(|e| e.to_string())?;
    println!("statehash/{}: prefilled {saved_digest}", prefix.len());
    println!("statehash/{}: restored  {restored_digest}", prefix.len());
    let matches = saved_digest == restored_digest;
    println!(
        "statehash/{}: {}",
        prefix.len(),
        if matches {
            "MATCH (ok)"
        } else {
            "MISMATCH (BAD)"
        }
    );
    if matches {
        Ok(())
    } else {
        Err("state digests differ".to_string())
    }
}
