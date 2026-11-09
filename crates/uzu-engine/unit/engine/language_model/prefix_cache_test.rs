use uzu_engine_macros::uzu_test;

use super::*;
use crate::encodable_block::mixer::{
    MIXER_SNAPSHOT_KIND_ATTENTION, MIXER_SNAPSHOT_KIND_DELTA_NET, MixerSnapshotLayout,
};

#[uzu_test]
fn alignment_rounds_down_to_prefill_batch() {
    let aligned = SnapshotAlignment::PrefillBatch;
    assert_eq!(aligned.snapshot_length(0), 0);
    assert_eq!(aligned.snapshot_length(512), 0);
    assert_eq!(aligned.snapshot_length(1024), 1024);
    assert_eq!(aligned.snapshot_length(1500), 1024);
    assert_eq!(aligned.snapshot_length(57_344 + 17), 57_344);
    assert_eq!(SnapshotAlignment::Exact.snapshot_length(1500), 1500);
}

#[uzu_test]
fn payload_layout_aligns_buffers_and_skips_shared_layers() {
    let layers = vec![
        TransformerLayerSnapshot::Owned(MixerSnapshotLayout {
            kind: MIXER_SNAPSHOT_KIND_DELTA_NET,
            meta: vec![0],
            buffer_bytes: vec![10, 300],
        }),
        TransformerLayerSnapshot::Shared(0),
        TransformerLayerSnapshot::Owned(MixerSnapshotLayout {
            kind: MIXER_SNAPSHOT_KIND_ATTENTION,
            meta: vec![0; 8],
            buffer_bytes: vec![0, 1],
        }),
    ];
    let (entries, total) = payload_layout(&layers).unwrap();
    assert_eq!(entries, vec![(0, 10), (256, 300), (768, 0), (768, 1)]);
    assert_eq!(total, 769);
    assert!(entries.iter().all(|(offset, _)| offset % BUFFER_ALIGNMENT == 0));
}

#[uzu_test]
fn header_round_trips_through_json() {
    let header = SnapshotHeader {
        format_version: SNAPSHOT_FORMAT_VERSION,
        engine_version: "0.0.0".to_string(),
        backend: "metal".to_string(),
        device_fingerprint: "Apple M5 gpu_cores=10 mxu=true".to_string(),
        kernel_identity: "abc123".to_string(),
        batch_invariant: true,
        model_fingerprint: hex(&[7; 32]),
        prefix_token_count: 2048,
        prefix_tokens_blake3: hex(blake3::hash(&tokens_bytes(&[1, 2, 3]).unwrap()).as_bytes()),
        context_length: 2048,
        last_output_token: Some(42),
        layers: vec![TransformerLayerSnapshot::Shared(3)],
        payload_offset: 16 * 1024,
        payload_bytes: 0,
        payload_xxh3: format!("{:032x}", xxhash_rust::xxh3::xxh3_128(&[])),
    };
    let json = serde_json::to_vec(&header).unwrap();
    let parsed = serde_json::from_slice::<SnapshotHeader>(&json).unwrap();
    assert_eq!(parsed, header);
    assert_eq!(parsed.device_fingerprint, "Apple M5 gpu_cores=10 mxu=true");
    assert_eq!(parsed.kernel_identity, "abc123");
    assert!(parsed.batch_invariant);
    assert_eq!(parsed.format_version, 3);
}

#[uzu_test]
fn payload_layout_rejects_overflowing_sizes() {
    let owned = |buffer_bytes: Vec<u64>| {
        TransformerLayerSnapshot::Owned(MixerSnapshotLayout {
            kind: MIXER_SNAPSHOT_KIND_DELTA_NET,
            meta: vec![0],
            buffer_bytes,
        })
    };
    // A single buffer that ends exactly at u64::MAX does not overflow; one more byte does.
    assert!(payload_layout(&[owned(vec![u64::MAX])]).is_some());
    assert!(payload_layout(&[owned(vec![u64::MAX, 1])]).is_none());
    // Sum of two buffers overflows.
    assert!(payload_layout(&[owned(vec![u64::MAX / 2 + 1, u64::MAX / 2 + 1])]).is_none());
    // Aligning the next offset overflows.
    assert!(payload_layout(&[owned(vec![u64::MAX - 10, 0])]).is_none());
    // Overflow spread over several layers.
    assert!(payload_layout(&[owned(vec![u64::MAX - 300]), owned(vec![1000])]).is_none());
    // Normal sizes still work.
    assert_eq!(payload_layout(&[owned(vec![5, 5])]).unwrap().1, 261);
}

#[uzu_test]
fn tokens_above_u32_are_rejected() {
    assert!(tokens_bytes(&[u32::MAX as u64]).is_some());
    assert!(tokens_bytes(&[1, u32::MAX as u64 + 1]).is_none());
}

#[uzu_test]
fn tokens_are_little_endian_u32() {
    assert_eq!(tokens_bytes(&[1, 0x0102_0304]).unwrap(), vec![1, 0, 0, 0, 4, 3, 2, 1]);
}

#[uzu_test]
fn header_length_is_capped() {
    assert_eq!(header_len_to_u32(0), Some(0));
    assert_eq!(header_len_to_u32(MAX_HEADER_BYTES as usize), Some(MAX_HEADER_BYTES as u32));
    assert_eq!(header_len_to_u32(MAX_HEADER_BYTES as usize + 1), None);
    assert_eq!(header_len_to_u32(u32::MAX as usize + 1), None);

    assert_eq!(check_header_length(100, 200), Ok(()));
    assert_eq!(check_header_length(185, 200), Err(SnapshotMiss::Truncated));
    assert_eq!(check_header_length(MAX_HEADER_BYTES, u64::MAX), Ok(()));
    assert_eq!(check_header_length(MAX_HEADER_BYTES + 1, u64::MAX), Err(SnapshotMiss::BadHeader));
    assert_eq!(check_header_length(u32::MAX as u64, 1 << 40), Err(SnapshotMiss::BadHeader));
}

#[uzu_test]
fn payload_gaps_are_zeroed() {
    let mut payload = vec![0xAAu8; 1100];
    // Buffers at [0, 10), [256, 556), a zero-length one at 768, then [768, 769).
    let entries = vec![(0, 10), (256, 300), (768, 0), (768, 1)];
    zero_payload_gaps(&mut payload[..769], &entries);
    assert!(payload[..10].iter().all(|&byte| byte == 0xAA));
    assert!(payload[10..256].iter().all(|&byte| byte == 0));
    assert!(payload[256..556].iter().all(|&byte| byte == 0xAA));
    assert!(payload[556..768].iter().all(|&byte| byte == 0));
    assert_eq!(payload[768], 0xAA);
    // Bytes past the slice handed in are untouched.
    assert!(payload[769..].iter().all(|&byte| byte == 0xAA));

    // Trailing bytes after the last buffer are cleared too, and no entries clears everything.
    let mut tail = vec![0xAAu8; 8];
    zero_payload_gaps(&mut tail, &[(0, 4)]);
    assert_eq!(tail, vec![0xAA, 0xAA, 0xAA, 0xAA, 0, 0, 0, 0]);
    let mut none = vec![0xAAu8; 4];
    zero_payload_gaps(&mut none, &[]);
    assert_eq!(none, vec![0; 4]);
}

#[uzu_test]
fn restore_ranges_are_validated_up_front() {
    assert_eq!(validate_restore_ranges(&[(0, 10), (256, 0)], &[10, 5]), Ok(()));
    assert!(validate_restore_ranges(&[(0, 10)], &[10, 10]).unwrap_err().contains("count mismatch"));
    assert!(validate_restore_ranges(&[(0, 11)], &[10]).unwrap_err().contains("larger than target"));
    assert!(validate_restore_ranges(&[(0, u64::MAX)], &[usize::MAX]).is_ok() || usize::BITS < 64);
    assert!(validate_restore_ranges(&[(0, 1)], &[0]).is_err());
}

#[uzu_test]
fn split_entries_gives_disjoint_slices_at_the_entry_offsets() {
    let mut payload = vec![0u8; 1024];
    let entries = [(0u64, 10u64), (256, 4), (512, 0), (768, 8)];
    {
        let mut slices = split_entries_mut(&mut payload, &entries);
        assert_eq!(slices.iter().map(|slice| slice.len()).collect::<Vec<_>>(), vec![10, 4, 0, 8]);
        slices.iter_mut().for_each(|slice| slice.fill(0xEE));
    }
    for (index, &byte) in payload.iter().enumerate() {
        let live = entries.iter().any(|&(offset, bytes)| (offset..offset + bytes).contains(&(index as u64)));
        assert_eq!(byte == 0xEE, live, "byte {index}");
    }
}

#[uzu_test]
fn temp_file_names_are_recognised_exactly() {
    let stem = "0123456789abcdef0123456789abcdef";
    assert!(is_temp_snapshot_name(&format!("{stem}.tmp.123.4")));
    assert!(!is_temp_snapshot_name(&format!("{stem}.uzkv")));
    assert!(!is_temp_snapshot_name(&format!("{stem}.tmp.123")));
    assert!(!is_temp_snapshot_name(&format!("{stem}.tmp.x.4")));
    assert!(!is_temp_snapshot_name("notes.tmp.1.2"));
    assert!(!is_temp_snapshot_name(&format!("{}.tmp.1.2", &stem[..31])));
}

#[cfg(unix)]
#[uzu_test]
fn private_directory_and_file_modes() {
    use std::os::unix::fs::PermissionsExt;
    let root = std::env::temp_dir().join(format!("uzu_prefix_cache_modes_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let directory = root.join("cache");
    create_private_directory(&directory).unwrap();
    assert_eq!(fs::metadata(&directory).unwrap().permissions().mode() & 0o777, 0o700);
    // An existing directory is never loosened.
    create_private_directory(&directory).unwrap();
    assert_eq!(fs::metadata(&directory).unwrap().permissions().mode() & 0o777, 0o700);
    let file = directory.join("a.bin");
    drop(create_private_file(&file).unwrap());
    assert_eq!(fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o600);
    assert!(create_private_file(&file).is_err());

    // Stale temp files go, fresh ones and other files stay.
    let stem = "0123456789abcdef0123456789abcdef";
    let temp = directory.join(format!("{stem}.tmp.1.1"));
    let other = directory.join(format!("{stem}.uzkv"));
    fs::write(&temp, b"x").unwrap();
    fs::write(&other, b"x").unwrap();
    assert_eq!(remove_stale_temp_files(&directory, Duration::from_secs(3600)), 0);
    assert!(temp.exists());
    assert_eq!(remove_stale_temp_files(&directory, Duration::ZERO), 1);
    assert!(!temp.exists() && other.exists());
    let _ = fs::remove_dir_all(&root);
}
