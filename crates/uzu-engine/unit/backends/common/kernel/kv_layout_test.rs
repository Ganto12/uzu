use uzu_engine_macros::uzu_test;

use super::*;

fn layout(
    head_dim: u32,
    opaque: u32,
) -> KVLayout {
    KVLayout::new(head_dim, 3, TILE_KEYS as u32, opaque)
}

#[uzu_test]
fn full_block_codes_are_a_bijection() {
    for head_dim in [64u32, 128, 256] {
        for opaque in [OPAQUE_ROW_MAJOR, OPAQUE_MXU_TILED] {
            for is_value in [false, true] {
                let layout = layout(head_dim, opaque);
                let mut seen = vec![false; TILE_KEYS * head_dim as usize];
                for row in 0..TILE_KEYS {
                    for channel in 0..head_dim as usize {
                        let offset = layout.code_offset_in_region(row, channel, is_value);
                        assert!(!seen[offset], "hd {head_dim} layout {opaque} v {is_value}: offset {offset} reused");
                        seen[offset] = true;
                    }
                }
                assert!(seen.iter().all(|&byte| byte));
            }
        }
    }
}

#[uzu_test]
fn zero_unwritten_codes_only_clears_tail_slots() {
    for opaque in [OPAQUE_ROW_MAJOR, OPAQUE_MXU_TILED] {
        for is_value in [false, true] {
            let layout = layout(128, opaque);
            let keys = 37u32;
            let mut codes = vec![0xFFu8; layout.code_bytes(keys)];
            assert!(layout.zero_unwritten_codes(&mut codes, keys, is_value));
            let region = TILE_KEYS * 128;
            let mut live = vec![false; codes.len()];
            for key in 0..keys as usize {
                for head in 0..3 {
                    for channel in 0..128 {
                        let base = (key / TILE_KEYS * 3 + head) * region;
                        live[base + layout.code_offset_in_region(key % TILE_KEYS, channel, is_value)] = true;
                    }
                }
            }
            for (index, (&byte, &live)) in codes.iter().zip(&live).enumerate() {
                assert_eq!(byte == 0xFF, live, "byte {index} layout {opaque} value {is_value}");
            }
        }
    }
}

#[uzu_test]
fn zero_unwritten_scales_only_clears_tail_slots() {
    let layout = layout(128, OPAQUE_MXU_TILED);
    let keys = 37u32;
    let mut scales = vec![0xFFu8; layout.scale_bytes(keys)];
    assert!(layout.zero_unwritten_scales(&mut scales, keys));
    for head in 0..3usize {
        for slot in 0..TILE_KEYS {
            let index = (3 + head) * TILE_KEYS + slot;
            let expected = if slot < 5 {
                0xFF
            } else {
                0
            };
            assert_eq!(&scales[index * 2..index * 2 + 2], &[expected; 2]);
        }
    }
    assert!(scales[..3 * TILE_KEYS * 2].iter().all(|&byte| byte == 0xFF));
}

#[uzu_test]
fn unknown_geometry_is_not_claimed() {
    let layout = KVLayout::new(128, 3, 16, 0);
    assert!(!layout.zero_unwritten_codes(&mut vec![0; layout.code_bytes(5)], 5, false));
    let cpu = KVLayout::new(128, 3, 1, 0);
    assert!(cpu.zero_unwritten_codes(&mut vec![1; cpu.code_bytes(5)], 5, false));
}

#[cfg(backend = "metal")]
#[uzu_test]
fn constants_match_metal_kernel() {
    use crate::backends::metal::gpu_types::kv_layout as metal;
    assert_eq!(TILE_KEYS as u32, metal::TILE_KEYS);
    assert_eq!(TILE_DIMS as u32, metal::TILE_DIMS);
    assert_eq!(FRAGMENT_BYTES as u32, metal::FRAGMENT_BYTES);
    assert_eq!(SLICE_DIMS as u32, metal::SLICE_DIMS);
    assert_eq!(GROUP_ROWS as u32, metal::GROUP_ROWS);
    assert_eq!(LANE_COLS as u32, metal::LANE_COLS);
    assert_eq!(LANE_BYTES as u32, metal::LANE_BYTES);
    assert_eq!(OPAQUE_ROW_MAJOR, metal::KVLayout::RowMajor as u32);
    assert_eq!(OPAQUE_MXU_TILED, metal::KVLayout::MxuTiled as u32);
}
