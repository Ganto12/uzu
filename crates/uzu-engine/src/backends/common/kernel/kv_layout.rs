use half::bf16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KVLayout {
    pub head_dim: u32,
    pub num_kv_heads: u32,
    pub key_alignment: u32,
    pub backend_opaque_layout: u32,
}

impl KVLayout {
    pub fn new(
        head_dim: u32,
        num_kv_heads: u32,
        key_alignment: u32,
        backend_opaque_layout: u32,
    ) -> Self {
        Self {
            head_dim,
            num_kv_heads,
            key_alignment,
            backend_opaque_layout,
        }
    }

    pub fn code_bytes(
        &self,
        keys: u32,
    ) -> usize {
        keys.next_multiple_of(self.key_alignment) as usize * self.num_kv_heads as usize * self.head_dim as usize
    }

    pub fn scale_bytes(
        &self,
        keys: u32,
    ) -> usize {
        keys.next_multiple_of(self.key_alignment) as usize * self.num_kv_heads as usize * size_of::<bf16>()
    }
}

// Block geometry of the Metal KV layouts (`backends/metal/gpu_types/kv_layout.rs`); a test checks
// that these stay equal to the kernel constants.
const TILE_KEYS: usize = 32;
const TILE_DIMS: usize = 16;
const FRAGMENT_BYTES: usize = TILE_KEYS * TILE_DIMS;
const SLICE_DIMS: usize = 64;
const GROUP_ROWS: usize = 8;
const LANE_COLS: usize = 4;
const LANE_BYTES: usize = TILE_KEYS / GROUP_ROWS * LANE_COLS;
const OPAQUE_ROW_MAJOR: u32 = 0;
const OPAQUE_MXU_TILED: u32 = 1;

fn mma_lane(
    row_in_group: usize,
    channel_group: usize,
) -> usize {
    ((row_in_group >> 2) << 4)
        | ((channel_group >> 1) << 3)
        | (((row_in_group >> 1) & 1) << 2)
        | ((row_in_group & 1) << 1)
        | (channel_group & 1)
}

impl KVLayout {
    /// Byte offset of code `channel` of key row `row` (0..TILE_KEYS) inside one block-head region
    /// of `TILE_KEYS * head_dim` bytes. Mirrors `k_code_index` / `v_code_index` in
    /// `backends/metal/kernel/attention/kv_layout.h`.
    fn code_offset_in_region(
        &self,
        row: usize,
        channel: usize,
        is_value: bool,
    ) -> usize {
        if self.backend_opaque_layout == OPAQUE_ROW_MAJOR {
            return row * self.head_dim as usize + channel;
        }
        if !is_value {
            let lane = mma_lane(row % GROUP_ROWS, channel % TILE_DIMS / LANE_COLS);
            let lane_offset = row / GROUP_ROWS * LANE_COLS + channel % LANE_COLS;
            return channel / TILE_DIMS * FRAGMENT_BYTES + lane * LANE_BYTES + lane_offset;
        }
        let slice_channel = channel % SLICE_DIMS;
        let tile = slice_channel / TILE_DIMS;
        let tiles = SLICE_DIMS / TILE_DIMS;
        let half_rows = TILE_KEYS / 2;
        let lane = mma_lane(row % GROUP_ROWS, slice_channel % TILE_DIMS / LANE_COLS);
        let pair = tiles * (channel / SLICE_DIMS) + tiles / 2 * (row / half_rows) + tile / 2;
        let lane_offset = tile % 2 * (LANE_BYTES / 2) + row % half_rows / GROUP_ROWS * LANE_COLS + channel % LANE_COLS;
        pair * FRAGMENT_BYTES + lane * LANE_BYTES + lane_offset
    }

    /// True if this layout's byte geometry is known, so unwritten tail bytes can be located.
    fn geometry_known(&self) -> bool {
        match self.key_alignment as usize {
            1 => true,
            TILE_KEYS => {
                self.head_dim > 0
                    && match self.backend_opaque_layout {
                        OPAQUE_ROW_MAJOR => true,
                        OPAQUE_MXU_TILED => (self.head_dim as usize).is_multiple_of(SLICE_DIMS),
                        _ => false,
                    }
            },
            _ => false,
        }
    }

    /// Zeroes every byte of `codes` (the first `code_bytes(keys)` bytes of a K or V code buffer)
    /// that no key below `keys` occupies: the unwritten slots of the last, partly filled key
    /// block. Returns `false` (touching nothing) if this layout's geometry is not known, in which
    /// case the caller cannot prove those bytes are initialised.
    pub fn zero_unwritten_codes(
        &self,
        codes: &mut [u8],
        keys: u32,
        is_value: bool,
    ) -> bool {
        if !self.geometry_known() || codes.len() < self.code_bytes(keys) {
            return false;
        }
        let rows = keys as usize % self.key_alignment as usize;
        if self.key_alignment == 1 || rows == 0 {
            return true;
        }
        let head_dim = self.head_dim as usize;
        let region = TILE_KEYS * head_dim;
        let mut written = vec![false; region];
        for row in 0..rows {
            for channel in 0..head_dim {
                written[self.code_offset_in_region(row, channel, is_value)] = true;
            }
        }
        let last_block = keys as usize / TILE_KEYS;
        for head in 0..self.num_kv_heads as usize {
            let base = (last_block * self.num_kv_heads as usize + head) * region;
            for (offset, _) in written.iter().enumerate().filter(|(_, written)| !**written) {
                codes[base + offset] = 0;
            }
        }
        true
    }

    /// Same as [`Self::zero_unwritten_codes`] for a bf16 scale buffer (layout `[block][head][key]`).
    pub fn zero_unwritten_scales(
        &self,
        scales: &mut [u8],
        keys: u32,
    ) -> bool {
        if !self.geometry_known() || scales.len() < self.scale_bytes(keys) {
            return false;
        }
        let rows = keys as usize % self.key_alignment as usize;
        if self.key_alignment == 1 || rows == 0 {
            return true;
        }
        let last_block = keys as usize / TILE_KEYS;
        for head in 0..self.num_kv_heads as usize {
            let base = (last_block * self.num_kv_heads as usize + head) * TILE_KEYS;
            let start = (base + rows) * size_of::<bf16>();
            let end = (base + TILE_KEYS) * size_of::<bf16>();
            scales[start..end].fill(0);
        }
        true
    }
}

#[cfg(test)]
#[path = "../../../../unit/backends/common/kernel/kv_layout_test.rs"]
mod tests;
