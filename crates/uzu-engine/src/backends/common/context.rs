use std::{path::Path, sync::Arc};

use bytemuck::AnyBitPattern;

use crate::backends::common::{Backend, BufferRef, CommandBuffer, DeviceCapabilities};

pub trait Context: Sized + Send + Sync {
    type Backend: Backend<Context = Self>;

    fn new() -> impl Future<Output = Result<Arc<Self>, <Self::Backend as Backend>::Error>> + Send;

    fn device_name(&self) -> Option<&str>;

    /// Identity of the device for numerics purposes: anything that can change a kernel's result
    /// (name, core count, matrix-extension support). Used to key persisted state, so two devices
    /// that could compute differently must not share a fingerprint. `"unknown"` means the backend
    /// cannot tell, and callers must then refuse to persist or restore.
    fn device_fingerprint(&self) -> String {
        self.device_name().unwrap_or("unknown").to_string()
    }

    /// Batch-invariant numerics. When enabled, kernels pick the same plan (tiling, split count,
    /// kernel variant) for every multi-token batch, so each token's result is bit-identical no
    /// matter how its prompt was split into batches. Needed for exact prefix-cache restores at
    /// any breakpoint. Backends without batch-dependent plans can ignore it.
    fn set_batch_invariant(
        &self,
        _enabled: bool,
    ) {
    }

    fn batch_invariant(&self) -> bool {
        false
    }

    fn create_command_buffer(
        &self,
        name: Option<&str>,
        allocation_pool: Option<Arc<<Self::Backend as Backend>::AllocationPool>>,
    ) -> Result<<<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding, <Self::Backend as Backend>::Error>;

    fn create_buffer_gpu(
        &self,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::GlobalGpuBuffer, <Self::Backend as Backend>::Error>;

    fn create_buffer_cpu_gpu(
        &self,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::GlobalBufferCpuGpu, <Self::Backend as Backend>::Error>;

    fn create_sparse_buffer(
        &self,
        capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error>;

    fn copyout<T: AnyBitPattern>(
        &self,
        buffer: impl BufferRef<Backend = Self::Backend, Buffer = <Self::Backend as Backend>::GlobalGpuBuffer>,
    ) -> impl Future<Output = Result<Vec<T>, <Self::Backend as Backend>::Error>> + Send;

    fn create_allocation_pool(&self) -> Arc<<Self::Backend as Backend>::AllocationPool>;

    fn peak_memory_usage(&self) -> Option<usize>;

    fn start_capture(
        &self,
        output_path: &Path,
    ) -> Result<<Self::Backend as Backend>::CaptureSpan, <Self::Backend as Backend>::Error>;

    fn device_capabilities(&self) -> DeviceCapabilities;
}
