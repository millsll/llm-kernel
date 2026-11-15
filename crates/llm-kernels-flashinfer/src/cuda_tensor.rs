use candle_core::{
    DType, Device, Storage, Tensor,
    cuda::cudarc::driver::{CudaStream, DevicePtr, SyncOnDrop},
};
use half::bf16;
use std::ffi::c_void;

pub(crate) struct CudaTensorView {
    pub ptr: *const c_void,
    pub shape: [usize; 3],
    pub stride_n: u32,
    pub stride_h: u32,
}

pub(crate) fn with_cuda_tensor<R>(
    tensor: &Tensor,
    stream: &CudaStream,
    f: impl FnOnce(CudaTensorView) -> R,
) -> candle_core::Result<R> {
    if tensor.dtype() != DType::BF16 {
        candle_core::bail!("expected BF16 tensor, got {:?}", tensor.dtype());
    }

    if tensor.rank() != 3 {
        candle_core::bail!("expected rank-3 tensor, got rank{}", tensor.rank());
    }

    if !matches!(tensor.device(), Device::Cuda(_)) {
        candle_core::bail!("expected CUDA tensor");
    }

    let shape = tensor.dims();

    if shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
        candle_core::bail!("empty tensors are not supported");
    }

    let strides = tensor.stride();

    if strides[2] != 1 {
        candle_core::bail!(
            "last dimension must be contiguous, got stride {}",
            strides[2]
        );
    }

    let stride_n = u32::try_from(strides[0])
        .map_err(|_| candle_core::Error::Msg("token stride exceeds u32".to_string()))?;
    let stride_h = u32::try_from(strides[1])
        .map_err(|_| candle_core::Error::Msg("head stride exceeds u32".to_string()))?;

    let shape = [shape[0], shape[1], shape[2]];

    let (storage, layout) = tensor.storage_and_layout();
    let cuda_storage = match &*storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("tensor storage is not CUDA storage"),
    };
    let cuda_slice = cuda_storage.as_cuda_slice::<bf16>()?;
    let start = layout.start_offset();
    let max_relative_offset = shape
        .iter()
        .zip(strides.iter())
        .try_fold(0usize, |offset, (&dim, &stride)| {
            offset.checked_add((dim - 1).checked_mul(stride)?)
        })
        .ok_or_else(|| candle_core::Error::Msg("tensor storage offset overflow".to_string()))?;
    let end = start
        .checked_add(max_relative_offset)
        .and_then(|offset| offset.checked_add(1))
        .ok_or_else(|| candle_core::Error::Msg("tensor storage range overflow".to_string()))?;
    if end > cuda_slice.len() {
        candle_core::bail!(
            "tensor view exceeds CUDA storage: end={}, storage_len={}",
            end,
            cuda_slice.len()
        );
    }

    let cuda_view = cuda_slice.slice(start..end);
    let (device_ptr, guard) = cuda_view.device_ptr(stream);
    let view = CudaTensorView {
        ptr: device_ptr as usize as *const c_void,
        shape,
        stride_n,
        stride_h,
    };
    let result = f(view);
    drop(guard);
    Ok(result)
}

fn qkv_shape_check(tensor:&Tensor)->candle_core::Result<()>{

    Ok(())
}

pub(crate) fn with_qkv<R>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    stream: &CudaStream,
    f: impl FnOnce(CudaTensorView, CudaTensorView, CudaTensorView) -> R,
) -> candle_core::Result<R> {
    let q_shape_check
    todo!()
}

pub(crate) fn with_cuda_tensors<const N: usize, R>(
    tensors: [&Tensor; N],
    stream: &CudaStream,
    f: impl FnOnce([CudaTensorView; N]) -> candle_core::Result<R>,
) -> candle_core::Result<R> {
    let mut views = Vec::with_capacity(N);
    let mut guards = Vec::with_capacity(N);

    for tensor in tensors {
        let (view, guard) = make_cuda_view(tensor, stream)?;
    }
    todo!()
}
