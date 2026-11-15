use std::ffi::{self, c_void};

use crate::sys::flashinfer_rmsnorm_bf16;
use candle_core::backend::BackendStorage;
use candle_core::cuda::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle_core::op::BackpropOp;
use candle_core::{CudaStorage, DType, Device, Storage, Tensor};
use llm_kernels::KernelError::{self, Cuda};
use llm_kernels::Result;

pub fn rms_norm_wrap_bf16(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    // check dtype
    if input.dtype() != DType::BF16 {
        return Err(KernelError::UnsuportedDType(format!(
            "rms norm input: {}",
            input.dtype().as_str()
        )));
    }
    if weight.dtype() != DType::BF16 {
        return Err(KernelError::UnsuportedDType(format!(
            "rms norm weight: {}",
            weight.dtype().as_str()
        )));
    }
    // check device
    if !matches!(input.device(), Device::Cuda(_)) {
        return Err(KernelError::UnsupportedDevice(format!(
            "{:?}",
            input.device()
        )));
    }
    if !matches!(weight.device(), Device::Cuda(_)) {
        return Err(KernelError::UnsupportedDevice(format!(
            "{:?}",
            weight.device()
        )));
    }
    // check shapes
    if input.rank() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "rms norm input shape: {:?}",
            input.shape()
        )));
    }
    if weight.rank() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "rms norm weight shape: {:?}",
            input.shape()
        )));
    }
    // input shape match weight shape
    if input.dim(1)? != weight.dim(0)? {
        return Err(KernelError::InvalidShape(format!(
            "rms norm shape mismatch: {:?}, {:?}",
            input.shape(),
            weight.shape(),
        )));
    }
    // device check
    if !input.device().same_device(weight.device()) {
        return Err(KernelError::InvalidDevice(format!(
            "device not match:{:?}, {:?}",
            input.device(),
            weight.device()
        )));
    }

    // 预备数据
    let (input_storage, input_layout) = input.storage_and_layout();
    let (weight_storage, weight_layout) = weight.storage_and_layout();

    let Storage::Cuda(input_storage) = &*input_storage else {
        return Err(KernelError::InvalidDevice(
            "expected CUDA storage".to_string(),
        ));
    };
    let Storage::Cuda(weight_storage) = &*weight_storage else {
        return Err(KernelError::InvalidDevice(
            "expected CUDA storage".to_string(),
        ));
    };

    let input_slice = input_storage.as_cuda_slice::<half::bf16>()?;
    let weight_slice = weight_storage.as_cuda_slice::<half::bf16>()?;

    let (input_statr, input_end) = input_layout
        .contiguous_offsets()
        .ok_or_else(|| KernelError::InvalidLayout("rms norm input must be contiguous".into()))?;
    let (weight_statr, weight_end) = weight_layout
        .contiguous_offsets()
        .ok_or_else(|| KernelError::InvalidLayout("rms norm weight must be contiguous".into()))?;
    let input_view = input_slice.slice(input_statr..input_end);
    let weight_view = weight_slice.slice(weight_statr..weight_end);
    let stream = input_slice.stream();
    let (input_ptr, _input_guard) = input_view.device_ptr(stream);
    let (weight_ptr, _weight_guard) = weight_view.device_ptr(stream);

    // 分配输出tensor
    let output_shape = input.shape().clone();
    let output_numel = input.elem_count();
    let mut out_slice = unsafe { input_storage.device().alloc::<half::bf16>(output_numel)? };
    let (out_ptr, _out_guard) = out_slice.device_ptr_mut(stream);
    let status = unsafe {
        flashinfer_rmsnorm_bf16(
            input_ptr as *const c_void,
            weight_ptr as *const c_void,
            out_ptr as *mut c_void,
            input.dim(0)? as u32,
            input.dim(1)? as u32,
            input.layout().stride()[0] as u32,
            input.dim(1)? as u32,
            eps,
            stream.cu_stream() as *mut c_void,
        )
    };
    if status != 0 {
        return Err(KernelError::Execution(format!("{}", status)));
    }
    drop(_out_guard);
    let out_storage = CudaStorage::wrap_cuda_slice(out_slice, input_storage.device().clone());
    let output = Tensor::from_storage(
        Storage::Cuda(out_storage),
        output_shape,
        BackpropOp::none(),
        false,
    );
    Ok(output)
}
