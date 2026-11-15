use candle_core::DType;
use candle_core::Tensor;
use llm_kernels::KernelError;
use llm_kernels::Result;

pub fn single_prefill(q: &Tensor, k: &Tensor, v: &Tensor, sm_scale: Option<f32>) -> Result<Tensor> {
    let q_shape = q.dims();
    let k_shape = k.dims();
    let v_shape = v.dims();

    if q_shape.len() != 3 || k_shape.len() != 3 || v_shape.len() != 3 {
        return Err(KernelError::InvalidShape(
            "Q/K/V must be rank-3 tensors".to_string(),
        ));
    }

    let qo_len = q_shape[0];
    let num_q_heads = q_shape[1];
    let head_dim = q_shape[2];

    let kv_len = k_shape[0];
    let num_kv_heads = k_shape[1];

    if head_dim != 128 {
        return Err(KernelError::InvalidShape(
            "temperarily support head dim =128 only".to_string(),
        ));
    }

    if k_shape != v_shape {
        return Err(KernelError::InvalidShape("kv shape must match".to_string()));
    }

    if k_shape[2] != head_dim {
        return Err(KernelError::InvalidShape(
            "QKV head dim must match".to_string(),
        ));
    }

    if kv_len < qo_len {
        return Err(KernelError::InvalidShape(
            "kv_len must >= qo_len".to_string(),
        ));
    }

    if num_kv_heads == 0 || num_q_heads % num_kv_heads != 0 {
        return Err(KernelError::InvalidConfig(
            "Invalid GQA head configuration".to_string(),
        ));
    }

    let output =
        unsafe { Tensor::empty((qo_len, num_q_heads, head_dim), DType::BF16, q.device())? };
    todo!()
}
