use std::ffi::c_void;

unsafe extern "C" {
    pub fn flashinfer_rmsnorm_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        batch_size: u32,
        hidden_size: u32,
        stride_input: u32,
        stride_output: u32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;

    pub fn flashinfer_single_prefill_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        qo_len: u32,
        kv_len: u32,
        num_qo_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        q_stride_n: u32,
        q_stride_h: u32,
        k_stride_n: u32,
        k_stride_h: u32,
        v_stride_n: u32,
        v_stride_h: u32,
        sm_scale: f32,
        stream: *mut c_void,
    ) -> i32;
}
