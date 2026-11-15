#pragma once

#include <cstdint>
#include<stdint.h>
#include<cuda_runtime_api.h>

#ifdef __cplusplus
extern "C"{
#endif

int flashinfer_rmsnorm_bf16(
    const void *input,
    const void* weight,
    void* output,
    uint32_t batch_size,
    uint32_t hidden_size,
    uint32_t stride_input,
    uint32_t stride_output,
    float eps,
    cudaStream_t stream
);

int flashinfer_single_prefill_bf16(
    const void*q,
    const void*k,
    const void*v,
    void* output,
    uint32_t qo_len,
    uint32_t kv_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t q_stride_n,
    uint32_t q_stride_h,
    uint32_t k_stride_n,
    uint32_t k_stride_h,
    uint32_t v_stride_n,
    uint32_t v_stride_h,
    float sm_scale,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
