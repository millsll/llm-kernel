#include "flashinfer_adapter.h"
#include <cuda_bf16.h>
#include<flashinfer/norm.cuh>

extern "C" int flashinfer_rmsnorm_bf16(
    const void *input,
    const void* weight,
    void* output,
    uint32_t batch_size,
    uint32_t hidden_size,
    uint32_t stride_input,
    uint32_t stride_output,
    float eps,
    cudaStream_t stream
){
    return static_cast<int>(
        flashinfer::norm::RMSNorm(
            const_cast<nv_bfloat16*>(
                static_cast<const nv_bfloat16*>(input)
            ),
            const_cast<nv_bfloat16*>(
                static_cast<const nv_bfloat16*>(weight)
            ),
            static_cast<nv_bfloat16*>(output),
            batch_size,
            hidden_size,
            stride_input,
            stride_output,
            eps,
            false,
            stream)
    );
}
