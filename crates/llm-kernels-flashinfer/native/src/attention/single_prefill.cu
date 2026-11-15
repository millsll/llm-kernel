#include "flashinfer_adapter.h"
#include<cstdint>

#include<cuda_runtime.h>
#include <cuda_bf16.h>

#include <driver_types.h>
#include<flashinfer/attention/mask.cuh>
#include<flashinfer/attention/prefill.cuh>
#include<flashinfer/attention/variants.cuh>
#include<flashinfer/fastdiv.cuh>
#include <flashinfer/pos_enc.cuh>

namespace flashinfer{
    struct PrefillParams{
        using DTypeQ=nv_bfloat16;
        using DTypeKV=nv_bfloat16;
        using DTypeO=nv_bfloat16;
        using IdType=int32_t;

        DTypeQ*q;
        DTypeKV*k;
        DTypeKV*v;
        DTypeO*o;

        float*lse;
        uint_fastdiv group_size;

        float sm_scale;

        uint32_t qo_len;
        uint32_t kv_len;

        uint32_t num_qo_heads;
        uint32_t num_kv_heads;

        // stride by elements
        uint32_t q_stride_n;
        uint32_t q_stride_h;
        uint32_t k_stride_n;
        uint32_t k_stride_h;
        uint32_t v_stride_n;
        uint32_t v_stride_h;

        uint32_t head_dim;
        int32_t window_left;

        bool partition_kv;

        __host__ __device__ __forceinline__
        uint32_t get_qo_len(uint32_t batch_idx)const{
            return qo_len;
        }

        __host__ __device__ __forceinline__
        uint32_t get_kv_len(uint32_t batch_idx)const{
            return kv_len;
        }
    };
};

using PrefillAttentionVariant=flashinfer::DefaultAttention<false, false, false, false>;
extern "C" int flashinfer_single_prefill_bf16(
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
){
    // 初版只支持head_dim =128
    if(q==nullptr||k==nullptr||v==nullptr||output==nullptr){
        return static_cast<int>(cudaErrorInvalidValue);
    }

    if(head_dim!=128||
        qo_len==0||
        kv_len==0||
        num_qo_heads==0||
        num_kv_heads==0||
        num_qo_heads%num_kv_heads!=0||
        kv_len<qo_len
    ){
            return static_cast<int>(cudaErrorInvalidValue);
    }
    flashinfer::PrefillParams params{};

    params.q=static_cast<nv_bfloat16*>(const_cast<void*>(q));
    params.k=static_cast<nv_bfloat16*>(const_cast<void*>(k));
    params.v=static_cast<nv_bfloat16*>(const_cast<void*>(v));
    params.o=static_cast<nv_bfloat16*>(output);

    params.lse=nullptr;
    params.group_size=flashinfer::uint_fastdiv(num_qo_heads/num_kv_heads);

    params.sm_scale=sm_scale;

    params.qo_len=qo_len;
    params.kv_len=kv_len;

    params.num_qo_heads=num_qo_heads;
    params.num_kv_heads=num_kv_heads;

    params.q_stride_n=q_stride_n;
    params.q_stride_h=q_stride_h;
    params.k_stride_n=k_stride_n;
    params.k_stride_h=k_stride_h;
    params.v_stride_n=v_stride_n;
    params.v_stride_h=v_stride_h;

    params.head_dim=head_dim;
    params.window_left=-1;
    params.partition_kv=false;

    return static_cast<int>(
        flashinfer::SinglePrefillWithKVCacheDispatched<
            128,
            128,
            flashinfer::PosEncodingMode::kNone,
            false,
            flashinfer::MaskMode::kCausal,
            false,
            PrefillAttentionVariant,
            flashinfer::PrefillParams
        >(params, nullptr, stream)
    );
}
