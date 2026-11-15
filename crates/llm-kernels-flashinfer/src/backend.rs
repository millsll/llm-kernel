use crate::ops::normalization;
use crate::ops::normalization::rms_norm_wrap_bf16;
use candle_core::Tensor;
use llm_kernels::Result;
use llm_kernels::backend::AttentionBackend;
use llm_kernels::backend::DecodeParams;
use llm_kernels::normalization::Normalization;

pub struct FlashInferBackend {}

impl AttentionBackend for FlashInferBackend {
    fn batch_decode(&self, params: &DecodeParams) -> Result<Tensor> {
        todo!()
    }
    fn single_prefill(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        config: &llm_kernels::backend::PrefillConfig,
    ) -> Result<Tensor> {
        todo!()
    }
}

impl Normalization for FlashInferBackend {
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        rms_norm_wrap_bf16(input, weight, eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;
    #[test]
    fn rms_norm_running() {
        let x = vec![bf16::from_f32(3.0); 32];
        let g = vec![bf16::from_f32(1.0); 32];
        let device = candle_core::Device::new_cuda(0).unwrap();
        let xt = Tensor::from_vec(x, (1, 32), &device).unwrap();
        let gt = Tensor::from_vec(g, (32,), &device).unwrap();
        println!("input tensor dtype: x{:?},g{:?}", xt.dtype(), gt.dtype());
        let res = rms_norm_wrap_bf16(&xt, &gt, 0.0).unwrap();
        println!("res:{:?}", res);
        let resvec = res.to_vec2::<bf16>();
        println!("{:?}", resvec);
    }
}
