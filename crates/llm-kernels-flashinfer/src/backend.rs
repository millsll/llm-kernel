use candle_core::Tensor;
use llm_kernels::Result;
use llm_kernels::backend::AttentionBackend;
use llm_kernels::backend::DecodeParams;

pub struct FlashInferBackend {}

impl AttentionBackend for FlashInferBackend {
    fn batch_decode(&self, params: &DecodeParams) -> Result<Tensor> {
        todo!()
    }
}
