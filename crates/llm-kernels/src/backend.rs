use crate::Result;
use candle_core::Tensor;

pub trait AttentionBackend: Send + Sync {
    fn batch_decode(&self, params: &DecodeParams) -> Result<Tensor>;
}

pub struct DecodeParams<'a> {
    pub q: &'a Tensor,
    pub kv_data: &'a Tensor,
}
