use crate::Result;
use candle_core::Tensor;

pub trait AttentionBackend: Send + Sync {
    fn batch_decode(&self, params: &DecodeParams) -> Result<Tensor>;

    fn single_prefill(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        config: &PrefillConfig,
    ) -> Result<Tensor>;
}

pub struct DecodeParams<'a> {
    pub q: &'a Tensor,
    pub kv_data: &'a Tensor,
}

pub struct PrefillConfig {
    pub causual: bool,
    pub sm_scal: Option<f32>,
}
