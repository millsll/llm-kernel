use crate::Result;
use candle_core::Tensor;

pub trait Normalization: Send + Sync {
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;
}
