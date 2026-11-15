use thiserror::Error;

pub type Result<T> = std::result::Result<T, KernelError>;

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("unsupported device: {0}")]
    UnsupportedDevice(String),

    #[error("unsupported dtype: {0}")]
    UnsuportedDType(String),

    #[error("invalid kernel configuration:{0}")]
    InvalidConfig(String),

    #[error("invalide tensor shape: {0}")]
    InvalidShape(String),

    #[error("invalid tensor layout: {0}")]
    InvalidLayout(String),

    #[error("invalid device: {0}")]
    InvalidDevice(String),

    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("kernel execution failed {0}")]
    Execution(String),

    #[error("workspace allocation failed: {0}")]
    Workspace(String),

    #[error("internal kernel error: {0}")]
    Internal(String),

    #[error("Candle error: {0}")]
    CandleErr(#[from] candle_core::Error),
}
