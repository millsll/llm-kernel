pub mod backend;
pub mod device;
pub mod error;
pub mod ops;

pub use error::{KernelError, Result};
pub use ops::normalization;
