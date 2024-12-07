pub use serde::{Serialize, Deserialize};
pub use std::fs::File;
pub use std::io::{Read, Write};
pub use std::f64::consts::E as e;

pub use ndarray::*;
pub use ndarray_rand::RandomExt;
pub use ndarray_rand::rand_distr::Uniform;

pub use crate::models::Sequential;
pub use crate::error::*;

// Internal re-exports
pub use crate::core::{
    Activation,
    Dense,
    LayerTrait,
    Loss,
    apply_optimization,
    Regularizer,
    OptimizerType,
    OptimizerConfig,
    Normalization,
};