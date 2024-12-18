// src/core.rs
pub mod activations;
pub mod layers;
pub mod losses;
pub mod optimizers;
pub mod normalization;
// Re-export commonly used items
pub use activations::Activation;
pub use layers::{LayerTrait, Dense};
pub use losses::Loss;
pub use optimizers::*;
pub use normalization::Normalization;