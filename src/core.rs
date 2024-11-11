// src/core.rs
pub mod activations;
pub mod layers;
pub mod losses;
pub mod optimizers;

// Re-export commonly used items
pub use activations::Activation;
pub use layers::{LayerTrait, Dense};
pub use losses::Loss;
pub use optimizers::{Optimizer, Optimization, apply_optimization};
