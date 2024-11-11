extern crate plotters;

pub mod core;
pub mod error;
pub mod models;
pub mod prelude;
pub mod utils;


// Re-export types
pub use core::{Activation, Dense, LayerTrait, Loss, Optimizer};
//pub use models::Sequential;

// Re-export macros without using utils::*
//pub use crate::{rand_array, Model, Dense as dense};

pub mod benchmark {
    pub mod functions;
}

pub mod plot {
    pub mod plot_comparision;
}