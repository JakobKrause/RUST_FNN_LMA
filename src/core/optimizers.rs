use crate::prelude::*;
use ndarray::Array2;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Optimizer {
    SGD(f64),
    Adam{
        lr: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64
    },
    None
}

pub trait Optimization {
    fn optimize(&mut self, dw: Array2<f64>, db: Array2<f64>, optimizer: Optimizer);
}

pub fn apply_optimization(weights: &mut Array2<f64>, bias: &mut Array2<f64>, dw: Array2<f64>, db: Array2<f64>, optimizer: Optimizer) {
    use Optimizer::*;
    match optimizer {
        SGD(lr) => {
            *weights = weights.clone() - lr * dw;
            *bias = bias.clone() - lr * db;
        },
        Adam { lr, beta1, beta2, epsilon } => {
            unimplemented!("Adam optimizer not implemented yet lr={}, beta1={}, beta2={}, epsilon={}", lr, beta1, beta2, epsilon);
        },
        None => (),
    }
}