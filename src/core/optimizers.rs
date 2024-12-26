use crate::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Regularizer {
    None,
    L1(f64),
    L2(f64),
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ClipValue {
    None,
    Value(f64),
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GradientClipConfig {
    pub dw: ClipValue, // for weight gradients
    pub db: ClipValue, // for bias gradients
}

impl Default for GradientClipConfig {
    fn default() -> Self {
        Self {
            dw: ClipValue::None,
            db: ClipValue::None,
        }
    }
}

impl GradientClipConfig {
    pub fn new(dw: Option<f64>, db: Option<f64>) -> Self {
        Self {
            dw: match dw {
                Some(value) => ClipValue::Value(value),
                None => ClipValue::None,
            },
            db: match db {
                Some(value) => ClipValue::Value(value),
                None => ClipValue::None,
            },
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OptimizerConfig {
    pub optimizer_type: OptimizerType,
    pub regularizer: Regularizer,
    pub gradientclip: GradientClipConfig,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum OptimizerType {
    SGD(f64),
    Adam {
        lr: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    Marquardt {
        mu: f64,
        mu_increase: f64,
        mu_decrease: f64,
        min_error: f64,
    },
    None,
}

pub trait Optimization {
    fn optimize(&mut self, dw: Array2<f64>, db: Array2<f64>, optimizer: &OptimizerConfig);
}

pub fn apply_optimization(
    weights: &mut Array2<f64>,
    bias: &mut Array2<f64>,
    mut dw: Array2<f64>,
    mut db: Array2<f64>,
    config: &OptimizerConfig,
) {
    // Apply regularization to gradients
    match &config.regularizer {
        Regularizer::L1(lambda) => {
            let l1_grad = weights.mapv(|w| {
                if w > 0.0 {
                    1.0
                } else if w < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            });
            dw = dw + (*lambda * l1_grad);
        }
        Regularizer::L2(lambda) => {
            dw = dw + (*lambda * weights.clone());
        }
        Regularizer::None => (),
    }

    // Clip gradients based on config
    match config.gradientclip.dw {
        ClipValue::Value(clip_value) => clip_gradients(&mut dw, clip_value),
        ClipValue::None => (),
    }

    match config.gradientclip.db {
        ClipValue::Value(clip_value) => clip_gradients(&mut db, clip_value),
        ClipValue::None => (),
    }

    // Apply optimizer
    match &config.optimizer_type {
        OptimizerType::SGD(lr) => {
            *weights = weights.clone() - *lr * dw;
            *bias = bias.clone() - *lr * db;
        }
        OptimizerType::Adam {
            lr,
            beta1,
            beta2,
            epsilon,
        } => {
            unimplemented!(
                "Adam optimizer not implemented yet lr={}, beta1={}, beta2={}, epsilon={}",
                lr,
                beta1,
                beta2,
                epsilon
            );
        }
        OptimizerType::Marquardt {
            mu: _,
            mu_increase: _,
            mu_decrease: _,
            min_error: _,
        } => {
            // Marquardt optimization is handled separately in the fit() method
        }
        OptimizerType::None => (),
    }
}

fn clip_gradients(grad: &mut Array2<f64>, clip_value: f64) {
    grad.mapv_inplace(|x| x.clamp(-clip_value, clip_value));
}