use crate::prelude::*;
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;

// #[derive(Serialize, Deserialize, Debug, Clone)]
// pub enum Optimizer {
//     SGD(f64),
//     Adam {
//         lr: f64,
//         beta1: f64,
//         beta2: f64,
//         epsilon: f64,
//     },
//     None,
// }

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
    pub dw: ClipValue,  // for weight gradients
    pub db: ClipValue,  // for bias gradients
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
        OptimizerType::Marquardt { mu: _, mu_increase: _, mu_decrease: _, min_error: _ } => {
            // Marquardt optimization is handled separately in the fit() method
        },
        OptimizerType::None => (),
    }
}

fn clip_gradients(grad: &mut Array2<f64>, clip_value: f64) {
    grad.mapv_inplace(|x| x.clamp(-clip_value, clip_value));
}

pub struct MarquardtOptimizer {
    pub mu: f64,
    pub mu_increase: f64,
    pub mu_decrease: f64,
    pub min_error: f64,
    jacobian: Option<Array2<f64>>,
    error_vector: Option<Array1<f64>>,
}

impl MarquardtOptimizer {
    pub fn new(mu: f64, mu_increase: f64, mu_decrease: f64, min_error: f64) -> Self {
        Self {
            mu,
            mu_increase,
            mu_decrease,
            min_error,
            jacobian: None,
            error_vector: None,
        }
    }pub fn update_weights(&mut self, network: &mut Sequential<Dense>) -> Result<f64> {
        let j = self.jacobian.as_ref().unwrap();
        let error_vec = self.error_vector.as_ref().unwrap();
        
        // Compute J^T J and J^T e
        let jt = j.t();
        let jt_j = jt.dot(j);
        let jt_e = jt.dot(error_vec);
        
        // Add damping factor
        let n = jt_j.nrows();
        let mut damped_jt_j = jt_j.clone();
        for i in 0..n {
            damped_jt_j[[i, i]] += self.mu;
        }
        
        // Solve the system (J^T J + μI)δ = J^T e
        let delta = match damped_jt_j.solve(&jt_e) {
            Ok(d) => d,
            Err(linalg_err) => return Err(NNError::SerializationError(Box::new(
                bincode::ErrorKind::Custom(format!("Failed to solve linear system: {:?}", linalg_err))
            ))),
        };
        
        // Update weights
        let mut param_idx = 0;
        for layer in network.layers.iter_mut() {
            // Update weights
            for i in 0..layer.w.nrows() {
                for j in 0..layer.w.ncols() {
                    layer.w[[i, j]] += delta[param_idx];
                    param_idx += 1;
                }
            }
            
            // Update biases
            for i in 0..layer.b.nrows() {
                for j in 0..layer.b.ncols() {
                    layer.b[[i, j]] += delta[param_idx];
                    param_idx += 1;
                }
            }
        }
        
        // Return sum of squared errors
        Ok(error_vec.dot(error_vec))
    }
    
    pub fn compute_jacobian(&mut self, network: &mut Sequential<Dense>, x: &Array2<f64>, y: &Array2<f64>) -> Result<()> {
        let batch_size = x.nrows();
        let output_size = y.ncols();
        let num_params = network.count_parameters();
        
        let mut jacobian = Array2::zeros((batch_size * output_size, num_params));
        let mut error_vector = Array1::zeros(batch_size * output_size);
        
        // Forward pass to get current outputs
        let outputs = network.predict(x.clone())?;
        
        // Compute error vector
        for i in 0..batch_size {
            for j in 0..output_size {
                let error_idx = i * output_size + j;
                error_vector[error_idx] = y[[i, j]] - outputs[[i, j]];
            }
        }
    
        // Compute Jacobian using finite differences
        let epsilon = 1e-7;
        let mut param_idx = 0;
        
        // Clone the network to avoid borrow checker issues
        let mut network_clone = network.clone();
        
        for (layer_idx, layer) in network.layers.iter().enumerate() {
            let w_shape = layer.w.raw_dim();
            let b_shape = layer.b.raw_dim();
            
            // For weights
            for i in 0..w_shape[0] {
                for j in 0..w_shape[1] {
                    let orig_value = layer.w[[i, j]];
                    
                    // Perturb weight in clone
                    network_clone.layers[layer_idx].w[[i, j]] = orig_value + epsilon;
                    let outputs_plus = network_clone.predict(x.clone())?;
                    
                    // Compute derivative
                    for k in 0..batch_size {
                        for l in 0..output_size {
                            let error_idx = k * output_size + l;
                            jacobian[[error_idx, param_idx]] = 
                                (outputs_plus[[k, l]] - outputs[[k, l]]) / epsilon;
                        }
                    }
                    
                    // Restore original value
                    network_clone.layers[layer_idx].w[[i, j]] = orig_value;
                    param_idx += 1;
                }
            }
            
            // For biases
            for i in 0..b_shape[0] {
                for j in 0..b_shape[1] {
                    let orig_value = layer.b[[i, j]];
                    
                    // Perturb bias in clone
                    network_clone.layers[layer_idx].b[[i, j]] = orig_value + epsilon;
                    let outputs_plus = network_clone.predict(x.clone())?;
                    
                    // Compute derivative
                    for k in 0..batch_size {
                        for l in 0..output_size {
                            let error_idx = k * output_size + l;
                            jacobian[[error_idx, param_idx]] = 
                                (outputs_plus[[k, l]] - outputs[[k, l]]) / epsilon;
                        }
                    }
                    
                    // Restore original value
                    network_clone.layers[layer_idx].b[[i, j]] = orig_value;
                    param_idx += 1;
                }
            }
        }
        
        self.jacobian = Some(jacobian);
        self.error_vector = Some(error_vector);
        Ok(())
    }
}