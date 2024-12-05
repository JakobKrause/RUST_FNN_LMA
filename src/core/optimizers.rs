use crate::prelude::*;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use rayon::prelude::*;

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
    }

    pub fn compute_jacobian(
        &mut self,
        network: &Sequential<Dense>,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<()> {
        // Number of samples and outputs
        let num_samples = x.nrows();
        let num_outputs = y.ncols();
        let num_parameters: usize = network // number of parameters
            .count_parameters();

        // Initialize Jacobian matrix and error vector
        let mut j = Array2::<f64>::zeros((num_samples * num_outputs, num_parameters));
        let mut err = Array1::<f64>::zeros(num_samples * num_outputs);

        // Compute initial predictions
        let y_pred = network.predict(x.clone())?;
        let error = &y_pred - y;

        // Flatten error vector
        err.assign(
            &error
                .clone()
                .to_shape((num_samples * num_outputs,))
                .unwrap(),
        );

        // Flatten all weights and biases into a single vector
        let w: Vec<f64> = network
            .layers
            .iter()
            .flat_map(|layer| layer.w.iter().chain(layer.b.iter()))
            .cloned()
            .collect();

        let delta = 1e-7;

        let derivs: Vec<Array1<f64>> = (0..num_parameters)
            .into_par_iter()
            .map(|i| {
                // Create a local copy of weights
                let mut w_clone = w.clone();

                // Perturb weight
                w_clone[i] += delta;

                // Use the perturbed weights directly without cloning the network
                let y_pred_perturbed = network.predict_with_weights(x.clone(), &w_clone).unwrap();

                let delta_y = y_pred_perturbed - y_pred.clone();
                let delta_y_flat = delta_y.to_shape((num_samples * num_outputs,)).unwrap();
                let deriv = &delta_y_flat / delta;

                deriv
            })
            .collect();

        // Assemble the Jacobian matrix from the derivatives
        for (i, deriv) in derivs.into_iter().enumerate() {
            j.slice_mut(s![.., i]).assign(&deriv);
        }

        self.jacobian = Some(j);
        self.error_vector = Some(err);
        Ok(())
    }

    pub fn update_weights(
        &mut self,
        network: &mut Sequential<Dense>,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<f64> {
        let j = self
            .jacobian
            .as_ref()
            .ok_or(NNError::Other("Jacobian not computed".to_string()))?;
        let err = self
            .error_vector
            .as_ref()
            .ok_or(NNError::Other("Error vector not computed".to_string()))?;

        // Convert to nalgebra types
        let j_nalgebra = DMatrix::from_row_slice(j.nrows(), j.ncols(), j.as_slice().unwrap());
        let e_nalgebra = DVector::from_column_slice(err.as_slice().unwrap());

        // Compute j^T j + mu * I
        let jt_j = &j_nalgebra.transpose() * &j_nalgebra;
        let identity = DMatrix::<f64>::identity(j.ncols(), j.ncols());
        let jt_j_mu_i = jt_j.clone() + self.mu * identity;

        // Compute j^T err
        let jt_e = &j_nalgebra.transpose() * &e_nalgebra;

        // Solve for delta_w
        let delta_w = jt_j_mu_i
            .lu()
            .solve(&(-jt_e))
            .ok_or(NNError::Other("Failed to solve linear system".to_string()))?;

        // Update weights
        let mut w: Vec<f64> = Vec::new();
        for layer in &network.layers {
            w.extend(layer.w.iter());
            w.extend(layer.b.iter());
        }

        let w_new: Vec<f64> = w
            .iter()
            .zip(delta_w.iter())
            .map(|(wi, dwi)| wi + dwi)
            .collect();

        let mut idx = 0;
        for layer in &mut network.layers {
            let num_w = layer.w.len();
            let num_b = layer.b.len();

            layer.w.assign(
                &Array2::from_shape_vec(layer.w.raw_dim(), w_new[idx..idx + num_w].to_vec())
                    .unwrap(),
            );
            idx += num_w;

            layer.b.assign(
                &Array2::from_shape_vec(layer.b.raw_dim(), w_new[idx..idx + num_b].to_vec())
                    .unwrap(),
            );
            idx += num_b;
        }

        // Compute new error
        let y_pred_new = network.predict(x.clone())?;
        let e_new = &y_pred_new - y;
        let error_new = e_new.mapv(|x| x.powi(2)).sum();

        // Compare errors
        let error_old = self
            .error_vector
            .as_ref()
            .unwrap()
            .mapv(|x| x.powi(2))
            .sum();
        if error_new < error_old {
            self.mu *= self.mu_decrease;
            self.error_vector = Some(e_new.clone().into_shape_with_order(e_new.len()).unwrap());
            Ok(error_new)
        } else {
            self.mu *= self.mu_increase;
            // Restore old weights
            let mut idx = 0;
            for layer in &mut network.layers {
                let num_w = layer.w.len();
                let num_b = layer.b.len();

                layer.w.assign(
                    &Array2::from_shape_vec(layer.w.raw_dim(), w[idx..idx + num_w].to_vec())
                        .unwrap(),
                );
                idx += num_w;

                layer.b.assign(
                    &Array2::from_shape_vec(layer.b.raw_dim(), w[idx..idx + num_b].to_vec())
                        .unwrap(),
                );
                idx += num_b;
            }
            Ok(error_old)
        }
    }
}
