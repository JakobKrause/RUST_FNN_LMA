use crate::core::losses::criteria;
use crate::core::optimizers::*;
use crate::core::GradientClipConfig;
use crate::core::ClipValue;
use crate::prelude::*;
use std::fs::File;
use std::io::{Read, Write};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Sequential<T: LayerTrait> {
    pub layers: Vec<T>,
    pub optimizer_config: OptimizerConfig,
    pub loss: Loss,
    #[serde(skip)] // Skip serialization
    pub errors: Vec<f64>,
}

pub struct SequentialBuilder {
    layers: Vec<Dense>,
    optimizer_config: OptimizerConfig,
    loss: Loss,
    
}

impl Sequential<Dense> {
    pub fn builder() -> SequentialBuilder {
        SequentialBuilder::new()
    }
    pub fn new(layers: &[Dense]) -> Result<Self> {
        if layers.is_empty() {
            return Err(NNError::EmptyModel);
        }
        Ok(Self {
            layers: layers.try_into().unwrap(),
            optimizer_config: OptimizerConfig {
                optimizer_type: OptimizerType::None,
                regularizer: Regularizer::None,
                gradientclip: GradientClipConfig::default(),
            },
            loss: Loss::None,
            errors: vec![], 
        })
    }

    pub fn summary(&self) {
        let mut total_param = 0;
        let mut res = "\nModel Sequential\n".to_string();
        res.push_str("-------------------------------------------------------------\n");
        res.push_str("Layer (Type)\t\t Output shape\t\t No.of params\n");
        for layer in self.layers.iter() {
            let a = layer.w.len();
            let b = layer.b.len();
            total_param += a + b;
            res.push_str(&format!(
                "{}\t\t\t  (None, {})\t\t  {}\n",
                layer.typ(),
                b,
                a + b
            ));
        }
        res.push_str("-------------------------------------------------------------\n");
        res.push_str(&format!("Total params: {}\n", total_param));
        println!("{}", res);
    }

    pub fn compile(&mut self, optimizer_type: OptimizerType, regularizer: Regularizer, loss: Loss, gradientclip: GradientClipConfig) {
        self.optimizer_config = OptimizerConfig {
            optimizer_type,
            regularizer,
            gradientclip,
        };
        self.loss = loss;
    }

    pub fn fit(&mut self, x: Array2<f64>, y: Array2<f64>, epochs: usize, verbose: bool) -> Result<()> {
        // let _scope_guard = flame::start_guard("fit");
        if matches!(self.optimizer_config.optimizer_type, OptimizerType::None) {
            return Err(NNError::OptimizerNotSet);
        }
        if matches!(self.loss, Loss::None) {
            return Err(NNError::LossNotSet);
        }
    
        // Check input shapes
        if x.ncols() != self.layers[0].w.nrows() {
            return Err(NNError::InvalidInputShape(format!(
                "Input shape {:?} doesn't match first layer input shape {:?}",
                x.shape(), (x.nrows(), self.layers[0].w.nrows())
            )));
        }

        match self.optimizer_config.optimizer_type {
            OptimizerType::Marquardt { mu, mu_increase, mu_decrease, min_error } => {
                let mut optimizer = MarquardtOptimizer::new(mu, mu_increase, mu_decrease, min_error);
                let num_samples = x.nrows();
                for epoch in 0..epochs {
                    // Compute Jacobian and error vector
                    // println!("Computing Jacobian...........");
                    optimizer.compute_jacobian(self, &x, &y)?;
                    // println!("Computing Jacobian finished!");

                    // Update weights and get new error
                    // println!("Updating parameters...........");
                    let error = optimizer.update_weights(self, &x, &y)?;
                    let mse  = error/num_samples as f64;

                    // println!("Updating parameters finished!");

                    self.errors.push(mse);
                    
                    if verbose {
                        // Print the current epoch, error, and mu
                        println!(
                            "Epoch: {}/{} | Error: {:.12} | mu: {:.6}",
                            epoch + 1,
                            epochs,
                            mse,
                            optimizer.mu
                        );
                    }
                    
                    // Check for convergence
                    if error < optimizer.min_error {
                        break;
                    }
                }
                Ok(())
            },
            OptimizerType::SGD(_) | OptimizerType::Adam { .. } => {

                for epoch in 0..epochs {
                    let mut z_cache = vec![];
                    let mut a_cache = vec![];
                    let mut a = x.clone();
                    a_cache.push(a.clone());
            
                    // Forward propagation
                    for layer in self.layers.iter() {
                        let (z, a_next) = layer.forward(a)?;
                        z_cache.push(z);
                        a_cache.push(a_next.clone());
                        a = a_next;
                    }
            
                    // Collect weights for regularization
                    let weights: Vec<&Array2<f64>> = self.layers.iter()
                        .map(|layer| &layer.w)
                        .collect();
            
                    // Compute loss and initial gradient
                    let (raw_loss, mut da, reg_loss) = criteria(
                        a_cache.last().unwrap().clone(),
                        y.clone(),
                        self.loss.clone(),
                        &weights,
                        &self.optimizer_config.regularizer
                    )?;

                    self.errors.push(raw_loss);
            
                    if verbose {
                        // Helps you tune λ (regularization strength)
                        // If reg_loss >> raw_loss: regularization might be too strong
                        // If reg_loss << raw_loss: regularization might be too weak
                        println!("Epoch: {}/{} raw loss: {} reg loss: {}", epoch, epochs, raw_loss, reg_loss);
                    }
            
                    // Backward propagation
                    let mut dw_cache = vec![];
                    let mut db_cache = vec![];
            
                    for ((layer, z), a_prev) in self.layers.iter().rev()
                        .zip(z_cache.iter().rev())
                        .zip(a_cache.iter().rev().skip(1)) {
                        let (dw, db, da_prev) = layer.backward(z.clone(), a_prev.clone(), da)?;
                        dw_cache.insert(0, dw);
                        db_cache.insert(0, db);
                        da = da_prev;
                    }
            
                    // Update weights
                    for (layer, (dw, db)) in self.layers.iter_mut()
                        .zip(dw_cache.iter().zip(db_cache.iter())) {
                        layer.optimize(dw.clone(), db.clone(), &self.optimizer_config);
                    }
                }
                Ok(())                
            },
            OptimizerType::None => {
                Err(NNError::OptimizerNotSet)
            }
        }
    }

    pub fn evaluate(&self, x: Array2<f64>, y: Array2<f64>) -> Result<f64> {
        let weights: Vec<&Array2<f64>> = self.layers.iter()
            .map(|layer| &layer.w)
            .collect();
    
        // Handle the Result returned by criteria
        let (_, _, loss) = criteria(
            self.predict(x)?,
            y,
            self.loss.clone(),
            &weights,
            &self.optimizer_config.regularizer
        )?;  // Use ? to propagate the error
    
        Ok(loss)
    }

    pub fn predict(&self, mut x: Array2<f64>) -> Result<Array2<f64>> {
        //FF let _scope_guard = flame::start_guard("predict");
        for layer in self.layers.iter() {
            (_, x) = layer.forward(x)?;
        }
        Ok(x)
    }

    pub fn predict_with_weights(&self, mut x: Array2<f64>, weights: &Vec<f64>) -> Result<Array2<f64>> {
        let mut idx = 0;
        for layer in &self.layers {
            let num_w = layer.w.len();
            let num_b = layer.b.len();

            // Extract weights and biases for this layer
            let w_slice = &weights[idx..idx + num_w];
            idx += num_w;
            let b_slice = &weights[idx..idx + num_b];
            idx += num_b;

            // Reshape weights and biases to the correct shapes
            let w_shape = layer.w.raw_dim();
            let b_shape = layer.b.raw_dim();

            let w = Array2::from_shape_vec(w_shape, w_slice.to_vec())
            .map_err(|_| NNError::InvalidWeightShape("Failed to reshape weight vector".to_string()))?;
            let b = Array2::from_shape_vec(b_shape, b_slice.to_vec())
            .map_err(|_| NNError::InvalidBiasShape("Failed to reshape bias vector".to_string()))?;

            // Compute z = x * w + b
            let z = x.dot(&w) + &b;

            // Apply activation function
            x = layer.activation.forward(z)?;
        }
        Ok(x)
    }

    

    pub fn count_parameters(&self) -> usize {
        self.layers.iter().map(|layer| {
            layer.w.len() + layer.b.len()
        }).sum()
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let encoded: Vec<u8> =
            bincode::serialize(&self.layers).map_err(NNError::SerializationError)?;

        File::create(path)
            .map_err(NNError::IoError)? 
            .write_all(&encoded)
            .map_err(NNError::IoError)?; 

        Ok(())
    }

    pub fn load(path: &str) -> Result<Sequential<Dense>> {
        let mut buffer = Vec::new();

        File::open(path)
            .map_err(NNError::IoError)? 
            .read_to_end(&mut buffer)
            .map_err(NNError::IoError)?;

        let layers: Vec<Dense> =
            bincode::deserialize(&buffer).map_err(NNError::SerializationError)?;
        Ok(Sequential {
            layers,
            optimizer_config: OptimizerConfig {
                optimizer_type: OptimizerType::None,
                regularizer: Regularizer::None,
                gradientclip: GradientClipConfig::default(),
            },
            loss: Loss::None,
            errors: vec![],
        })
    }
}

impl SequentialBuilder {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            optimizer_config: OptimizerConfig {
                optimizer_type: OptimizerType::None,
                regularizer: Regularizer::None,
                gradientclip: GradientClipConfig::default(),
            },
            loss: Loss::None,
        }
    }

    pub fn add_dense(
        mut self,
        input_size: usize,
        output_size: usize,
        activation: Activation,
    ) -> Result<Self> {
        let layer = Dense::new(output_size, input_size, activation)?;
        self.layers.push(layer);
        Ok(self)
    }

    pub fn optimizer(mut self, optimizer_type: OptimizerType) -> Self {
        self.optimizer_config.optimizer_type = optimizer_type;
        self
    }

    pub fn regularizer(mut self, regularizer: Regularizer) -> Self {
        self.optimizer_config.regularizer = regularizer;
        self
    }

    pub fn clip_weights(mut self, value: f64) -> Self {
        self.optimizer_config.gradientclip.dw = ClipValue::Value(value);
        self
    }

    pub fn clip_biases(mut self, value: f64) -> Self {
        self.optimizer_config.gradientclip.db = ClipValue::Value(value);
        self
    }

    pub fn loss(mut self, loss: Loss) -> Self {
        self.loss = loss;
        self
    }

    pub fn build(self) -> Result<Sequential<Dense>> {
        // let _scope_guard = flame::start_guard("build");
        if self.layers.is_empty() {
            return Err(NNError::EmptyModel);
        }

        let mut model = Sequential::new(&self.layers)?;
        model.compile(
            self.optimizer_config.optimizer_type,
            self.optimizer_config.regularizer,
            self.loss,
            self.optimizer_config.gradientclip,
        );
        Ok(model)
    }
}