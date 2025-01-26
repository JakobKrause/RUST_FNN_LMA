use crate::core::losses::criteria;
use crate::core::optimizers::*;
use crate::core::ClipValue;
use crate::core::GradientClipConfig;
use crate::prelude::*;
use faer::linalg::solvers::Solve;
use faer::Mat;
use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom; // for shuffle
use rand::thread_rng; // for random state
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Sequential<T: LayerTrait> {
    pub layers: Vec<T>,
    pub optimizer_config: OptimizerConfig,
    pub loss: Loss,
    lb_input: Vec<f64>,
    ub_input: Vec<f64>,
    lb_output: Vec<f64>,
    ub_output: Vec<f64>,
    #[serde(skip)] // Skip serialization
    pub errors: Vec<f64>,
    pub errors_new: Vec<f64>,
    pub errors_val: Vec<f64>,
    pub errors_test: Vec<f64>,
    pub errors_val_new: Vec<f64>,
    pub time: Vec<f64>,
    pub mus: Vec<f64>,
    pub m: Vec<f64>,
    pub gradients: Vec<f64>,
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
            errors_new: vec![],
            errors_val: vec![],
            errors_test: vec![],
            errors_val_new: vec![],
            time: vec![],
            mus: vec![],
            m: vec![],
            gradients: vec![],
            lb_input: vec![],
            ub_input: vec![],
            lb_output: vec![],
            ub_output: vec![],
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

    pub fn compile(
        &mut self,
        optimizer_type: OptimizerType,
        regularizer: Regularizer,
        loss: Loss,
        gradientclip: GradientClipConfig,
    ) {
        self.optimizer_config = OptimizerConfig {
            optimizer_type,
            regularizer,
            gradientclip,
        };
        self.loss = loss;
    }

    pub fn fit(
        &mut self,
        mut x: Array2<f64>,
        mut y: Array2<f64>,
        epochs: usize,
        verbose: bool,
    ) -> Result<()> {
        // let _scope_guard = flame::start_guard("fit");
        self.calc_boundaries(x.clone(), y.clone());

        x.to_unity(self.lb_input[0], self.ub_input[0]);
        y.to_unity(self.lb_output[0], self.ub_output[0]);

        let mut num_samples = x.nrows();
        let num_outputs = y.ncols();

        // Calculate how many samples go to validation and test
        let val_fraction = 0.2;
        let test_fraction = 0.1;

        let val_size = (val_fraction * num_samples as f64).round() as usize;
        let test_size = (test_fraction * num_samples as f64).round() as usize;

        // Shuffle the row indices
        let mut indices: Vec<usize> = (0..num_samples).collect();
        indices.shuffle(&mut thread_rng());

        // Split indices into validation, test, and training parts
        let val_indices = &indices[..val_size];
        let test_indices = &indices[val_size..(val_size + test_size)];
        let train_indices = &indices[(val_size + test_size)..];

        // Gather validation arrays
        let val_x = x.select(Axis(0), val_indices);
        let val_y = y.select(Axis(0), val_indices);

        // Gather test arrays
        let test_x = x.select(Axis(0), test_indices);
        let test_y = y.select(Axis(0), test_indices);

        // Gather training arrays (override x, y)
        let x = x.select(Axis(0), train_indices);
        let y = y.select(Axis(0), train_indices);

        num_samples = num_samples - val_size - test_size;

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
                x.shape(),
                (x.nrows(), self.layers[0].w.nrows())
            )));
        }

        match self.optimizer_config.optimizer_type {
            OptimizerType::Marquardt {
                ref mut mu,
                mu_increase,
                mu_decrease,
                min_error: _,
            } => {
                let mut current_mu = *mu;
                let start = Instant::now();

                // let mut j_nalgebra = DMatrix::<f64>::zeros(num_samples, self.count_parameters());
                // let mut e_nalgebra = DVector::<f64>::zeros(num_samples);

                for epoch in 0..epochs {
                    // // Initialize error vector
                    // let mut err = Array1::<f64>::zeros(num_samples * num_outputs);

                    // // Compute initial predictions
                    // let y_pred = self.predict_with_normalization(x.clone(), false)?;
                    // let error = &y_pred - y.clone();

                    // // Flatten error vector
                    // err.assign(
                    //     &error
                    //         .clone()
                    //         .to_shape((num_samples * num_outputs,))
                    //         .unwrap(),
                    // );

                    // // println!("err: {}", err);
                    // Initialize error vector
                    let mut err = Array1::<f64>::zeros(num_samples * num_outputs);

                    // Compute initial predictions
                    let y_pred = self.predict_with_normalization(x.clone(), false)?;
                    let error = &y_pred - y.clone();

                    // Flatten error vector
                    err.assign(
                        &error
                            .clone()
                            .to_shape((num_samples * num_outputs,))
                            .unwrap(),
                    );

                    let mse = err
                        .mapv(|val| val.powi(2)) // square each error
                        .mean() // take the mean
                        .unwrap(); // safe to unwrap if non-empty

                    self.errors.push(mse);

                    // Initialize error vector val
                    let mut err_val = Array1::<f64>::zeros(val_size * num_outputs);

                    // Compute initial predictions
                    let y_pred = self.predict_with_normalization(val_x.clone(), false)?;
                    let error = &y_pred - val_y.clone();
                    // println!("{}", val_y);

                    // Flatten error vector
                    err_val.assign(&error.clone().to_shape((val_size * num_outputs,)).unwrap());

                    let mse_val = err_val
                        .mapv(|val| val.powi(2)) // square each error
                        .mean() // take the mean
                        .unwrap(); // safe to unwrap if non-empty

                    self.errors_val.push(mse_val);

                    // Initialize error vector test
                    let mut err_test = Array1::<f64>::zeros(test_size * num_outputs);

                    // Compute initial predictions
                    let y_pred = self.predict_with_normalization(test_x.clone(), false)?;
                    let error = &y_pred - test_y.clone();

                    // Flatten error vector
                    err_test.assign(&error.clone().to_shape((test_size * num_outputs,)).unwrap());

                    let mse_test = err_test
                        .mapv(|val| val.powi(2)) // square each error
                        .mean() // take the mean
                        .unwrap(); // safe to unwrap if non-empty

                    self.errors_test.push(mse_test);

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
                    let weights: Vec<&Array2<f64>> =
                        self.layers.iter().map(|layer| &layer.w).collect();

                    // Compute loss and initial gradient
                    let (raw_loss, da, reg_loss) = criteria(
                        a_cache.last().unwrap().clone(),
                        y.clone(),
                        self.loss.clone(),
                        &weights,
                        &self.optimizer_config.regularizer,
                    )?;
                    // da = y_hat - y

                    // self.errors.push(raw_loss);

                    if verbose {
                        // Helps you tune λ (regularization strength)
                        // If reg_loss >> raw_loss: regularization might be too strong
                        // If reg_loss << raw_loss: regularization might be too weak
                        println!(
                            "Epoch: {}/{} raw loss: {} reg loss: {}",
                            epoch, epochs, raw_loss, reg_loss
                        );
                    }

                    // Initialize Jacobian matrix and error vector

                    let jacobian: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
                        self.backward_jacobian(&x, &y)?;
                    // if true {
                        // let jacobian = self.compute_jacobian_finite(&x, &y)?;
                    // }

                    // let delta_w = self.compute_delta_w_lm(jacobian, da, current_mu);
                    let mut mm = 1;
                    const ALPHA: f64 = 0.0;
                    const MAX_MU: f64 = 1000.;

                    loop {
                        // println!("m: {}", mm);
                        // println!("mu: {}", current_mu);
                        let delta_w = self.compute_delta_w_lm(jacobian.clone(), da.clone(), current_mu);

                        // Update weights
                        let mut w: Vec<f64> = Vec::new();
                        for layer in &self.layers {
                            w.extend(layer.w.iter());
                            w.extend(layer.b.iter());
                        }

                        let w_new: Vec<f64> = w
                            .iter()
                            .zip(delta_w.iter())
                            .map(|(wi, dwi)| wi + dwi)
                            .collect();

                        let mut idx = 0;
                        for layer in &mut self.layers {
                            let num_w = layer.w.len();
                            let num_b = layer.b.len();

                            layer.w.assign(
                                &Array2::from_shape_vec(
                                    layer.w.raw_dim(),
                                    w_new[idx..idx + num_w].to_vec(),
                                )
                                .unwrap(),
                            );
                            idx += num_w;

                            layer.b.assign(
                                &Array2::from_shape_vec(
                                    layer.b.raw_dim(),
                                    w_new[idx..idx + num_b].to_vec(),
                                )
                                .unwrap(),
                            );
                            idx += num_b;
                        }
                        // if current_mu >= MAX_MU {break}

                        // Forward propagation
                        let mut z_cache = vec![];
                        let mut a_cache = vec![];
                        let mut a = x.clone();
                        a_cache.push(a.clone());

                        for layer in self.layers.iter() {
                            let (z, a_next) = layer.forward(a)?;
                            z_cache.push(z);
                            a_cache.push(a_next.clone());
                            a = a_next;
                        }

                        // Collect weights for regularization
                        let weights: Vec<&Array2<f64>> =
                            self.layers.iter().map(|layer| &layer.w).collect();

                        // Compute loss and initial gradient
                        let (raw_loss_new, _, _) = criteria(
                            a_cache.last().unwrap().clone(),
                            y.clone(),
                            self.loss.clone(),
                            &weights,
                            &self.optimizer_config.regularizer,
                        )?;

                        // Initialize error vector val
                        let mut err_val_new = Array1::<f64>::zeros(val_size * num_outputs);

                        // Compute initial predictions
                        let y_pred = self.predict_with_normalization(val_x.clone(), false)?;
                        let error = &y_pred - val_y.clone();

                        // Flatten error vector
                        err_val_new
                            .assign(&error.clone().to_shape((val_size * num_outputs,)).unwrap());

                        let mse_val_new = err_val_new
                            .mapv(|val| val.powi(2)) // square each error
                            .mean() // take the mean
                            .unwrap(); // safe to unwrap if non-empty

                        self.errors_val_new.push(mse_val_new);

                        // Compare errors and adjust mu
                        // let diff = raw_loss - raw_loss_new;
                        let diff = mse_val - mse_val_new;


                        if diff > 0. {
                            // println!("decrease");
                            // self.mu *= self.mu_decrease;
                            current_mu =
                                ALPHA * current_mu + (1. - ALPHA) * current_mu * mu_decrease;
                                // self.m.push(m as f64);
                            break
                            // Some(e_new.clone().into_shape_with_order(e_new.len()).unwrap());
                            //Ok(raw_loss_new)
                        }
                        else if mm > 5 {
                            // println!("break");
                            // self.m.push(m as f64);
                            break
                        }                        
                        else {
                            // self.mu *= self.mu_increase;

                            if current_mu < MAX_MU {
                                // println!("mu:{}", current_mu);
                                current_mu = ALPHA * current_mu + (1. - ALPHA) * current_mu * mu_increase;
                                // println!("mu:{}", current_mu);
                            }
                            // Restore old weights
                            let mut idx = 0;
                            for layer in &mut self.layers {
                                let num_w = layer.w.len();
                                let num_b = layer.b.len();

                                layer.w.assign(
                                    &Array2::from_shape_vec(
                                        layer.w.raw_dim(),
                                        w[idx..idx + num_w].to_vec(),
                                    )
                                    .unwrap(),
                                );
                                idx += num_w;

                                layer.b.assign(
                                    &Array2::from_shape_vec(
                                        layer.b.raw_dim(),
                                        w[idx..idx + num_b].to_vec(),
                                    )
                                    .unwrap(),
                                );
                                idx += num_b;                                
                            }
                            // println!("mm+1");
                            mm+=1;
                        }

                        self.errors_new.push(raw_loss_new);
                        

                        
    
                        
    
                        // At the end, write back current_mu to self.optimizer_config
                        if let OptimizerType::Marquardt { ref mut mu, .. } =
                            self.optimizer_config.optimizer_type
                        {
                            *mu = current_mu;
                        }
                        
                    }
                    self.m.push((mm) as f64);
                    self.mus.push(current_mu);
                    self.time.push(start.elapsed().as_secs_f64());
                }
                Ok(())
            }
            OptimizerType::SGD(_) | OptimizerType::Adam { .. } => {
                let start = Instant::now();

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
                    let weights: Vec<&Array2<f64>> =
                        self.layers.iter().map(|layer| &layer.w).collect();

                    // Compute loss and initial gradient
                    let (raw_loss, mut da, reg_loss) = criteria(
                        a_cache.last().unwrap().clone(),
                        y.clone(),
                        self.loss.clone(),
                        &weights,
                        &self.optimizer_config.regularizer,
                    )?;
                    // da = y_hat - y

                    self.errors.push(raw_loss);

                    if verbose {
                        // Helps you tune λ (regularization strength)
                        // If reg_loss >> raw_loss: regularization might be too strong
                        // If reg_loss << raw_loss: regularization might be too weak
                        println!(
                            "Epoch: {}/{} raw loss: {} reg loss: {}",
                            epoch, epochs, raw_loss, reg_loss
                        );
                    }

                    // Backward propagation
                    let mut dw_cache = vec![];
                    let mut db_cache = vec![];

                    for ((layer, z), a_prev) in self
                        .layers
                        .iter()
                        .rev()
                        .zip(z_cache.iter().rev())
                        .zip(a_cache.iter().rev().skip(1))
                    {
                        let (dw, db, da_prev) = layer.backward(z.clone(), a_prev.clone(), da)?;
                        dw_cache.insert(0, dw);
                        db_cache.insert(0, db);
                        da = da_prev;
                    }

                    // Update weights
                    for (layer, (dw, db)) in self
                        .layers
                        .iter_mut()
                        .zip(dw_cache.iter().zip(db_cache.iter()))
                    {
                        layer.optimize(dw.clone(), db.clone(), &self.optimizer_config);
                    }

                    self.time.push(start.elapsed().as_secs_f64());
                }
                Ok(())
            }
            OptimizerType::None => Err(NNError::OptimizerNotSet),
        }
    }

    pub fn backward_jacobian(
        &mut self,
        x_batch: &Array2<f64>,
        y_batch: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let batch_size = x_batch.nrows(); // P
        let output_dim = y_batch.ncols(); // M
        let total_params = self.count_parameters();

        // 1) Forward pass to collect (Z, A) for each layer
        //    ( same as your usual forward pass, but we keep A's & Z's in a vec )
        let mut a_cache: Vec<Array2<f64>> = vec![];
        let mut z_cache: Vec<Array2<f64>> = vec![];

        let mut a = x_batch.clone();
        a_cache.push(a.clone()); // input is a0
        for layer in &self.layers {
            let z = a.dot(&layer.w) + &layer.b; // shape [P, out_dim]
            let a_next = layer.activation.forward(z.clone())?;
            z_cache.push(z);
            a_cache.push(a_next.clone());
            a = a_next;
        }

        // "a" is now the final output (prediction). Suppose your error is e_{p,m} = (a - y)
        let error = &a - y_batch; // shape: [P, M]

        // 2) We'll do a "reverse iteration" but keep partial derivatives *per sample*
        //    Start with dA(L) = derivative of error wrt final output activation = error
        let mut da = error;

        let mut col_offset = 0;
        let mut jacobian = Array2::<f64>::zeros((batch_size * output_dim, total_params));

        // Iterate layers in reverse
        for (layer, (z, a_prev)) in self
            .layers
            .iter()
            .rev()
            .zip(z_cache.iter().rev().zip(a_cache.iter().rev().skip(1)))
        {
            let (d_w_ps, d_b_ps, da_prev) =
                layer.backward_jacobian(z.clone(), a_prev.clone(), da.clone())?;

            da = da_prev;

            let dw_db = ndarray::concatenate(Axis(1), &[d_w_ps.view(), d_b_ps.view()])
                .expect("Concatenate did not work!");

            let nrows = dw_db.nrows();
            let ncols = dw_db.ncols();
            let start = col_offset - ncols + total_params;
            let end = col_offset + total_params;

            jacobian.slice_mut(s![..nrows, start..end]).assign(&dw_db);

            // Update column offset for next iteration
            col_offset -= ncols;
        }
        //output::write_jacobian_to_csv(&jacobian, "jacobian.csv").unwrap();

        divide_rows_by_last_value(&mut jacobian);

        // output::write_jacobian_to_csv(&jacobian, "jacobian.csv").unwrap();

        Ok(jacobian)
    }

    fn calc_boundaries(&mut self, x: Array2<f64>, y: Array2<f64>) {
        self.lb_input = x
            .iter()
            .cloned()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .map(|max_value| vec![max_value]) // Wrap the maximum value in a Vec
            .unwrap_or_else(Vec::new);

        self.ub_input = x
            .iter()
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .map(|max_value| vec![max_value]) // Wrap the maximum value in a Vec
            .unwrap_or_else(Vec::new);

        self.lb_output = y
            .iter()
            .cloned()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .map(|max_value| vec![max_value]) // Wrap the maximum value in a Vec
            .unwrap_or_else(Vec::new);

        self.ub_output = y
            .iter()
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .map(|max_value| vec![max_value]) // Wrap the maximum value in a Vec
            .unwrap_or_else(Vec::new);
    }

    pub fn evaluate(&self, x: Array2<f64>, y: Array2<f64>) -> Result<f64> {
        let weights: Vec<&Array2<f64>> = self.layers.iter().map(|layer| &layer.w).collect();
        let y_hat = self.predict(x)?;
        // println!("{}",y);
        // Handle the Result returned by criteria
        let (loss, _, _) = criteria(
            y_hat,
            y,
            self.loss.clone(),
            &weights,
            &self.optimizer_config.regularizer,
        )?; // Use ? to propagate the error

        Ok(loss)
    }

    pub fn predict(&self, x: Array2<f64>) -> Result<Array2<f64>> {
        self.predict_with_normalization(x, true)
    }

    pub fn predict_with_normalization(
        &self,
        mut x: Array2<f64>,
        apply_normalization: bool,
    ) -> Result<Array2<f64>> {
        if apply_normalization {
            x.to_unity(self.lb_input[0], self.ub_input[0]);
        }

        for layer in self.layers.iter() {
            (_, x) = layer.forward(x)?;
        }

        if apply_normalization {
            x.from_unity(self.lb_output[0], self.ub_output[0]);
        }

        Ok(x)
    }

    pub fn predict_with_weights(
        &self,
        mut x: Array2<f64>,
        weights: &Vec<f64>,
    ) -> Result<Array2<f64>> {
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

            let w = Array2::from_shape_vec(w_shape, w_slice.to_vec()).map_err(|_| {
                NNError::InvalidWeightShape("Failed to reshape weight vector".to_string())
            })?;
            let b = Array2::from_shape_vec(b_shape, b_slice.to_vec()).map_err(|_| {
                NNError::InvalidBiasShape("Failed to reshape bias vector".to_string())
            })?;

            // Compute z = x * w + b
            let z = x.dot(&w) + &b;

            // Apply activation function
            x = layer.activation.forward(z)?;
        }
        Ok(x)
    }

    pub fn count_parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.w.len() + layer.b.len())
            .sum()
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
            errors_new: vec![],
            errors_val: vec![],
            errors_test: vec![],
            errors_val_new: vec![],
            time: vec![],
            mus: vec![],
            m: vec![],
            gradients: vec![],
            lb_input: vec![],
            ub_input: vec![],
            lb_output: vec![],
            ub_output: vec![],
        })
    }

    fn compute_jacobian_finite(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<Array2<f64>> {
        // Number of samples and outputs
        let num_samples = x.nrows();
        let num_outputs = y.ncols();
        let num_parameters: usize = self // number of parameters
            .count_parameters();

        // Initialize Jacobian matrix and error vector
        let mut j = Array2::<f64>::zeros((num_samples * num_outputs, num_parameters));
        let mut err = Array1::<f64>::zeros(num_samples * num_outputs);

        // Compute initial predictions
        let y_pred = self.predict_with_normalization(x.clone(), false)?;
        let error = &y_pred - y;

        // Flatten error vector
        err.assign(
            &error
                .clone()
                .to_shape((num_samples * num_outputs,))
                .unwrap(),
        );

        // Flatten all weights and biases into a single vector
        let w: Vec<f64> = self
            .layers
            .iter()
            .flat_map(|layer| layer.w.iter().chain(layer.b.iter()))
            .cloned()
            .collect();

        let delta = 1e-5;

        // const BASE_DELTA: f64 = 1e-7;
        // const MIN_DELTA: f64 = 1e-8;
        // const MAX_DELTA: f64 = 1e-6;

        let derivs: Vec<Array1<f64>> = (0..num_parameters)
            .into_par_iter()
            .map(|i| {
                // Create a local copy of weights
                let mut w_clone = w.clone();

                // Perturb weight
                w_clone[i] += delta;

                // Use the perturbed weights directly without cloning the network
                let y_pred_perturbed = self.predict_with_weights(x.clone(), &w_clone).unwrap();

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

        // output::write_jacobian_to_csv(&j, "j_finite.csv").unwrap();
        Ok(j)
    }

    fn compute_delta_w_lm(
        &mut self,
        jacobian: Array2<f64>,
        da: Array2<f64>,
        mu: f64,
    ) -> Array2<f64> {
        // Convert to nalgebra types
        // let j_nalgebra = DMatrix::from_row_slice(jacobian.nrows(), jacobian.ncols(), jacobian.as_slice().unwrap());
        // let e_nalgebra = DVector::from_column_slice(da.as_slice().unwrap());

        // // Convert to faer types
        let nrows = jacobian.nrows();
        let ncols = jacobian.ncols();
        let mut j_faer = Mat::<f64>::zeros(nrows, ncols);
        for j in 0..j_faer.ncols() {
            for i in 0..j_faer.nrows() {
                j_faer[(i, j)] = jacobian[(i, j)];
            }
        }

        let nrows = da.nrows();
        let ncols = da.ncols();
        let mut e_faer = Mat::<f64>::zeros(nrows, ncols);
        for j in 0..e_faer.ncols() {
            for i in 0..e_faer.nrows() {
                e_faer[(i, j)] = da[(i, j)];
            }
        }

        // Compute j^T j + mu * I
        // let jt_j = &j_nalgebra.transpose() * &j_nalgebra;
        // // let jt_j = compute_jt_j_rayon(&j_nalgebra);
        // // let jt_j = j_nalgebra.tr_mul(&j_nalgebra);
        // let identity = DMatrix::<f64>::identity(jacobian.ncols(), jacobian.ncols());
        // let jt_j_mu_i = jt_j.clone() + mu * identity;

        let jt_j = j_faer.transpose() * &j_faer;
        let identity = Mat::<f64>::identity(jacobian.ncols(), jacobian.ncols());
        let jt_j_mu_i = &jt_j + mu * &identity;

        // Compute j^T err
        // let jt_e = &j_nalgebra.transpose() * &e_nalgebra;
        // // let jt_e = compute_jt_e_rayon(&j_nalgebra, &e_nalgebra);
        let jt_e = j_faer.transpose() * &e_faer;

        // Calculate gradient norm (J^T * e)
        let gradient_norm = jt_e.norm_l2();
        self.gradients.push(gradient_norm);

        // // Solve for delta_w
        // let delta_w = jt_j_mu_i
        //     .lu()
        //     .solve(&(-jt_e))
        //     .ok_or(NNError::Other("Failed to solve linear system".to_string()))?;

        // let delta_w = match jt_j_mu_i.cholesky().and_then(|ch| Some(ch.solve(&(-jt_e)))) {
        //     Some(d) => d,
        //     None => return jacobian
        // };
        
        // let delta_w = match jt_j_mu_i.cholesky(Side::Lower) {
        //     Some(chol) => chol.solve(&(-jt_e)),
        //     None => return Err(NNError::SerializationError(Box::new(
        //         bincode::ErrorKind::Custom("Failed to solve linear system".to_string())
        //     ))),
        // };

        // Compute the QR decomposition.
        // let qr = jt_j_mu_i.qr();
        let test = jt_j_mu_i.llt(faer::Side::Lower).unwrap();
        // let test = jt_j_mu_i.qr();
        // let delta_w = qr.solve(&(-jt_e));
        let delta_w = test.solve(&(-jt_e));

        let mut delta: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
            Array2::<f64>::zeros(delta_w.shape());

        for j in 0..delta_w.ncols() {
            for i in 0..delta_w.nrows() {
                delta[(i, j)] = delta_w[(i, j)];
            }
        }

        delta
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

fn divide_rows_by_last_value(array: &mut Array2<f64>) {
    // Iterate over each row in the array along Axis(0)
    for mut row in array.axis_iter_mut(Axis(0)) {
        // Safely get the last element of the row
        if let Some(&last_val) = row.last() {
            // Ensure the last_val is not zero to avoid division by zero
            if last_val != 0.0 {
                // Divide each element in the row by last_val
                for elem in row.iter_mut() {
                    *elem /= last_val;
                }
            } else {
                // If the last_val is zero, you can choose to handle it as needed.
                // For simplicity, we'll skip division for this row.
                println!("Warning: Last element is zero. Skipping division for this row.");
            }
        }
    }
}