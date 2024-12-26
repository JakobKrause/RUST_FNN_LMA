use crate::prelude::*;
use crate::core::optimizers::Optimization;
use crate::core::activations::Activation;

pub trait LayerTrait {
    fn new(perceptron: usize, prev: usize, activation: Activation) -> Result<Self>
    where
        Self: Sized;
    
    fn typ(&self) -> String;
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Dense {
    pub w: Array2<f64>,
    pub b: Array2<f64>,
    pub activation: Activation,
}

impl LayerTrait for Dense {
    fn new(perceptron: usize, prev: usize, activation: Activation) -> Result<Self> {
        // if perceptron == 0 || prev == 0 {
        //     return Err(NNError::InvalidLayerConfiguration(
        //         "Layer dimensions must be greater than 0".to_string()
        //     ));
        // }
        // Ok(Self {
        //     w: rand_array!(prev, perceptron),
        //     b: rand_array!(1, perceptron),
        //     activation,
        // })

        if perceptron == 0 || prev == 0 {
            return Err(NNError::InvalidLayerConfiguration(
                "Layer dimensions must be greater than 0".to_string(),
            ));
        }

        // Compute the bound based on the Normalized Xavier Initialization
        let bound = (6.0_f64).sqrt() / ((prev + perceptron) as f64).sqrt();
        let lower = -bound;
        let upper = bound;

        // Initialize weights with Uniform distribution in [lower, upper]
        let w = Array2::random((prev, perceptron), Uniform::new(lower, upper));

        // Initialize biases to zero
        let b = Array2::random((1, perceptron), Uniform::new(lower, upper));

        Ok(Self {
            w,
            b,
            activation,
        })

    }

    fn typ(&self) -> String {
        "Dense".into()
    }
}

impl Dense {
    pub fn forward(&self, a: Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        // Check shapes
        if a.ncols() != self.w.nrows() {
            return Err(NNError::LayerShapeMismatch(format!(
                "Input shape {:?} is incompatible with weight shape {:?}",
                a.shape(), self.w.shape()
            )));
        }

        // z = a * W + b
        let z = a.dot(&self.w) + &self.b;
        let a_next = self.activation.forward(z.clone())?;
        Ok((z, a_next))
    }

    pub fn backward(&self, z: Array2<f64>, a_prev: Array2<f64>, da: Array2<f64>) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        // Compute dZ
        let dz = self.activation.backward(z.clone(), da.clone())?;
        
        // Compute dW = (1/m) * (a_prev.T * dZ)
        let m = dz.nrows() as f64;
        let dw = a_prev.t().dot(&dz) / m;
        
        // Compute db = (1/m) * sum(dZ)
        let db = dz.sum_axis(Axis(0)).insert_axis(Axis(0)) / m;
        
        // Compute da_prev = dZ * W.T
        let da_prev = dz.dot(&self.w.t());

        // Check shapes
        if dw.shape() != self.w.shape() {
            return Err(NNError::LayerShapeMismatch(format!(
                "Weight gradient shape {:?} doesn't match weight shape {:?}",
                dw.shape(), self.w.shape()
            )));
        }

        if db.shape() != self.b.shape() {
            return Err(NNError::LayerShapeMismatch(format!(
                "Bias gradient shape {:?} doesn't match bias shape {:?}",
                db.shape(), self.b.shape()
            )));
        }

        Ok((dw, db, da_prev))
    }

    pub fn backward_jacobian(
        &self,
        z: Array2<f64>,        // shape: [P, out_dim_this_layer]
        a_prev: Array2<f64>,   // shape: [P, in_dim_this_layer]
        da: Array2<f64>,       // shape: [P, out_dim_this_layer]
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {

        // 1) Compute dZ = dE/dZ = activation'(Z) * dE/dA
        let dz = self.activation.backward(z, da.clone())?;
        // shape of dz: [P, out_dim_this_layer]

        // println!("dz: R:{}, C:{}",dz.nrows(), dz.ncols());
        // println!("aprev: R:{}, C:{}",a_prev.nrows(), a_prev.ncols());
        // println!("da: R:{}, C:{}",da.nrows(), da.ncols());

        let batch_size = dz.nrows();
        let in_dim     = a_prev.ncols();
        let out_dim    = dz.ncols();

        // 2) We want dW per sample:
        //    for each sample p: dW_p = a_prev[p,:]^T (1x in_dim)  * dz[p,:] (1x out_dim)
        //    so shape: dW_per_sample => [P, in_dim, out_dim]

        let mut d_w_per_sample = Array2::<f64>::zeros((batch_size, in_dim*out_dim));
        for p in 0..batch_size {
            let a_prev_row: ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>> = a_prev.slice(s![p, ..]);  // shape [in_dim]
            let dz_row: ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>     = dz.slice(s![p, ..]);      // shape [out_dim]
            // Outer product => (in_dim x out_dim)
            // a_prev_row is 1D, so we can do something like:

            //dW_per_sample[[p, ..]]=a_prev_row.t().dot(&dz_row);            
            for i in 0..in_dim {
                let base = i * out_dim;
                for j in 0..out_dim {
                    let index = base + j;
                    d_w_per_sample[[p, index]] = a_prev_row[i] * dz_row[j];
                }
            }
        }

        // 3) We want dB per sample:
        //    for each sample p: dB_p = dz[p,:]
        //    so shape: [P, 1, out_dim]
        let mut d_b_per_sample = Array2::<f64>::zeros((batch_size, out_dim));
        for p in 0..batch_size {
            for j in 0..out_dim {
                d_b_per_sample[[p, j]] = dz[[p,j]];
            }
        }

        // 4) Compute da_prev = dZ * W^T as usual (still shape [P, in_dim])
        let da_prev = dz.dot(&self.w.t());

        // Return them
        Ok((d_w_per_sample, d_b_per_sample, da_prev))
    }
    
    
}

impl Optimization for Dense {
    fn optimize(&mut self, dw: Array2<f64>, db: Array2<f64>, config: &OptimizerConfig) {
        apply_optimization(&mut self.w, &mut self.b, dw, db, config);
    }
}