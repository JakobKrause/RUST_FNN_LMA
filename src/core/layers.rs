use crate::prelude::*;
use crate::core::optimizers::Optimization;
use crate::rand_array;
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
        if perceptron == 0 || prev == 0 {
            return Err(NNError::InvalidLayerConfiguration(
                "Layer dimensions must be greater than 0".to_string()
            ));
        }
        Ok(Self {
            w: rand_array!(prev, perceptron),
            b: rand_array!(1, perceptron),
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
        let dz = self.activation.backward(z, da.clone())?;
        
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
}

impl Optimization for Dense {
    fn optimize(&mut self, dw: Array2<f64>, db: Array2<f64>, config: &OptimizerConfig) {
        apply_optimization(&mut self.w, &mut self.b, dw, db, config);
    }
}