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
        let z = a.dot(&self.w) + self.b.clone();
        let a = self.activation.forward(z.clone())?;
        Ok((z, a))
    }

    pub fn backward(&self, z: Array2<f64>, a: Array2<f64>, da: Array2<f64>) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        let dz = self.activation.backward(z, da)?;
        let dw = (a.reversed_axes().dot(&dz))/(dz.len() as f64);
        let db = dz.clone().sum_axis(Axis(0)).insert_axis(Axis(0))/(dz.len() as f64);
        let da = dz.dot(&self.w.t());
        Ok((dw, db, da))
    }
}

impl Optimization for Dense {
    fn optimize(&mut self, dw: Array2<f64>, db: Array2<f64>, optimizer: Optimizer) {
        apply_optimization(&mut self.w, &mut self.b, dw, db, optimizer);
    }
}