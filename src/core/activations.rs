#[allow(unused)]
use crate::prelude::*;
use fastapprox::fast::tanh as faster_tanh;
// use fastapprox::fast::exp as fast_exp;
// use fastapprox::fast::sigmoid as sigmoid_fast;

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum Activation {
    Linear,
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
}

impl Activation {
    pub fn forward(&self, z: Array2<f64>) -> Result<Array2<f64>> {
        Ok(match self {
            Self::Linear => z,
            Self::Relu => relu_forward(z),
            Self::Sigmoid => sigmoid_forward(z),
            Self::Tanh => tanh_forward(z),
            Self::Softmax => softmax_forward(z),
        })
    }

    pub fn backward(&self, z: Array2<f64>, da: Array2<f64>) -> Result<Array2<f64>> {
        Ok(match self {
            Self::Linear => da,
            Self::Relu => da * relu_backward(z),
            Self::Sigmoid => da * sigmoid_backward(z),
            Self::Tanh => da * tanh_backward(z),
            Self::Softmax => da * softmax_backward(z),
        })
    }
}

fn sigmoid_forward(z: Array2<f64>) -> Array2<f64> {
    z.mapv(|z| 1.0 / (1.0 + (-z).exp()))

    // z.mapv(|z| 1.0 / (1.0 + fast_exp(-z as f32) as f64))

    // z.mapv(|z| sigmoid_fast(z as f32) as f64)
}

fn sigmoid_backward(z: Array2<f64>) -> Array2<f64> {
    z.mapv(|z| {
        let s = 1.0 / (1.0 + (-z).exp());
        s * (1.0 - s)
    })

    
    // z.mapv(|z| {
    //     let s = 1.0 / (1.0 + fast_exp(-z as f32) as f64);
    //     s * (1.0 - s)
    // })

    // z.mapv(|z| {
    //     let s = sigmoid_fast(z as f32) as f64;
    //     s * (1.0 - s)
    // })

}

fn relu_forward(z: Array2<f64>) -> Array2<f64> {
    z.mapv(|z| if z >= 0.0 {z} else {0.0})
}

fn relu_backward(z: Array2<f64>) -> Array2<f64> {
    z.mapv(|z| if z >= 0.0 {1.0} else {0.0})
}

fn tanh_forward(z: Array2<f64>) -> Array2<f64> {
    z.mapv(|z| z.tanh())
    
    // z.mapv(|z| faster_tanh(z as f32) as f64)
}

fn tanh_backward(z: Array2<f64>) -> Array2<f64> {
    // z.mapv(|z| 1.0 - z.tanh().powf(2.0))

    z.mapv(|z| {
        let t = faster_tanh(z as f32) as f64;
        1.0 - t * t
    })
}

fn softmax_forward(z: Array2<f64>) -> Array2<f64> {
    let exp = z.mapv(|z| e.powf(z));
    exp.clone() / exp.sum()
}

fn softmax_backward(z: Array2<f64>) -> Array2<f64> {
    softmax_forward(z)
}