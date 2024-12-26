/// This is a multimodal 1d function
/// https://machinelearningmastery.com/1d-test-functions-for-function-optimization/
/// -> Multimodal Function 3
pub trait Multimodal1D {
    /// Multimodal Function 3
    fn multimodal1_d(& self) -> Self;
}

impl Multimodal1D for f64 {
    fn multimodal1_d(&self) -> Self {
        multimodal1_d(*self)
    }
}

impl Multimodal1D for Vec<f64> {
    fn multimodal1_d(&self) -> Self {
        self.iter().map(|&x| multimodal1_d(x)).collect()
    }
}


fn multimodal1_d(x: f64) -> f64 {
    let x_scaled = x*10 as f64;
    -x * x_scaled.sin() * 1.
    //x*x
    //0 as f64 + x*0 as f64
}