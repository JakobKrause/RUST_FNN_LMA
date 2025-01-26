# Neural Network Regression in Rust with Levenberg Marquardt Training

A Rust implementation demonstrating neural network regression on multimodal functions using the [RKL (Rust Keras Like)](https://github.com/AhmedBoin/Rust-Keras-Like) library.

## Features

- Neural network regression using Levenberg-Marquardt optimization
- Visualization of predictions vs actual data
- Performance metrics tracking and export
- Custom activation functions and layer architectures
- CSV output for detailed analysis

## Example Usage

```rust
use rkl::benchmark::functions::Multimodal1D;
use rkl::prelude::*;
use rkl::plot::*;
use rkl::core::output;

fn main() -> Result<()> {
    // Generate sample data
    let x_vec: Vec<f64> = (1..=1000).map(|x| 0.001 * x as f64).collect();
    let y_vec: Vec<f64> = multimodal1_d(x_vec);

    // Convert to Array2 where each row is a single sample
    let x = Array2::from_shape_vec((x_vec.len(), 1), x_vec.clone()).unwrap();
    let y = Array2::from_shape_vec((y_vec.len(), 1), y_vec.clone()).unwrap();

    // Create model architecture
    let mut model = Sequential::builder()
        .add_dense(1, 20, Activation::Tanh)?
        .add_dense(20, 20, Activation::Tanh)?
        .add_dense(20, 10, Activation::Tanh)?
        .add_dense(10, 1, Activation::Linear)?
        .optimizer(OptimizerType::Marquardt { 
            mu: 0.01, 
            mu_increase: 10., 
            mu_decrease: 0.1, 
            min_error: 1e-10 
        })
        .loss(Loss::MSE)
        .build()?;

    // Train and evaluate
    model.fit(x.clone(), y.clone(), epochs: 2000, verbose: true)?;
    let prediction = model.predict(x.clone())?;
    let mse = model.evaluate(x.clone(), y.clone());

    // Plot results
    plot_comparision::plot_comparison(
        x.as_slice().unwrap(),
        y.as_slice().unwrap(),
        prediction.as_slice().unwrap(),
        "multimodal_comparison.png",
    )?;

    // Export results to CSV
    output::write_to_csv(&model.errors, "error.csv")?;

    Ok(())
}