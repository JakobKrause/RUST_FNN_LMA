use rkl::benchmark::functions::Multimodal1D;
use rkl::prelude::*;
use rkl::plot::*;
use rkl::core::output;

// use flame;
// use flamescope;
// use std::fs::File;

fn main() -> Result<()> {

    let x_vec: Vec<f64> = (1..=1000).map(|x|0.001* x as f64 ).collect();
    let y_vec: Vec<f64> = x_vec.multimodal1_d();

    // Convert to Array2 where each row is a single sample
    let x = Array2::from_shape_vec((x_vec.len(), 1), x_vec.clone()).unwrap();
    let y = Array2::from_shape_vec((y_vec.len(), 1), y_vec.clone()).unwrap();

    let mut model = Sequential::builder()
    .add_dense(1, 20, Activation::Tanh)?
    .add_dense(20, 20, Activation::Tanh)?
    .add_dense(20, 10, Activation::Tanh)?
    .add_dense(10, 1, Activation::Linear)?
    .optimizer(OptimizerType::Marquardt { mu: 0.01, mu_increase: 10., mu_decrease: 0.1, min_error: 1e-10 })
    // .optimizer(OptimizerType::SGD(0.1))
    // .regularizer(Regularizer::L2(0.01))
    // .clip_weights(0.1)
    // .clip_biases(0.1)
    .loss(Loss::MSE)
    .build()?;

    model.summary();

    model.fit(x.clone(), y.clone(), 2000, true)?;

    let prediction = model.predict(x.clone())?;
    let mse = model.evaluate(x.clone(), y.clone());

    println!("Evaluation...");
    println!("MSE: {:?}", mse);
    // output::write_to_csv( "mse.csv").unwrap();

    // Call the plotting function
    plot_comparision::plot_comparison(
        x.as_slice().unwrap(),
        y.as_slice().unwrap(),
        prediction.as_slice().unwrap(),
        "multimodal_comparison.png",
    ).unwrap();

    // plot_errors_over_epochs::plot_errors_over_epochs(&model.errors, "error_over_epochs.png").unwrap();
    // plot_errors_over_epochs::plot_errors_over_epochs(&model.errors_new, "errornew_over_epochs.png").unwrap();
    // plot_errors_over_epochs::plot_errors_over_epochs(&model.mus, "mu_over_epochs.png").unwrap();
    output::write_to_csv(&model.errors, "error.csv").unwrap();
    // output::write_to_csv(&model.errors_new, "errornew.csv").unwrap();
    output::write_to_csv(&model.mus, "mus.csv").unwrap();
    output::write_to_csv(&model.time, "time.csv").unwrap();
    output::write_to_csv(&model.errors_val, "error_val.csv").unwrap();
    output::write_to_csv(&model.errors_test, "error_test.csv").unwrap();
    output::write_to_csv(&model.m, "m.csv").unwrap();
    // output::write_to_csv(&model.errors_val_new, "error_val_new.csv").unwrap();
    // plot_errors_over_epochs::plot_errors_over_epochs(&model.gradients, "gradient_over_epochs.png").unwrap();

    
    //model.save("./test.model")?;

    Ok(())
}
