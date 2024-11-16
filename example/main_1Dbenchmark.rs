use rkl::benchmark::functions::Multimodal1D;
use rkl::prelude::*;
use rkl::plot::plot_comparision;
fn main() -> Result<()> {
    let x_vec: Vec<f64> = (1..=100).map(|x| x as f64 * 0.01).collect();
    let y_vec: Vec<f64> = x_vec.multimodal1_d();

    // Convert to Array2 where each row is a single sample
    let x = Array2::from_shape_vec((x_vec.len(), 1), x_vec.clone()).unwrap();
    let y = Array2::from_shape_vec((y_vec.len(), 1), y_vec.clone()).unwrap();

    let mut model = Sequential::builder()
    .add_dense(1, 10, Activation::Sigmoid)?
    .add_dense(10, 10, Activation::Sigmoid)?
    .add_dense(10, 1, Activation::Linear)?
    .optimizer(OptimizerType::Marquardt { mu: 0.1, mu_increase: 10., mu_decrease: 0.001, min_error: 1e-6 })
    .regularizer(Regularizer::L1(0.05))
    .clip_weights(1.0)
    .clip_biases(1.0)
    .loss(Loss::MSE)
    .build()?;

    model.summary();

    model.fit(x.clone(), y.clone(), 2000, true)?;

    let prediction = model.predict(x.clone())?;
    let mse = model.evaluate(x.clone(), y.clone());

    println!("Evaluation...");
    println!("MSE: {:?}", mse);

    // Call the plotting function
    plot_comparision::plot_comparison(
        &x.into_raw_vec(),
        &y.into_raw_vec(),
        &prediction.into_raw_vec(),
        "multimodal_comparison.png",
    ).unwrap();


    model.save("./test.model")?;

    Ok(())
}
