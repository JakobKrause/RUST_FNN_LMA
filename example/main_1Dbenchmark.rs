use rkl::benchmark::functions::Multimodal1D;
use rkl::prelude::*;
use rkl::plot::plot_comparision;
fn main() -> Result<()> {
    let x_vec: Vec<f64> = (1..=1000).map(|x| x as f64 * 0.01).collect();
    let y_vec: Vec<f64> = x_vec.multimodal1_d();

    // Convert to Array2 where each row is a single sample
    let x = Array2::from_shape_vec((x_vec.len(), 1), x_vec.clone()).unwrap();
    let y = Array2::from_shape_vec((y_vec.len(), 1), y_vec.clone()).unwrap();

    // let num_samples = input_vec.len();
    // let input_dim = 1;
    // let output_dim = 1;

    // let x = Matrix::new(num_samples, input_dim, input_vec.clone());
    // let y = Matrix::new(num_samples, output_dim, output_vec.clone());

    let mut model = Sequential::builder()
    .add_dense(1, 10, Activation::Sigmoid)?
    .add_dense(10, 6, Activation::Sigmoid)?
    .add_dense(6, 1, Activation::Linear)?
    .optimizer(Optimizer::SGD(0.01))
    .loss(Loss::MSE)
    .build()?;

    model.summary();

    model.fit(x.clone(), y.clone(), 500, true)?;

    // let x_test = array![[0.5]];
    // let y_test = array![[5.]];

    // let eval = model.evaluate(x_test.clone(), y_test)?;
    // println!("\ncost: {}\n", eval);

    // let prediction = model.predict(x_test)?;
    // println!("prediction: {}", prediction);

    let prediction = model.predict(x.clone())?;
    let mse = model.evaluate(x, y);

    println!("Evaluation...");
    println!("MSE: {:?}", mse);

    // Call the plotting function
    plot_comparision::plot_comparison(
        &x_vec,
        &y_vec,
        &prediction.into_raw_vec(),
        "multimodal_comparison.png",
    ).unwrap();


    model.save("./test.model")?;

    Ok(())
}
