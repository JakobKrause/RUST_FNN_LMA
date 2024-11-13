use rkl::prelude::*;
use rkl::plot::plot_comparision;
fn main() -> Result<()> {
    let x = array![[1., 2.], [3., 4.], [5., 6.]];
    let y = array![[3.], [7.], [11.]];

    let mut model = Sequential::builder()
        .add_dense(2, 10, Activation::Linear)?
        .add_dense(10, 6, Activation::Linear)?
        .add_dense(6, 1, Activation::Linear)?
        .optimizer(OptimizerType::SGD(0.01))
        .regularizer(Regularizer::L2(0.01))
        .loss(Loss::MSE)
        .build()?;
    
    model.summary();
        
    model.fit(x.clone(), y.clone(), 5000, true)?;
    
    // Get predictions for all training points
    let predictions = model.predict(x.clone())?;
    
    // Convert data for plotting
    let inputs: Vec<[f64; 2]> = x.outer_iter()
        .map(|row| [row[0], row[1]])
        .collect();
    let targets: Vec<f64> = y.iter().copied().collect();
    let pred_vec: Vec<f64> = predictions.iter().copied().collect();

    // Generate the plot
    plot_comparision::plot_2d_surface(
        &inputs,
        &targets,
        "2d_surface.png",
    ).unwrap();

    // Continue with test prediction
    let x_test = array![[2., 3.]];
    let y_test = array![[5.]];
    
    let eval = model.evaluate(x_test.clone(), y_test)?;
    println!("\ncost: {}\n", eval);
    
    let prediction = model.predict(x_test)?;
    println!("prediction: {}", prediction);

    model.save("./test.model")?;

    Ok(())
}