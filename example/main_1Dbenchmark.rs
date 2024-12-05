use rkl::benchmark::functions::Multimodal1D;
use rkl::prelude::*;
use rkl::plot::plot_comparision;

// use flame;
// use flamescope;
// use std::fs::File;

fn main() -> Result<()> {

    //FF let main_guard = flame::start_guard("main");

    let x_vec: Vec<f64> = (1..=1000).map(|x| x as f64 * 0.001).collect();
    let y_vec: Vec<f64> = x_vec.multimodal1_d();

    // Convert to Array2 where each row is a single sample
    let x = Array2::from_shape_vec((x_vec.len(), 1), x_vec.clone()).unwrap();
    let y = Array2::from_shape_vec((y_vec.len(), 1), y_vec.clone()).unwrap();

    let mut model = Sequential::builder()
    .add_dense(1, 20, Activation::Tanh)?
    .add_dense(20, 20, Activation::Tanh)?
    .add_dense(20, 10, Activation::Tanh)?
    .add_dense(10, 1, Activation::Linear)?
    .optimizer(OptimizerType::Marquardt { mu: 0.1, mu_increase: 10.0, mu_decrease: 0.1, min_error: 1e-5 })
    //.optimizer(OptimizerType::SGD(0.3))
    // .regularizer(Regularizer::L1(1.))
    // .clip_weights(0.01)
    // .clip_biases(0.01)
    .loss(Loss::MSE)
    .build()?;

    model.summary();

    model.fit(x.clone(), y.clone(), 100, true)?;

    let prediction = model.predict(x.clone())?;
    let mse = model.evaluate(x.clone(), y.clone());

    println!("Evaluation...");
    println!("MSE: {:?}", mse);

    // Call the plotting function
    plot_comparision::plot_comparison(
        x.as_slice().unwrap(),
        y.as_slice().unwrap(),
        prediction.as_slice().unwrap(),
        "multimodal_comparison.png",
    ).unwrap();


    model.save("./test.model")?;

    //FF main_guard.end();
    //FF flamescope::dump(&mut File::create("flamescope.json").unwrap()).unwrap();


    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main() {
        // Call main and unwrap the Result
        let result = main().unwrap();
        
        // Since main returns Ok(()), we can assert the execution completed successfully
        assert!(result == ());
        
        // Additional assertions can be added here to verify model performance
        // For example, you could load the saved model and check its predictions
    }
    }

