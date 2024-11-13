use crate::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Loss {
    MSE,
    NLL,
    None,
}

pub fn criteria(
    y_hat: Array2<f64>, 
    y: Array2<f64>, 
    loss_ty: Loss,
    weights: &[&Array2<f64>],
    regularizer: &Regularizer
) -> Result<(f64, Array2<f64>, f64)> {
    // Check shapes
    if y_hat.shape() != y.shape() {
        return Err(NNError::LayerShapeMismatch(format!(
            "Prediction shape {:?} doesn't match target shape {:?}",
            y_hat.shape(), y.shape()
        )));
    }

    let (loss, da) = match loss_ty {
        Loss::MSE => {
            let da = y_hat.clone() - y.clone();
            let loss = (0.5 * (&y_hat - &y).mapv(|a| a.powf(2.0))).mean().unwrap();
            (loss, da)
        },
        Loss::NLL => {
            // Avoid division by zero and log(0)
            let epsilon = 1e-15;
            let y_hat_safe = y_hat.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
            
            let da = -((&y / &y_hat_safe) - ((1.0 - &y)/(1.0 - &y_hat_safe)));
            let loss = -(y.clone() * y_hat_safe.mapv(|y| y.ln()) 
                + (1.0 - y) * (1.0 - &y_hat_safe).mapv(|y| y.ln())).mean().unwrap();
            (loss, da)
        },
        Loss::None => {
            let da = y_hat.clone() - y.clone();
            let loss = (&y_hat - &y).mean().unwrap();
            (loss, da)
        },
    };

    // Add regularization loss
    let reg_loss = match regularizer {
        Regularizer::L1(lambda) => {
            lambda * weights.iter()
                .map(|w| w.mapv(|x| x.abs()).sum())
                .sum::<f64>()
        },
        Regularizer::L2(lambda) => {
            0.5 * lambda * weights.iter()
                .map(|w| w.mapv(|x| x.powi(2)).sum())
                .sum::<f64>()
        },
        Regularizer::None => 0.0,
    };

    Ok((loss, da, loss + reg_loss))
}

// pub fn regularization_loss(weights: &[&Array2<f64>], regularizer: &Regularizer) -> f64 {
//     match regularizer {
//         Regularizer::L1(lambda) => {
//             weights.iter()
//                 .map(|w| lambda * w.mapv(|x| x.abs()).sum())
//                 .sum()
//         },
//         Regularizer::L2(lambda) => {
//             weights.iter()
//                 .map(|w| 0.5 * lambda * w.mapv(|x| x.powi(2)).sum())
//                 .sum()
//         },
//         Regularizer::None => 0.0,
//     }
// }