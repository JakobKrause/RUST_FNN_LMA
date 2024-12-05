#[allow(unused)]
use crate::prelude::*;

// #[macro_export]
// macro_rules! rand_array {
//     ($($x:expr),*) => {
//         {
//             //Array::random(($($x,)*), Uniform::new(-0.01, 0.01))
//             // let n = 1;
//             Array::random(($($x,)*), Uniform::new(-1.0, 1.0))
//         }
//     };
// }

#[macro_export]
macro_rules! rand_array {
    ($($x:expr),*) => {
        {
            Array2::random(($($x,)*), Uniform::new(-3., 3.))
        }
    };
}

#[macro_export]
macro_rules! Model {
    (input_shape $i:expr ,$(dense $x:expr, activation $a:expr),*) => {
        {
            let x = vec![$($x),*];
            let a = vec![$($a),*];
            let mut layers = vec![];
            layers.push(Dense::new($i, x[0], a[0])?);
            for i in 0..x.len()-1 {
                layers.push(Dense::new(x[i], x[i+1], a[i+1])?);
            }
            Sequential::new(&layers)
        }
    };
}

#[macro_export]
macro_rules! Dense {
    ($x:expr ,$y:expr, $a:expr) => {
        {
            Dense::new(
                $x, $y, $a
            );
        }
    };
}