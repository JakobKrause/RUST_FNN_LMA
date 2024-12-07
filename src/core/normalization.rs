use ndarray::Array2;

pub trait Normalization {
    fn to_unity(&mut self, lb: f64, ub: f64);
    fn from_unity(&mut self, lb: f64, ub: f64);
}

impl Normalization for Array2<f64> {
    fn to_unity(&mut self, lb: f64, ub: f64) {
        let range = ub - lb;

        // If the range is zero or nearly zero, all values become 0.0
        if range.abs() < f64::EPSILON {
            for val in self.iter_mut() {
                *val = 0.0;
            }
        } else {
            for val in self.iter_mut() {
                *val = (*val - lb) / range;
            }
        }
    }

    fn from_unity(&mut self, lb: f64, ub: f64) {
        let range = ub - lb;

        // If the range is zero or nearly zero, all values become lb
        if range.abs() < f64::EPSILON {
            for val in self.iter_mut() {
                *val = lb;
            }
        } else {
            for val in self.iter_mut() {
                *val = *val * range + lb;
            }
        }
    }
}