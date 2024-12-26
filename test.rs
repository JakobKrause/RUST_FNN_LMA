use ndarray::{array, Array2, s};

// Assume the `scale_and_concatenate` function is defined in the same module.
// If it's in a different module, adjust the `use` statement accordingly.

// fn scale_and_concatenate(a_prev: &Array2<f64>, dz: &Array2<f64>) -> Array2<f64> {
//     // ... function implementation ...
// }

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Test the `scale_and_concatenate` function with valid input.
    #[test]
    fn test_scale_and_concatenate_success() {
        // Define the input arrays
        let a_prev = array![
            [1.0, 2.0],
            [3.0, 4.0]
        ];

        let dz = array![
            [5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0]
        ];

        // Define the expected result
        let expected = array![
            [5.0, 10.0, 6.0, 12.0, 7.0, 14.0],
            [24.0, 32.0, 27.0, 36.0, 30.0, 40.0]
        ];

        // Call the function
        let result = scale_and_concatenate(&a_prev, &dz);

        // Assert that the result matches the expected output
        assert_eq!(result, expected);
    }

    /// Test the `scale_and_concatenate` function for dimension mismatch.
    #[test]
    #[should_panic(expected = "Dimension mismatch between a_prev and dz")]
    fn test_scale_and_concatenate_dimension_mismatch() {
        // Define input arrays with mismatched dimensions
        let a_prev = array![
            [1.0, 2.0],
            [3.0, 4.0]
        ];

        let dz = array![
            [5.0, 6.0, 7.0]  // Only 1 row instead of 2
        ];

        // This should panic due to dimension mismatch
        scale_and_concatenate(&a_prev, &dz);
    }

    /// Additional test to verify behavior with empty arrays.
    #[test]
    fn test_scale_and_concatenate_empty_arrays() {
        let a_prev: Array2<f64> = Array2::zeros((0, 0));
        let dz: Array2<f64> = Array2::zeros((0, 0));
        let expected: Array2<f64> = Array2::zeros((0, 0));

        let result = scale_and_concatenate(&a_prev, &dz);
        assert_eq!(result, expected);
    }

    /// Additional test with larger matrices.
    #[test]
    fn test_scale_and_concatenate_large_matrices() {
        let a_prev = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];

        let dz = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ];

        let expected = array![
            [1.0, 2.0, 2.0, 4.0, 3.0, 6.0],
            [12.0, 15.0, 20.0, 25.0, 18.0, 30.0],
            [35.0, 40.0, 48.0, 54.0, 45.0, 54.0]
        ];

        let result = scale_and_concatenate(&a_prev, &dz);
        assert_eq!(result, expected);
    }
}
