// use ndarray::Array2;
// use csv::Writer;
// use std::error::Error;

// pub fn write_jacobian_to_csv(jacobian: &Array2<f64>, file_path: &str) -> Result<(), Box<dyn Error>> {
//     // Create a CSV writer that writes to the specified file path
//     let mut wtr = Writer::from_path(file_path)?;

//     // Iterate over each row in the Array2
//     for row in jacobian.outer_iter() {
//         // Convert each element in the row to a string
//         let record: Vec<String> = row.iter().map(|x| x.to_string()).collect();
        
//         // Write the record to the CSV file
//         wtr.write_record(&record)?;
//     }

//     // Ensure all data is flushed to the file
//     wtr.flush()?;
//     Ok(())
// }


use csv::Writer;
use ndarray::Array2;
use std::error::Error;
// use std::iter::IntoIterator;

/// Trait to abstract over different types that can be iterated into CSV records.
pub trait ToCsvRecord {
    fn to_csv_record(&self) -> Vec<String>;
}

impl ToCsvRecord for Array2<f64> {
    fn to_csv_record(&self) -> Vec<String> {
        self.iter().map(|x| x.to_string()).collect()
    }
}

impl ToCsvRecord for Vec<f64> {
    fn to_csv_record(&self) -> Vec<String> {
        self.iter().map(|x| x.to_string()).collect()
    }
}

/// Writes data to a CSV file. Supports both 1D and 2D data structures.
///
/// # Arguments
///
/// * `data` - A reference to the data implementing `ToCsvRecord`.
/// * `file_path` - The path to the CSV file where the data will be written.
pub fn write_to_csv<T: ToCsvRecord>(data: &T, file_path: &str) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(file_path)?;

    // For simplicity, treat all data as a single record
    let record = data.to_csv_record();
    wtr.write_record(&record)?;

    wtr.flush()?;
    Ok(())
}
