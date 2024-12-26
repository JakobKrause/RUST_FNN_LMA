use ndarray::Array2;
use csv::Writer;
use std::error::Error;

pub fn write_jacobian_to_csv(jacobian: &Array2<f64>, file_path: &str) -> Result<(), Box<dyn Error>> {
    // Create a CSV writer that writes to the specified file path
    let mut wtr = Writer::from_path(file_path)?;

    // Iterate over each row in the Array2
    for row in jacobian.outer_iter() {
        // Convert each element in the row to a string
        let record: Vec<String> = row.iter().map(|x| x.to_string()).collect();
        
        // Write the record to the CSV file
        wtr.write_record(&record)?;
    }

    // Ensure all data is flushed to the file
    wtr.flush()?;
    Ok(())
}
