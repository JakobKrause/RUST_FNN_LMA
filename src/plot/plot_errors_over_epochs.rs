use plotters::prelude::*;

pub fn plot_errors_over_epochs(
    errors: &[f64],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let epochs = errors.len();

    // Adjust errors to avoid log of zero or negative numbers
    let adjusted_errors: Vec<f64> = errors
        .iter()
        .map(|&e| if e <= 0.0 { 1e-10 } else { e })
        .collect();

    // Compute log of errors
    let log_errors: Vec<f64> = adjusted_errors.iter().map(|&e| e.log10()).collect();

    // Find min and max log error values
    let y_min = log_errors
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        .floor();
    let y_max = log_errors
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil();

    let mut chart = ChartBuilder::on(&root)
        .caption("Model Error over Epochs (Log Scale)", ("sans-serif", 30).into_font())
        .margin(5)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0..epochs, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Error (Log Scale)")
        .y_label_formatter(&|y| format!("1e{:.0}", y))
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            log_errors.iter().enumerate().map(|(epoch, &log_error)| (epoch, log_error)),
            &BLUE,
        ))?
        .label("Training Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Draw the legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Error plot has been saved as '{}'", filename);

    Ok(())
}
