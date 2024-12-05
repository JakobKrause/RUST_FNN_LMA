use plotters::prelude::*;

/// doc TODO
pub fn plot_comparison(
    inputs_vec: &[f64],
    targets_vec: &[f64],
    predictions: &[f64],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // let _scope_guard = flame::start_guard("plot");
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find min and max values for y-axis scaling
    let y_min = targets_vec
        .iter()
        .chain(predictions.iter())
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = targets_vec
        .iter()
        .chain(predictions.iter())
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Multimodal Function: Analytical vs Predicted",
            ("sans-serif", 30).into_font(),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f64..1.0f64, y_min..y_max)?;

    chart.configure_mesh().x_desc("x").y_desc("f(x)").draw()?;

    // Plot the analytical function
    chart
        .draw_series(LineSeries::new(
            inputs_vec.iter().zip(targets_vec.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))?
        .label("Analytical")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Plot the predictions
    chart
        .draw_series(LineSeries::new(
            inputs_vec.iter().zip(predictions.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?
        .label("Predicted")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Draw the legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Plot has been saved as '{}'", filename);

    Ok(())
}


pub fn plot_2d_comparison(
    inputs: &[[f64; 2]],
    targets: &[f64],
    predictions: &[f64],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find min and max values for y-axis scaling
    let y_min = targets
        .iter()
        .chain(predictions.iter())
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = targets
        .iter()
        .chain(predictions.iter())
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "2D Input Function: Target vs Predicted",
            ("sans-serif", 30).into_font(),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..inputs.len(), y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Sample Index")
        .y_desc("Value")
        .draw()?;

    // Plot the target values
    chart
        .draw_series(LineSeries::new(
            targets.iter().enumerate().map(|(i, &y)| (i, y)),
            &RED,
        ))?
        .label("Target")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Plot the predictions
    chart
        .draw_series(LineSeries::new(
            predictions.iter().enumerate().map(|(i, &y)| (i, y)),
            &BLUE,
        ))?
        .label("Predicted")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Draw scatter points for better visibility
    chart.draw_series(PointSeries::of_element(
        targets.iter().enumerate().map(|(i, &y)| (i, y)),
        5,
        &RED,
        &|c, s, st| {
            return EmptyElement::at(c)    // We want the point to be at (x,y)
                + Circle::new((0,0), s, st.filled()) // And a circle centered there
        },
    ))?;

    chart.draw_series(PointSeries::of_element(
        predictions.iter().enumerate().map(|(i, &y)| (i, y)),
        5,
        &BLUE,
        &|c, s, st| {
            return EmptyElement::at(c)
                + Circle::new((0,0), s, st.filled())
        },
    ))?;

    // Draw the legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Plot has been saved as '{}'", filename);

    Ok(())
}


pub fn plot_2d_surface(
    training_inputs: &[[f64; 2]],
    training_targets: &[f64],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Split the drawing area into main plot and legend
    let (main_area, legend_area) = root.split_horizontally(750);

    // Find bounds for x1 and x2
    let x1_min = training_inputs.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
    let x1_max = training_inputs.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
    let x2_min = training_inputs.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    let x2_max = training_inputs.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);

    // Add some padding to the bounds
    let padding = 0.5;
    let x1_range = (x1_min - padding)..(x1_max + padding);
    let x2_range = (x2_min - padding)..(x2_max + padding);

    let mut chart = ChartBuilder::on(&main_area)
        .caption(
            "2D Function Surface",
            ("sans-serif", 30).into_font(),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x1_range, x2_range)?;

    chart
        .configure_mesh()
        .x_desc("x₁")
        .y_desc("x₂")
        .draw()?;

    // Create a color gradient for the scatter points based on their values
    let y_min = training_targets.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = training_targets.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    // Create color gradient
    let color_gradient = colorous::VIRIDIS;

    // Plot training points as colored scatter points
    for (input, &target) in training_inputs.iter().zip(training_targets.iter()) {
        let normalized_value = (target - y_min) / (y_max - y_min);
        let color = color_gradient.eval_continuous(normalized_value);
        let rgb_color = RGBColor(color.r, color.g, color.b);

        chart.draw_series(PointSeries::of_element(
            vec![(input[0], input[1])],
            7,
            &rgb_color,
            &|c, s, st| {
                return EmptyElement::at(c)
                    + Circle::new((0,0), s, st.filled())
                    + Text::new(
                        format!("{:.1}", target),
                        (10, -10),
                        ("sans-serif", 15).into_font(),
                    );
            },
        ))?;
    }

    // Create color scale legend
    let mut legend_chart = ChartBuilder::on(&legend_area)
        .margin(5)
        .x_label_area_size(0)
        .y_label_area_size(30)
        .build_cartesian_2d(0f64..1f64, y_min..y_max)?;

    let n_colors = 20;
    for i in 0..n_colors {
        let y_coord = y_min + (y_max - y_min) * (i as f64 / n_colors as f64);
        let color = color_gradient.eval_continuous(i as f64 / n_colors as f64);
        let rgb_color = RGBColor(color.r, color.g, color.b);

        legend_chart.draw_series(std::iter::once(Rectangle::new(
            [(0f64, y_coord), (1f64, y_coord + (y_max - y_min) / n_colors as f64)],
            rgb_color.filled(),
        )))?;
    }

    legend_chart
        .configure_mesh()
        .y_desc("Value")
        .draw()?;

    root.present()?;
    println!("Plot has been saved as '{}'", filename);

    Ok(())
}