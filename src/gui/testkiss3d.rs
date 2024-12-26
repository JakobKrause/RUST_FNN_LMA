use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use rkl::benchmark::functions::Multimodal1D;
use rkl::prelude::*;
use std::sync::{Arc, Mutex};

struct NeuralNetApp {
    model: Option<Sequential<Dense>>,
    training_data: Option<(Array2<f64>, Array2<f64>)>,
    current_epoch: usize,
    errors: Vec<f64>,
    predictions: Option<Array2<f64>>,
    is_training: bool,
    plot_data: Arc<Mutex<PlotData>>,
}

struct PlotData {
    x: Vec<f64>,
    y: Vec<f64>,
    predictions: Vec<f64>,
    errors: Vec<f64>,
    mus: Vec<f64>,
    gradients: Vec<f64>,
}

impl Default for NeuralNetApp {
    fn default() -> Self {
        Self {
            model: None,
            training_data: None,
            current_epoch: 0,
            errors: Vec::new(),
            predictions: None,
            is_training: false,
            plot_data: Arc::new(Mutex::new(PlotData {
                x: Vec::new(),
                y: Vec::new(),
                predictions: Vec::new(),
                errors: Vec::new(),
                mus: Vec::new(),
                gradients: Vec::new(),
            })),
        }
    }
}

impl eframe::App for NeuralNetApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if ui.button("Initialize Model").clicked() && self.model.is_none() {
                self.initialize_model();
            }

            if let Some(model) = &mut self.model {
                if ui.button(if self.is_training { "Stop Training" } else { "Start Training" })
                    .clicked()
                {
                    self.is_training = !self.is_training;
                }

                if self.is_training {
                    if let Some((x, y)) = &self.training_data {
                        // Perform one training epoch
                        if let Ok(()) = model.fit(x.clone(), y.clone(), 1, true) {
                            self.current_epoch += 1;
                            
                            // Clone necessary data before update_plots
                            if let Ok(pred) = model.predict(x.clone()) {
                                self.predictions = Some(pred);
                                self.errors = model.errors.clone();
                            }
                        }
                    }
                }

                // Display current epoch and error
                ui.label(format!("Epoch: {}", self.current_epoch));
                if let Some(last_error) = self.errors.last() {
                    ui.label(format!("Current Error: {:.6}", last_error));
                }

                // Plot area
                self.draw_plots(ui);
            }
        });

        // Request continuous repaint while training
        if self.is_training {
            ctx.request_repaint();
        }
    }
}

impl NeuralNetApp {
    fn initialize_model(&mut self) {
        let x_vec: Vec<f64> = (1..=1000).map(|x| 0.001 * x as f64).collect();
        let y_vec: Vec<f64> = x_vec.multimodal1_d();

        let x = Array2::from_shape_vec((x_vec.len(), 1), x_vec.clone()).unwrap();
        let y = Array2::from_shape_vec((y_vec.len(), 1), y_vec.clone()).unwrap();

        let model = Sequential::builder()
            .add_dense(1, 20, Activation::Sigmoid)
            .unwrap()
            .add_dense(20, 20, Activation::Tanh)
            .unwrap()
            .add_dense(20, 10, Activation::Sigmoid)
            .unwrap()
            .add_dense(10, 1, Activation::Linear)
            .unwrap()
            .optimizer(OptimizerType::Marquardt {
                mu: 100.,
                mu_increase: 10.,
                mu_decrease: 0.1,
                min_error: 1e-10,
            })
            .loss(Loss::MSE)
            .build()
            .unwrap();

        self.model = Some(model);
        self.training_data = Some((x, y));
    }

    fn draw_plots(&self, ui: &mut egui::Ui) {
        // Plot comparison (only when not training)
        if !self.is_training {
            if let Some((x, _y)) = &self.training_data {
                Plot::new("comparison_plot")
                    .view_aspect(2.0)
                    .show(ui, |plot_ui| {
                        if let Some((x_data, y_data)) = &self.training_data {
                            let points: PlotPoints = x_data
                                .as_slice()
                                .unwrap()
                                .iter()
                                .zip(y_data.as_slice().unwrap().iter())
                                .map(|(&x, &y)| [x, y])
                                .collect();
                            
                            plot_ui.line(Line::new(points).name("Target"));
                        }

                        if let Some(predictions) = &self.predictions {
                            let pred_points: PlotPoints = x
                                .as_slice()
                                .unwrap()
                                .iter()
                                .zip(predictions.as_slice().unwrap().iter())
                                .map(|(&x, &y)| [x, y])
                                .collect();
                            
                            plot_ui.line(Line::new(pred_points).name("Prediction"));
                        }
                    });
            }
        }

        // Plot errors over epochs (during training)
        if self.is_training && !self.errors.is_empty() {
            Plot::new("error_plot")
                .view_aspect(2.0)
                .show(ui, |plot_ui| {
                    let points: PlotPoints = self
                        .errors
                        .iter()
                        .enumerate()
                        .map(|x| [x.0 as f64, *x.1])
                        .collect();
                    
                    plot_ui.line(Line::new(points).name("Error"));
                });
        }
    }
}

fn main() -> rkl::error::Result<()> {
    let options = eframe::NativeOptions::default();
    
    eframe::run_native(
        "Neural Network Training Visualizer",
        options,
        Box::new(|_cc| Ok(Box::new(NeuralNetApp::default()))),
    ).unwrap();

    Ok(())
}