use eframe::egui; // for main UI
use egui_extras::plot::{Line, Plot, Value, Values}; // for plotting
use rkl::benchmark::functions::Multimodal1D;
use rkl::prelude::*;
use rkl::plot::*;
use ndarray::Array2;

#[derive(Debug, Clone, Copy, PartialEq)]
enum ActivationChoice {
    Sigmoid,
    Tanh,
    Relu,
    Linear,
}

impl ActivationChoice {
    fn to_activation(self) -> Activation {
        match self {
            ActivationChoice::Sigmoid => Activation::Sigmoid,
            ActivationChoice::Tanh => Activation::Tanh,
            ActivationChoice::Relu => Activation::Relu,
            ActivationChoice::Linear => Activation::Linear,
        }
    }
}

#[derive(Debug, Clone)]
struct LayerConfig {
    neurons: usize,
    activation: ActivationChoice,
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self {
            neurons: 20,
            activation: ActivationChoice::Sigmoid,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum OptimizerChoice {
    Marquardt,
    SGD,
}

impl OptimizerChoice {
    fn to_optimizer(self) -> OptimizerType {
        match self {
            OptimizerChoice::Marquardt => OptimizerType::Marquardt {
                mu: 100.,
                mu_increase: 10.,
                mu_decrease: 0.1,
                min_error: 1e-10,
            },
            OptimizerChoice::SGD => OptimizerType::SGD(0.5),
        }
    }
}

struct MyApp {
    // Training configuration
    hidden_layers: Vec<LayerConfig>,
    optimizer_choice: OptimizerChoice,
    epochs: usize,

    // UI state
    num_hidden_layers: usize,
    training_result: Option<String>,

    // Store errors for plotting (epoch vs error) every 5 epochs
    training_data: Vec<(f64, f64)>,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            hidden_layers: vec![LayerConfig::default(), LayerConfig::default()],
            optimizer_choice: OptimizerChoice::Marquardt,
            epochs: 1000,
            num_hidden_layers: 2,
            training_result: None,
            training_data: Vec::new(),
        }
    }
}

impl MyApp {
    fn train_model(&mut self) -> Result<()> {
        // Generate the dataset
        let x_vec: Vec<f64> = (1..=1000).map(|x| 0.001 * x as f64).collect();
        let y_vec: Vec<f64> = x_vec.multimodal1_d();

        let x = Array2::from_shape_vec((x_vec.len(), 1), x_vec.clone()).unwrap();
        let y = Array2::from_shape_vec((y_vec.len(), 1), y_vec.clone()).unwrap();

        // Build the model dynamically
        let mut builder = Sequential::builder();

        let mut input_size = 1;
        for layer in &self.hidden_layers {
            builder = builder.add_dense(input_size, layer.neurons, layer.activation.to_activation())?;
            input_size = layer.neurons;
        }

        builder = builder
            .add_dense(input_size, 1, Activation::Linear)?
            .optimizer(self.optimizer_choice.to_optimizer())
            .loss(Loss::MSE);

        let mut model = builder.build()?;
        model.summary();

        model.fit(x.clone(), y.clone(), self.epochs, true)?;

        // Evaluate model
        let mse = model.evaluate(x.clone(), y.clone());
        println!("Evaluation...");
        println!("MSE: {:?}", mse);

        // Save model
        model.save("./test.model")?;

        // Plot comparison
        let prediction = model.predict(x.clone())?;
        plot_comparision::plot_comparison(
            x.as_slice().unwrap(),
            y.as_slice().unwrap(),
            prediction.as_slice().unwrap(),
            "multimodal_comparison.png",
        ).unwrap();

        plot_errors_over_epochs::plot_errors_over_epochs(&model.errors, "error_over_epochs.png").unwrap();

        // Extract errors every 5 epochs
        self.training_data = model.errors.iter().enumerate()
            .filter_map(|(i, &err)| {
                if i % 5 == 0 {
                    Some((i as f64, err))
                } else {
                    None
                }
            }).collect();

        self.training_result = Some(format!("Training done! MSE: {:?}", mse));

        Ok(())
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Neural Network Training Configuration");

            // Number of hidden layers
            ui.horizontal(|ui| {
                ui.label("Number of hidden layers:");
                ui.add(egui::DragValue::new(&mut self.num_hidden_layers).range(0..=10));
            });

            if self.num_hidden_layers != self.hidden_layers.len() {
                self.hidden_layers.resize(self.num_hidden_layers, LayerConfig::default());
            }

            ui.separator();

            // Hidden layers config
            for (i, layer) in self.hidden_layers.iter_mut().enumerate() {
                ui.group(|ui| {
                    ui.label(format!("Hidden Layer {}", i+1));
                    ui.horizontal(|ui| {
                        ui.label("Neurons:");
                        ui.add(egui::DragValue::new(&mut layer.neurons).range(1..=1000));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Activation:");
                        egui::ComboBox::new(format!("activation_{}", i), "Activation")
                            .selected_text(format!("{:?}", layer.activation))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut layer.activation, ActivationChoice::Sigmoid, "Sigmoid");
                                ui.selectable_value(&mut layer.activation, ActivationChoice::Tanh, "Tanh");
                                ui.selectable_value(&mut layer.activation, ActivationChoice::Relu, "ReLU");
                                ui.selectable_value(&mut layer.activation, ActivationChoice::Linear, "Linear");
                            });
                    });
                });
            }

            ui.separator();
            // Optimizer
            ui.horizontal(|ui| {
                ui.label("Optimizer:");
                egui::ComboBox::new("optimizer_choice", "Optimizer")
                    .selected_text(format!("{:?}", self.optimizer_choice))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.optimizer_choice, OptimizerChoice::Marquardt, "Marquardt");
                        ui.selectable_value(&mut self.optimizer_choice, OptimizerChoice::SGD, "SGD");
                    });
            });

            ui.horizontal(|ui| {
                ui.label("Epochs:");
                ui.add(egui::DragValue::new(&mut self.epochs).range(1..=100000));
            });

            ui.separator();

            if ui.button("Train").clicked() {
                match self.train_model() {
                    Ok(_) => {}
                    Err(err) => {
                        self.training_result = Some(format!("Error during training: {:?}", err));
                    }
                }
            }

            if let Some(result) = &self.training_result {
                ui.label(result);
            }

            // If we have training data, plot it
            if !self.training_data.is_empty() {
                let line = Line::new(Values::from_values_iter(
                    self.training_data.iter().map(|(x, y)| Value::new(*x, *y))
                )).name("Error every 5 epochs");

                Plot::new("Error Plot").view_aspect(2.0).show(ui, |plot_ui| {
                    plot_ui.line(line);
                    plot_ui.xlabel("Epoch");
                    plot_ui.ylabel("Error");
                });
            }
        });
    }
}

fn main() -> Result<()> {
    let options = eframe::NativeOptions::default();
    let eframe_result = eframe::run_native(
        "NN Training GUI",
        options,
        Box::new(|_cc| Ok(Box::new(MyApp::default()))),
    );

    match eframe_result {
        Ok(result) => result,
        Err(error)=> panic!("Problem with the gui: {error:?}"),
    };

    Ok(())
}