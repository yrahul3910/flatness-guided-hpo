use indicatif::{ProgressBar, ProgressStyle};
use tch::nn::{self, ConvConfig, Module, Sequential};
use tch::{Kind, Tensor};

use crate::config::{Config, Padding};
use crate::error::Result;

pub struct CifarModel {
    conv_blocks: Sequential,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl CifarModel {
    pub fn new(vs: &nn::Path, config: &Config, n_classes: i64) -> Self {
        let mut conv_blocks = nn::seq();

        // Build convolutional blocks
        let mut in_channels = 3; // RGB
        for i in 0..config.n_blocks {
            let out_channels = config.n_filters * 2_i64.pow(i as u32);

            // Determine padding based on config
            let padding = match config.padding {
                Padding::Same => {
                    // For "same" padding, we need to calculate padding to maintain spatial dimensions
                    (config.kernel_size - 1) / 2
                }
                Padding::Valid => 0,
            };

            let conv_cfg = ConvConfig {
                padding,
                ..Default::default()
            };

            // First conv in block
            conv_blocks = conv_blocks
                .add(nn::conv2d(
                    vs / format!("conv_{}a", i),
                    in_channels,
                    out_channels,
                    config.kernel_size,
                    conv_cfg,
                ))
                .add_fn(|x| x.relu());

            // Second conv in block
            conv_blocks = conv_blocks
                .add(nn::conv2d(
                    vs / format!("conv_{}b", i),
                    out_channels,
                    out_channels,
                    config.kernel_size,
                    conv_cfg,
                ))
                .add_fn(|x| x.max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false));

            in_channels = out_channels;
        }

        // Determine flattened size dynamically with a dummy forward pass
        let flattened_size = tch::no_grad(|| {
            // Create a dummy input (batch_size=1, channels=3, height=32, width=32)
            let dummy_input = Tensor::zeros(&[1, 3, 32, 32], (Kind::Float, vs.device()));
            let dummy_output = conv_blocks.forward(&dummy_input);
            let flat = dummy_output.flatten(1, -1);
            flat.size()[1]
        });

        // Fully connected layers
        let fc1 = nn::linear(
            vs / "fc1",
            flattened_size,
            config.n_units,
            Default::default(),
        );
        let fc2 = nn::linear(vs / "fc2", config.n_units, n_classes, Default::default());

        Self {
            conv_blocks,
            fc1,
            fc2,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.conv_blocks.forward(x);
        let x = x.flatten(1, -1);
        let x = self.fc1.forward(&x).relu();
        self.fc2.forward(&x)
    }

    /// Forward pass returning intermediate activations
    /// Returns: (final_output, penultimate_activation, last_layer_activation)
    pub fn forward_with_activations(&self, x: &Tensor) -> (Tensor, Tensor, Tensor) {
        // Get conv output
        let conv_out = self.conv_blocks.forward(x);
        let flattened = conv_out.flatten(1, -1);

        // Get penultimate layer (output of conv blocks, flattened)
        let penultimate = flattened.shallow_clone();

        // Get fc1 output (last hidden layer before final classification)
        let fc1_out = self.fc1.forward(&flattened).relu();
        let last_layer = fc1_out.shallow_clone();

        // Get final output
        let final_out = self.fc2.forward(&fc1_out);

        (final_out, penultimate, last_layer)
    }

    /// Get the weight matrix of the final layer
    pub fn get_final_layer_weights(&self) -> Tensor {
        self.fc2.ws.shallow_clone()
    }
}

fn calculate_flattened_size(config: &Config) -> i64 {
    // Calculate the size after convolutions and pooling
    // Starting with 32x32 for CIFAR-10
    let mut height = 32;
    let mut width = 32;

    for _i in 0..config.n_blocks {
        // After convolution
        match config.padding {
            Padding::Valid => {
                height -= config.kernel_size - 1;
                width -= config.kernel_size - 1;
                // Second conv
                height -= config.kernel_size - 1;
                width -= config.kernel_size - 1;
            }
            Padding::Same => {
                // Spatial dimensions stay the same
            }
        }

        // After pooling (2x2)
        height /= 2;
        width /= 2;
    }

    let last_channels = config.n_filters * 2_i64.pow((config.n_blocks - 1) as u32);
    height * width * last_channels
}

pub fn train_one_epoch(
    model: &CifarModel,
    optimizer: &mut nn::Optimizer,
    x_train: &Tensor,
    y_train: &Tensor,
    batch_size: i64,
) -> Result<f64> {
    train_one_epoch_with_pb(model, optimizer, x_train, y_train, batch_size, None)
}

pub fn train_one_epoch_with_pb(
    model: &CifarModel,
    optimizer: &mut nn::Optimizer,
    x_train: &Tensor,
    y_train: &Tensor,
    batch_size: i64,
    pb: Option<&ProgressBar>,
) -> Result<f64> {
    let n_samples = x_train.size()[0];
    let mut total_loss = 0.0;
    let mut n_batches = 0;

    for i in (0..n_samples).step_by(batch_size as usize) {
        let end = (i + batch_size).min(n_samples);
        let x_batch = x_train.narrow(0, i, end - i);
        let y_batch = y_train.narrow(0, i, end - i);

        let logits = model.forward(&x_batch);
        let loss = logits.cross_entropy_for_logits(&y_batch);

        optimizer.backward_step(&loss);

        total_loss += f64::try_from(loss)?;
        n_batches += 1;

        if let Some(pb) = pb {
            let avg_loss = total_loss / n_batches as f64;
            pb.set_position(n_batches as u64);
            pb.set_message(format!("loss: {:.4}", avg_loss));
        }
    }

    Ok(total_loss / n_batches as f64)
}

pub fn evaluate_accuracy(
    model: &CifarModel,
    x_test: &Tensor,
    y_test: &Tensor,
    batch_size: i64,
) -> Result<f64> {
    let n_samples = x_test.size()[0];
    let mut correct = 0i64;
    let mut total = 0i64;

    tch::no_grad(|| {
        for i in (0..n_samples).step_by(batch_size as usize) {
            let end = (i + batch_size).min(n_samples);
            let x_batch = x_test.narrow(0, i, end - i);
            let y_batch = y_test.narrow(0, i, end - i);

            let logits = model.forward(&x_batch);
            let predictions = logits.argmax(-1, false);

            let batch_correct = predictions
                .eq_tensor(&y_batch)
                .to_kind(Kind::Float)
                .sum(Kind::Float);
            if let Ok(count) = i64::try_from(batch_correct) {
                correct += count;
            }
            total += end - i;
        }
    });

    Ok(correct as f64 / total as f64)
}

pub fn train_with_early_stopping(
    model: &CifarModel,
    optimizer: &mut nn::Optimizer,
    x_train: &Tensor,
    y_train: &Tensor,
    batch_size: i64,
    max_epochs: i64,
    patience: i64,
) -> Result<()> {
    let n_samples = x_train.size()[0];
    let val_size = (n_samples as f64 * 0.2) as i64;
    let train_size = n_samples - val_size;

    let x_train_split = x_train.narrow(0, 0, train_size);
    let y_train_split = y_train.narrow(0, 0, train_size);
    let x_val = x_train.narrow(0, train_size, val_size);
    let y_val = y_train.narrow(0, train_size, val_size);

    let mut best_val_loss = f64::INFINITY;
    let mut patience_counter = 0;

    let n_train_batches = (train_size + batch_size - 1) / batch_size;

    for epoch in 0..max_epochs {
        // Create Keras-style progress bar for this epoch
        let pb = ProgressBar::new(n_train_batches as u64);
        pb.set_style(
            ProgressStyle::with_template(&format!(
                "Epoch {}/{} {{bar:30}} {{pos}}/{{len}} - {{msg}}",
                epoch + 1,
                max_epochs,
            ))
            .unwrap()
            .progress_chars("=> "),
        );

        // Train with progress bar
        let train_loss = train_one_epoch_with_pb(
            model,
            optimizer,
            &x_train_split,
            &y_train_split,
            batch_size,
            Some(&pb),
        )?;

        // Validate
        let val_loss = tch::no_grad(|| {
            let mut total_loss = 0.0;
            let mut n_batches = 0;

            for i in (0..val_size).step_by(batch_size as usize) {
                let end = (i + batch_size).min(val_size);
                let x_batch = x_val.narrow(0, i, end - i);
                let y_batch = y_val.narrow(0, i, end - i);

                let logits = model.forward(&x_batch);
                let loss = logits.cross_entropy_for_logits(&y_batch);
                total_loss += f64::try_from(loss).unwrap_or(0.0);
                n_batches += 1;
            }

            total_loss / n_batches as f64
        });

        pb.finish_with_message(format!(
            "loss: {:.4} - val_loss: {:.4}",
            train_loss, val_loss
        ));

        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= patience {
                println!("Early stopping at epoch {}", epoch + 1);
                break;
            }
        }
    }

    Ok(())
}
