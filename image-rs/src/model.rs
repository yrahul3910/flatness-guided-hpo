use candle_core::{DType, Module, Tensor, D};
use candle_nn::{self as nn, conv2d, linear, loss, Optimizer, VarBuilder};
use indicatif::{ProgressBar, ProgressStyle};

use crate::config::{Config, Padding};
use crate::error::Result;

/// A single convolutional block: conv -> relu -> conv -> relu -> max_pool(2x2)
struct ConvBlock {
    conv_a: nn::Conv2d,
    conv_b: nn::Conv2d,
}

impl ConvBlock {
    fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
    ) -> candle_core::Result<Self> {
        let cfg = nn::Conv2dConfig {
            padding,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv_a = conv2d(in_channels, out_channels, kernel_size, cfg, vb.pp("a"))?;
        let conv_b = conv2d(out_channels, out_channels, kernel_size, cfg, vb.pp("b"))?;
        Ok(Self { conv_a, conv_b })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.conv_a.forward(x)?.relu()?;
        let x = self.conv_b.forward(&x)?.relu()?;
        // max_pool2d(kernel=2, stride=2)
        x.max_pool2d_with_stride(2, 2)
    }
}

pub struct CifarModel {
    conv_blocks: Vec<ConvBlock>,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl CifarModel {
    pub fn new(vb: VarBuilder, config: &Config, n_classes: usize) -> candle_core::Result<Self> {
        let padding = match config.padding {
            Padding::Same => ((config.kernel_size - 1) / 2) as usize,
            Padding::Valid => 0,
        };

        let mut conv_blocks = Vec::new();
        let mut in_channels: usize = 3; // RGB

        for i in 0..config.n_blocks {
            let out_channels = (config.n_filters * 2_i64.pow(i as u32)) as usize;
            let block = ConvBlock::new(
                vb.pp(format!("conv_{}", i)),
                in_channels,
                out_channels,
                config.kernel_size as usize,
                padding,
            )?;
            conv_blocks.push(block);
            in_channels = out_channels;
        }

        // Determine flattened size with a dummy forward pass
        let dummy = Tensor::zeros(&[1, 3, 32, 32], DType::F32, vb.device())?;
        let mut x = dummy;
        for block in &conv_blocks {
            x = block.forward(&x)?;
        }
        // Flatten from dim 1 to last dim, then read dim 1 size
        let flat = x.flatten(1usize, 3usize)?;
        let flattened_size = flat.dims()[1];

        let fc1 = linear(flattened_size, config.n_units as usize, vb.pp("fc1"))?;
        let fc2 = linear(config.n_units as usize, n_classes, vb.pp("fc2"))?;

        Ok(Self {
            conv_blocks,
            fc1,
            fc2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = x.clone();
        for block in &self.conv_blocks {
            x = block.forward(&x)?;
        }
        let x = x.flatten(1, 3)?;
        let x = self.fc1.forward(&x)?.relu()?;
        self.fc2.forward(&x)
    }

    /// Forward pass returning intermediate activations.
    /// Returns: (final_output, penultimate_activation, last_layer_activation)
    pub fn forward_with_activations(
        &self,
        x: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let mut x = x.clone();
        for block in &self.conv_blocks {
            x = block.forward(&x)?;
        }
        let flattened = x.flatten(1, 3)?;
        let penultimate = flattened.clone();

        let fc1_out = self.fc1.forward(&flattened)?.relu()?;
        let last_layer = fc1_out.clone();

        let final_out = self.fc2.forward(&fc1_out)?;

        Ok((final_out, penultimate, last_layer))
    }

    /// Get the weight matrix of the final layer (fc2).
    pub fn get_final_layer_weights(&self) -> &Tensor {
        self.fc2.weight()
    }
}

pub fn train_one_epoch(
    model: &CifarModel,
    optimizer: &mut nn::AdamW,
    x_train: &Tensor,
    y_train: &Tensor,
    batch_size: usize,
) -> Result<f64> {
    train_one_epoch_with_pb(model, optimizer, x_train, y_train, batch_size, None)
}

pub fn train_one_epoch_with_pb(
    model: &CifarModel,
    optimizer: &mut nn::AdamW,
    x_train: &Tensor,
    y_train: &Tensor,
    batch_size: usize,
    pb: Option<&ProgressBar>,
) -> Result<f64> {
    let n_samples = x_train.dims()[0];
    let mut total_loss = 0.0;
    let mut n_batches: u64 = 0;

    let mut i = 0;
    while i < n_samples {
        let end = (i + batch_size).min(n_samples);
        let len = end - i;
        let x_batch = x_train.narrow(0, i, len)?;
        let y_batch = y_train.narrow(0, i, len)?;

        let logits = model.forward(&x_batch)?;
        let ce_loss = loss::cross_entropy(&logits, &y_batch)?;

        optimizer.backward_step(&ce_loss)?;

        total_loss += ce_loss.to_scalar::<f32>()? as f64;
        n_batches += 1;

        if let Some(pb) = pb {
            let avg_loss = total_loss / n_batches as f64;
            pb.set_position(n_batches);
            pb.set_message(format!("loss: {:.4}", avg_loss));
        }

        i = end;
    }

    Ok(total_loss / n_batches as f64)
}

pub fn evaluate_accuracy(
    model: &CifarModel,
    x_test: &Tensor,
    y_test: &Tensor,
    batch_size: usize,
) -> Result<f64> {
    let n_samples = x_test.dims()[0];
    let mut correct: u64 = 0;
    let mut total: u64 = 0;

    let mut i = 0;
    while i < n_samples {
        let end = (i + batch_size).min(n_samples);
        let len = end - i;
        let x_batch = x_test.narrow(0, i, len)?;
        let y_batch = y_test.narrow(0, i, len)?;

        let logits = model.forward(&x_batch)?;
        let predictions = logits.argmax(D::Minus1)?.to_dtype(DType::U32)?;

        let matches = predictions.eq(&y_batch)?.to_dtype(DType::F32)?.sum_all()?;
        correct += matches.to_scalar::<f32>()? as u64;
        total += len as u64;

        i = end;
    }

    Ok(correct as f64 / total as f64)
}

pub fn train_with_early_stopping(
    model: &CifarModel,
    optimizer: &mut nn::AdamW,
    x_train: &Tensor,
    y_train: &Tensor,
    batch_size: usize,
    max_epochs: usize,
    patience: usize,
) -> Result<()> {
    let n_samples = x_train.dims()[0];
    let val_size = (n_samples as f64 * 0.2) as usize;
    let train_size = n_samples - val_size;

    let x_train_split = x_train.narrow(0, 0, train_size)?;
    let y_train_split = y_train.narrow(0, 0, train_size)?;
    let x_val = x_train.narrow(0, train_size, val_size)?;
    let y_val = y_train.narrow(0, train_size, val_size)?;

    let mut best_val_loss = f64::INFINITY;
    let mut patience_counter = 0usize;

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
        let mut val_total_loss = 0.0;
        let mut val_n_batches: u64 = 0;

        let mut vi = 0;
        while vi < val_size {
            let end = (vi + batch_size).min(val_size);
            let len = end - vi;
            let x_batch = x_val.narrow(0, vi, len)?;
            let y_batch = y_val.narrow(0, vi, len)?;

            let logits = model.forward(&x_batch)?;
            let ce_loss = loss::cross_entropy(&logits, &y_batch)?;
            val_total_loss += ce_loss.to_scalar::<f32>()? as f64;
            val_n_batches += 1;

            vi = end;
        }

        let val_loss = val_total_loss / val_n_batches as f64;

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
