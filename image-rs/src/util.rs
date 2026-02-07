use std::panic;

use candle_core::Device;
use candle_nn::{self as nn, VarBuilder, VarMap};
use indicatif::{ProgressBar, ProgressStyle};

use crate::config::Config;
use crate::data::Dataset;
use crate::error::{Error, Result};
use crate::model::CifarModel;

const BATCH_SIZE: usize = 64;

/// Compute the convexity/smoothness metric for a model configuration
///
/// The metric is: mu = ||Ka(x)|| * ||Ka-1(x)|| / ||W||
/// where:
/// - Ka(x) is the activation of the last hidden layer (fc1 output)
/// - Ka-1(x) is the activation of the second-to-last layer (conv output, flattened)
/// - W is the weight matrix of the final classification layer
///
/// Lower values indicate "flatter" loss landscape / smoother networks
pub fn get_convexity(dataset: &Dataset, config: &Config, device: &Device) -> Result<f64> {
    // Create model - catch panics from invalid configurations
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let model = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        CifarModel::new(vb, config, dataset.n_classes as usize)
    }))
    .map_err(|_| Error::Msg("Failed to create model - invalid configuration".into()))?
    .map_err(|e| Error::Candle(e))?;

    let mut optimizer = nn::AdamW::new_lr(varmap.all_vars(), 1e-3)?;

    // Train for one epoch to get a trained model
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let n_train_batches = &dataset.x_train.shape().dim(0)?.div_ceil(BATCH_SIZE);
        let pb = ProgressBar::new(*n_train_batches as u64);
        pb.set_style(
            ProgressStyle::with_template(&format!("|{{bar:30}}|",))
                .unwrap()
                .progress_chars("=> "),
        );

        crate::model::train_one_epoch_with_pb(
            &model,
            &mut optimizer,
            &dataset.x_train,
            &dataset.y_train,
            BATCH_SIZE,
            Some(&pb),
        )
    }))
    .map_err(|_| Error::Msg("Panic during training".into()))?
    .map_err(|e| Error::from(e))?;

    // Get the weight matrix of the final layer
    let final_weights = model.get_final_layer_weights();
    let weight_norm = final_weights.norm()?.to_scalar::<f32>()? as f64;

    // Compute convexity metric across all batches
    let mut best_mu = f64::NEG_INFINITY;
    let n_samples = dataset.x_train.dims()[0];

    let mut i = 0;
    while i < n_samples {
        let end = (i + BATCH_SIZE).min(n_samples);
        let len = end - i;
        let x_batch = dataset.x_train.narrow(0, i, len)?;

        // Get intermediate activations
        let (_final_out, penultimate, last_layer) = model.forward_with_activations(&x_batch)?;

        // Compute norms
        let ka_norm = last_layer.norm()?.to_scalar::<f32>()? as f64;
        let ka1_norm = penultimate.norm()?.to_scalar::<f32>()? as f64;

        // Compute mu = ||Ka|| * ||Ka-1|| / ||W||
        let mu = (ka_norm * ka1_norm) / weight_norm;

        // Track the maximum mu value
        if mu > best_mu && !mu.is_infinite() && !mu.is_nan() {
            best_mu = mu;
        }

        i = end;
    }

    // Return the maximum mu value (higher mu = less convex/smooth)
    Ok(best_mu)
}

pub fn run_experiment(dataset: &Dataset, config: &Config, device: &Device) -> Result<f64> {
    // Create model - catch panics from invalid configurations
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let model = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        CifarModel::new(vb, config, dataset.n_classes as usize)
    }))
    .map_err(|_| Error::Msg("Failed to create model - invalid configuration".into()))?
    .map_err(|e| Error::Candle(e))?;

    let mut optimizer = nn::AdamW::new_lr(varmap.all_vars(), 1e-3)?;

    // Train with early stopping
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        crate::model::train_with_early_stopping(
            &model,
            &mut optimizer,
            &dataset.x_train,
            &dataset.y_train,
            BATCH_SIZE,
            100, // max_epochs
            10,  // patience
        )
    }))
    .map_err(|_| Error::Msg("Panic during training".into()))?
    .map_err(|e| Error::from(e))?;

    // Evaluate
    let accuracy =
        crate::model::evaluate_accuracy(&model, &dataset.x_test, &dataset.y_test, BATCH_SIZE)?;

    Ok(accuracy)
}
