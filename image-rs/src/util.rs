use std::panic;
use tch::nn::{self, OptimizerConfig};

use crate::config::Config;
use crate::data::Dataset;
use crate::error::{Error, Result};
use crate::model::CifarModel;

const BATCH_SIZE: i64 = 64;

/// Compute the convexity/smoothness metric for a model configuration
///
/// The metric is: mu = ||Ka(x)|| * ||Ka-1(x)|| / ||W||
/// where:
/// - Ka(x) is the activation of the last hidden layer (fc1 output)
/// - Ka-1(x) is the activation of the second-to-last layer (conv output, flattened)
/// - W is the weight matrix of the final classification layer
///
/// Lower values indicate "flatter" loss landscape / smoother networks
pub fn get_convexity(dataset: &Dataset, config: &Config, device: tch::Device) -> Result<f64> {
    // Create model - catch panics from invalid configurations
    let vs = nn::VarStore::new(device);
    let model = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        CifarModel::new(&vs.root(), config, dataset.n_classes)
    }))
    .map_err(|_| Error::Msg("Failed to create model - invalid configuration".into()))?;
    let mut optimizer = nn::Adam::default().build(&vs, 1e-3)?;

    // Train for one epoch to get a trained model
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        crate::model::train_one_epoch(
            &model,
            &mut optimizer,
            &dataset.x_train,
            &dataset.y_train,
            BATCH_SIZE,
        )
    }))
    .map_err(|_| Error::Msg("Panic during training".into()))??;

    // Get the weight matrix of the final layer
    let final_weights = model.get_final_layer_weights();
    let weight_norm = f64::try_from(final_weights.norm())?;

    // Compute convexity metric across all batches
    let mut best_mu = f64::NEG_INFINITY;
    let n_samples = dataset.x_train.size()[0];

    tch::no_grad(|| -> Result<()> {
        for i in (0..n_samples).step_by(BATCH_SIZE as usize) {
            let end = (i + BATCH_SIZE).min(n_samples);
            let x_batch = dataset.x_train.narrow(0, i, end - i);

            // Get intermediate activations
            // Returns: (final_output, penultimate_activation, last_layer_activation)
            let (_final_out, penultimate, last_layer) = model.forward_with_activations(&x_batch);

            // Compute norms
            // Ka = last hidden layer activation (fc1 output with ReLU)
            let ka_norm = f64::try_from(last_layer.norm())?;

            // Ka-1 = second-to-last layer activation (flattened conv output)
            let ka1_norm = f64::try_from(penultimate.norm())?;

            // Compute mu = ||Ka|| * ||Ka-1|| / ||W||
            let mu = (ka_norm * ka1_norm) / weight_norm;

            // Track the maximum mu value
            if mu > best_mu && !mu.is_infinite() && !mu.is_nan() {
                best_mu = mu;
            }
        }

        Ok(())
    })?;

    // Return the maximum mu value (higher mu = less convex/smooth)
    Ok(best_mu)
}

pub fn run_experiment(dataset: &Dataset, config: &Config, device: tch::Device) -> Result<f64> {
    // Create model - catch panics from invalid configurations
    let vs = nn::VarStore::new(device);
    let model = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        CifarModel::new(&vs.root(), config, dataset.n_classes)
    }))
    .map_err(|_| Error::Msg("Failed to create model - invalid configuration".into()))?;
    let mut optimizer = nn::Adam::default().build(&vs, 1e-3)?;

    // Train with early stopping
    // Use raw labels (class indices) not one-hot encoded
    // Catch panics from training
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
    .map_err(|_| Error::Msg("Panic during training".into()))??;

    // Evaluate
    let accuracy =
        crate::model::evaluate_accuracy(&model, &dataset.x_test, &dataset.y_test, BATCH_SIZE)?;

    Ok(accuracy)
}
