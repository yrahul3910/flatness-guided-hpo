mod config;
mod data;
mod error;
mod model;
mod util;

use candle_core::Device;
use serde_json::json;

use config::HpoSpace;
use data::load_cifar10;
use error::Result;
use util::{get_convexity, run_experiment};

const KEEP_CONFIGS: usize = 5;
const NUM_CONFIGS: usize = 30;

fn main() -> Result<()> {
    let device = if candle_core::utils::metal_is_available() {
        Device::new_metal(0)?
    } else {
        println!("Metal not available, falling back to CPU");
        Device::Cpu
    };
    println!("Using device: {:?}", device);

    let dataset = load_cifar10(&device)?;
    println!(
        "Dataset loaded: {} training samples",
        dataset.x_train.dims()[0]
    );

    let hpo_space = HpoSpace::default();

    let mut best_configs: Vec<config::Config> = Vec::new();
    let mut best_betas: Vec<f64> = Vec::new();

    let mut i = 0;
    while i < NUM_CONFIGS {
        let config = hpo_space.sample();
        println!("\nConfig {}/{}", i + 1, NUM_CONFIGS);
        println!("{:#?}", &config);

        match get_convexity(&dataset, &config, &device) {
            Ok(convexity) => {
                println!("  Convexity: {:.6}", convexity);

                // Keep top KEEP_CONFIGS configurations
                if best_betas.len() < KEEP_CONFIGS
                    || convexity
                        < *best_betas
                            .iter()
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap()
                {
                    best_betas.push(convexity);
                    best_configs.push(config);

                    // Sort and keep only top KEEP_CONFIGS
                    let mut pairs: Vec<(f64, config::Config)> = best_betas
                        .iter()
                        .copied()
                        .zip(best_configs.iter().cloned())
                        .collect();
                    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                    if pairs.len() > KEEP_CONFIGS {
                        pairs.truncate(KEEP_CONFIGS);
                    }

                    best_betas = pairs.iter().map(|(beta, _)| *beta).collect();
                    best_configs = pairs.iter().map(|(_, cfg)| cfg.clone()).collect();
                }
                i += 1;
            }
            Err(_) => {
                eprintln!("  Skipping failed config.");
            }
        }
    }

    // Train best configurations
    println!("\n\nTraining {} best configurations...", best_configs.len());

    for (beta, config) in best_betas.iter().zip(best_configs.iter()) {
        println!("\n{}", "=".repeat(60));
        println!("Training config with beta = {:.6}", beta);
        println!("Config: {:?}", config);

        match run_experiment(&dataset, config, &device) {
            Ok(accuracy) => {
                let result = json!({
                    "beta": beta,
                    "config": config,
                    "accuracy": accuracy
                });
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            Err(e) => {
                eprintln!("Error running experiment: {}", e);
            }
        }
    }

    Ok(())
}
