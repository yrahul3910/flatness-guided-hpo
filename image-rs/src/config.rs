use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub n_filters: i64,
    pub kernel_size: i64,
    pub padding: Padding,
    pub n_blocks: i64,
    pub dropout_rate: f64,
    pub final_dropout_rate: f64,
    pub n_units: i64,
    pub learning_rate: f64,
    pub weight_decay: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Padding {
    Valid,
    Same,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_filters: 32,
            kernel_size: 3,
            padding: Padding::Same,
            n_blocks: 2,
            dropout_rate: 0.2,
            final_dropout_rate: 0.4,
            n_units: 128,
            learning_rate: 1e-3,
            weight_decay: 1e-4,
        }
    }
}

pub struct HpoSpace {
    pub n_filters: (i64, i64),
    pub kernel_size: (i64, i64),
    pub padding: Vec<Padding>,
    pub n_blocks: (i64, i64),
    pub dropout_rate: (f64, f64),
    pub final_dropout_rate: (f64, f64),
    pub n_units: Vec<i64>,
    pub learning_rate: (f64, f64),
    pub weight_decay: (f64, f64),
}

impl Default for HpoSpace {
    fn default() -> Self {
        Self {
            n_filters: (32, 128),
            kernel_size: (2, 6),
            padding: vec![Padding::Valid, Padding::Same],
            n_blocks: (2, 5),
            dropout_rate: (0.2, 0.6),
            final_dropout_rate: (0.3, 0.7),
            n_units: vec![32, 64, 128, 256, 512],
            learning_rate: (1e-4, 1e-2),
            weight_decay: (1e-5, 1e-1),
        }
    }
}

impl HpoSpace {
    pub fn sample(&self) -> Config {
        let mut rng = rand::thread_rng();

        Config {
            n_filters: rng.gen_range(self.n_filters.0..=self.n_filters.1),
            kernel_size: rng.gen_range(self.kernel_size.0..=self.kernel_size.1),
            padding: self.padding[rng.gen_range(0..self.padding.len())].clone(),
            n_blocks: rng.gen_range(self.n_blocks.0..=self.n_blocks.1),
            dropout_rate: rng.gen_range(self.dropout_rate.0..self.dropout_rate.1),
            final_dropout_rate: rng.gen_range(self.final_dropout_rate.0..self.final_dropout_rate.1),
            n_units: self.n_units[rng.gen_range(0..self.n_units.len())],
            learning_rate: rng.gen_range(self.learning_rate.0..self.learning_rate.1),
            weight_decay: rng.gen_range(self.weight_decay.0..self.weight_decay.1),
        }
    }
}
