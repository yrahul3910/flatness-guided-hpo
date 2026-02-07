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
            n_filters: 3,
            kernel_size: 3,
            padding: Padding::Valid,
            n_blocks: 2,
            dropout_rate: 0.2,
            final_dropout_rate: 0.4,
            n_units: 128,
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
}

impl Default for HpoSpace {
    fn default() -> Self {
        Self {
            n_filters: (32, 128),
            kernel_size: (2, 6),
            padding: vec![Padding::Valid, Padding::Same],
            n_blocks: (2, 6),
            dropout_rate: (0.2, 0.5),
            final_dropout_rate: (0.2, 0.5),
            n_units: vec![32, 64, 128, 256, 512],
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
        }
    }
}
