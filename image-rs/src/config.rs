use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub n_filters: i64,
    pub kernel_size: i64,
    pub padding: Padding,
    pub n_blocks: i64,
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
            n_units: vec![32, 64, 128, 256, 512],
            learning_rate: (1e-4, 1e-2),
            weight_decay: (1e-5, 1e-3),
        }
    }
}

impl HpoSpace {
    /// Compute max valid n_blocks for Valid padding with a given kernel size,
    /// starting from a 32x32 input (two convs + maxpool per block).
    fn max_valid_blocks_for_kernel(kernel_size: i64, max_blocks: i64) -> i64 {
        let mut size = 32i64;
        let mut blocks = 0i64;
        loop {
            size -= (kernel_size - 1) * 2; // two valid-padded convs per block
            if size <= 0 {
                break;
            }
            size /= 2; // max pool
            if size <= 0 {
                break;
            }
            blocks += 1;
            if blocks >= max_blocks {
                break;
            }
        }
        blocks
    }

    pub fn sample(&self) -> Config {
        let mut rng = rand::thread_rng();

        let kernel_size = rng.gen_range(self.kernel_size.0..=self.kernel_size.1);
        let padding = self.padding[rng.gen_range(0..self.padding.len())].clone();

        // For Valid padding, cap n_blocks at the max that keeps the feature map alive.
        // Fall back to Same padding if no valid n_blocks count is possible.
        let (padding, n_blocks) = match padding {
            Padding::Valid => {
                let max_blocks =
                    Self::max_valid_blocks_for_kernel(kernel_size, self.n_blocks.1);
                if max_blocks < self.n_blocks.0 {
                    // This kernel/block combination is always invalid with Valid padding.
                    let n = rng.gen_range(self.n_blocks.0..=self.n_blocks.1);
                    (Padding::Same, n)
                } else {
                    let n = rng.gen_range(self.n_blocks.0..=max_blocks);
                    (Padding::Valid, n)
                }
            }
            Padding::Same => {
                let n = rng.gen_range(self.n_blocks.0..=self.n_blocks.1);
                (Padding::Same, n)
            }
        };

        // Log-uniform sampling for hyperparameters that span orders of magnitude.
        let learning_rate = {
            let lo = self.learning_rate.0.ln();
            let hi = self.learning_rate.1.ln();
            rng.gen_range(lo..hi).exp()
        };
        let weight_decay = {
            let lo = self.weight_decay.0.ln();
            let hi = self.weight_decay.1.ln();
            rng.gen_range(lo..hi).exp()
        };

        Config {
            n_filters: rng.gen_range(self.n_filters.0..=self.n_filters.1),
            kernel_size,
            padding,
            n_blocks,
            n_units: self.n_units[rng.gen_range(0..self.n_units.len())],
            learning_rate,
            weight_decay,
        }
    }
}
