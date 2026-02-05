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

impl Config {
    /// Validate that this configuration will produce valid feature map sizes
    /// Returns true if the configuration is valid for CIFAR-10 (32x32 input)
    pub fn is_valid(&self) -> bool {
        let mut height = 32i64;
        let mut width = 32i64;

        for _ in 0..self.n_blocks {
            // For Valid padding, we need kernel_size <= dimension before conv
            // For Same padding with even kernel sizes, we need:
            // padded_size = input_size + 2*padding = input_size + 2*((kernel_size-1)/2)
            // For the conv to work, we need padded_size >= kernel_size

            match self.padding {
                Padding::Valid => {
                    // For Valid padding, input must be >= kernel_size
                    if height < self.kernel_size || width < self.kernel_size {
                        return false;
                    }
                    // After first convolution
                    height -= self.kernel_size - 1;
                    width -= self.kernel_size - 1;

                    if height <= 0
                        || width <= 0
                        || height < self.kernel_size
                        || width < self.kernel_size
                    {
                        return false;
                    }
                    // After second convolution
                    height -= self.kernel_size - 1;
                    width -= self.kernel_size - 1;
                }
                Padding::Same => {
                    // For Same padding, check if padding will be sufficient
                    // padding = (kernel_size - 1) / 2 (integer division)
                    let padding = (self.kernel_size - 1) / 2;
                    let min_input_size = self.kernel_size - 2 * padding;

                    if height < min_input_size || width < min_input_size {
                        return false;
                    }

                    // With Same padding and stride=1, output size = input size
                    // (approximately, PyTorch rounds)
                    // Be conservative and require input >= kernel_size - 1
                    if height < self.kernel_size - 1 || width < self.kernel_size - 1 {
                        return false;
                    }
                    // Spatial dimensions stay roughly the same with Same padding
                }
            }

            // Check dimensions are still positive
            if height <= 0 || width <= 0 {
                return false;
            }

            // After pooling (2x2)
            if height < 2 || width < 2 {
                return false;
            }
            height /= 2;
            width /= 2;

            // Must have at least 1x1 after pooling
            if height <= 0 || width <= 0 {
                return false;
            }
        }

        // Final check: ensure there's enough spatial resolution
        if height < 1 || width < 1 {
            return false;
        }

        true
    }
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
            n_filters: (2, 6),
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

        // Keep sampling until we get a valid configuration
        loop {
            let config = Config {
                n_filters: rng.gen_range(self.n_filters.0..=self.n_filters.1),
                kernel_size: rng.gen_range(self.kernel_size.0..=self.kernel_size.1),
                padding: self.padding[rng.gen_range(0..self.padding.len())].clone(),
                n_blocks: rng.gen_range(self.n_blocks.0..=self.n_blocks.1),
                dropout_rate: rng.gen_range(self.dropout_rate.0..self.dropout_rate.1),
                final_dropout_rate: rng
                    .gen_range(self.final_dropout_rate.0..self.final_dropout_rate.1),
                n_units: self.n_units[rng.gen_range(0..self.n_units.len())],
            };

            if config.is_valid() {
                return config;
            }
        }
    }

    pub fn sample_many(&self, n: usize) -> Vec<Config> {
        (0..n).map(|_| self.sample()).collect()
    }
}
