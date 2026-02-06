use candle_core::{Device, Tensor};
use std::fs;
use std::path::Path;

use crate::error::Result;

pub struct Dataset {
    pub x_train: Tensor,
    pub y_train: Tensor,
    pub x_test: Tensor,
    pub y_test: Tensor,
    pub n_classes: i64,
}

/// Load a single CIFAR-10 binary batch file.
/// Each record: 1 byte label + 3072 bytes (32*32*3) pixel data.
/// Pixel layout: 1024 R, 1024 G, 1024 B (channel-first, row-major within channel).
fn load_batch(path: &Path, device: &Device) -> Result<(Tensor, Tensor)> {
    let data = fs::read(path)?;
    let record_size = 1 + 3072; // 1 label byte + 3072 pixel bytes
    let n_samples = data.len() / record_size;

    let mut labels: Vec<u32> = Vec::with_capacity(n_samples);
    let mut pixels: Vec<f32> = Vec::with_capacity(n_samples * 3072);

    for i in 0..n_samples {
        let offset = i * record_size;
        labels.push(data[offset] as u32);
        for j in 1..=3072 {
            pixels.push(data[offset + j] as f32 / 255.0);
        }
    }

    let x = Tensor::from_vec(pixels, (n_samples, 3, 32, 32), device)?;
    let y = Tensor::from_vec(labels, n_samples, device)?;

    Ok((x, y))
}

pub fn load_cifar10(device: &Device) -> Result<Dataset> {
    // Load training batches
    let mut all_x_train: Vec<Tensor> = Vec::new();
    let mut all_y_train: Vec<Tensor> = Vec::new();

    for i in 1..=5 {
        let path = Path::new("data").join(format!("data_batch_{}.bin", i));
        let (x, y) = load_batch(&path, device)?;
        all_x_train.push(x);
        all_y_train.push(y);
    }

    let x_train = Tensor::cat(&all_x_train, 0)?;
    let y_train = Tensor::cat(&all_y_train, 0)?;

    // Load test batch
    let test_path = Path::new("data").join("test_batch.bin");
    let (x_test, y_test) = load_batch(&test_path, device)?;

    Ok(Dataset {
        x_train,
        y_train,
        x_test,
        y_test,
        n_classes: 10,
    })
}
