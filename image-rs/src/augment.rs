use candle_core::Tensor;
use rand::Rng;

use crate::error::Result;

/// Apply standard CIFAR-10 augmentations to a batch of images.
///
/// Per sample, independently:
/// - Random horizontal flip with probability 0.5
/// - Random crop: pad 4 pixels on each side, then take a random 32×32 crop
///
/// Input shape: [N, C, H, W] with H=W=32, values in [0, 1].
pub fn augment_batch(x: &Tensor) -> Result<Tensor> {
    let n = x.dims()[0];
    let h = x.dims()[2];
    let w = x.dims()[3];
    let mut rng = rand::thread_rng();

    // Precompute reversed width indices for horizontal flip
    let flip_indices: Vec<u32> = (0..w as u32).rev().collect();
    let flip_idx = Tensor::from_vec(flip_indices, w, x.device())?;

    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let mut sample = x.narrow(0, i, 1)?;

        // Random horizontal flip
        if rng.gen::<bool>() {
            sample = sample.index_select(&flip_idx, 3)?;
        }

        // Random crop: pad by 4, then take a random 32×32 crop
        let pad = 4usize;
        let padded = sample
            .pad_with_zeros(2, pad, pad)?
            .pad_with_zeros(3, pad, pad)?;
        let top = rng.gen_range(0..2 * pad);
        let left = rng.gen_range(0..2 * pad);
        sample = padded.narrow(2, top, h)?.narrow(3, left, w)?;

        samples.push(sample);
    }

    Ok(Tensor::cat(&samples, 0)?)
}
