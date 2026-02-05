use tch::{vision, Kind, Tensor};

use crate::error::Result;

pub struct Dataset {
    pub x_train: Tensor,
    pub y_train: Tensor,
    pub x_test: Tensor,
    pub y_test: Tensor,
    pub n_classes: i64,
}

pub fn load_cifar10() -> Result<Dataset> {
    let dataset = vision::cifar::load_dir("data")?;

    // Normalize to [0, 1]
    let x_train = dataset.train_images.to_kind(Kind::Float) / 255.0;
    let x_test = dataset.test_images.to_kind(Kind::Float) / 255.0;

    // Labels are already in the correct format [0, 9]
    let y_train = dataset.train_labels;
    let y_test = dataset.test_labels;

    Ok(Dataset {
        x_train,
        y_train,
        x_test,
        y_test,
        n_classes: 10,
    })
}

#[allow(dead_code)]
pub fn load_mnist() -> Result<Dataset> {
    let dataset = vision::mnist::load_dir("data")?;

    // Normalize to [0, 1]
    let x_train = dataset.train_images.to_kind(Kind::Float) / 255.0;
    let x_test = dataset.test_images.to_kind(Kind::Float) / 255.0;

    let y_train = dataset.train_labels;
    let y_test = dataset.test_labels;

    Ok(Dataset {
        x_train,
        y_train,
        x_test,
        y_test,
        n_classes: 10,
    })
}

impl Dataset {
    pub fn to_categorical(&self) -> Result<(Tensor, Tensor)> {
        // Convert labels to one-hot encoding
        let y_train_cat = self.y_train.onehot(self.n_classes);
        let y_test_cat = self.y_test.onehot(self.n_classes);
        Ok((y_train_cat, y_test_cat))
    }
}
