import math
import random

import numpy as np
import scipy
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import ReLU
from keras.src.datasets.cifar10 import load_data as load_cifar10
from keras.src.datasets.mnist import load_data as load_mnist
from keras.src.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
)
from keras.src.models import Model, Sequential
from keras.src.optimizers import Adam
from keras.src.optimizers.schedules import CosineDecay
from keras.src.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

from image.src.config import Config, HpoOption, HpoSpace
from image.src.data import Dataset

BATCH_SIZE = 64


def get_mnist():
    return load_mnist()


def get_cifar10():
    return load_cifar10()


def get_svhn():
    train_data = scipy.io.loadmat("train_32x32.mat")
    x_train = train_data["X"]
    y_train = train_data["y"]
    test_data = scipy.io.loadmat("test_32x32.mat")
    x_test = test_data["X"]
    y_test = test_data["y"]

    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)

    return (x_train, y_train), (x_test, y_test)


def get_data(dataset: str = "svhn"):
    data_loaders = {
        "mnist": (get_mnist, (28, 28, 1)),
        "svhn": (get_svhn, (32, 32, 3)),
        "cifar10": (get_cifar10, (32, 32, 3)),
    }

    if dataset not in data_loaders:
        msg = "Invalid dataset name."
        raise ValueError(msg)

    data_loader, (img_rows, img_cols, img_channels) = data_loaders[dataset]
    (x_train, y_train), (x_test, y_test) = data_loader()

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    channel_stats = {
        "cifar10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        "svhn": ([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]),
        "mnist": ([0.1307], [0.3081]),
    }
    if dataset in channel_stats:
        mean, std = channel_stats[dataset]
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

    if K.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)

    """
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    """
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)
    return Dataset(x_train, y_train, x_test, y_test)


def run_experiment(
    data: Dataset, config: Config, n_class: int = 10, dataset: str = "mnist", epochs: int = 100
) -> float:
    n_train = int(len(data.x_train) * 0.8)
    decay_steps = epochs * math.ceil(n_train / BATCH_SIZE)
    model = get_model(data, config, n_class, dataset, decay_steps=decay_steps)
    if model is None:
        return 0.0

    model.fit(
        data.x_train,
        data.y_train,
        validation_split=0.20,
        epochs=epochs,
        verbose=1,
        batch_size=BATCH_SIZE,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
    )

    y_pred = np.argmax(model.predict(data.x_test), axis=-1)

    y_test_true = data.y_test
    if len(y_test_true.shape) > 1:
        y_test_true = np.argmax(y_test_true, axis=1)

    return accuracy_score(y_test_true, y_pred)


def get_convexity(
    data: Dataset,
    config: Config,
    n_class: int = 10,
    dataset: str = "mnist",
    subset_size: int = 1500,
) -> float:
    model = get_model(data, config, n_class, dataset)
    if model is None:
        return np.inf

    # Shuffle to remove class-sort bias (CIFAR-10 binary files are class-sorted)
    indices = np.random.permutation(len(data.x_train))[:subset_size]
    x_train_subset = data.x_train[indices]
    y_train_subset = data.y_train[indices]

    if n_class > 2 and len(y_train_subset.shape) == 1:  # noqa: PLR2004
        y_train_subset = np.array(to_categorical(y_train_subset, n_class))

    # 3 epochs: 1 epoch biases toward high-LR configs that appear smoother early
    model.fit(x_train_subset, y_train_subset, batch_size=BATCH_SIZE, epochs=1, verbose=0)

    # Precompute weight RMS (fixed per config, not per batch)
    w_rms = np.sqrt(np.mean(np.array(model.layers[-1].weights[0]) ** 2))

    # Submodels built once, not per-batch
    ka_model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    ka1_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    ka_outs = ka_model.predict(x_train_subset, batch_size=BATCH_SIZE, verbose=0)
    ka1_outs = ka1_model.predict(x_train_subset, batch_size=BATCH_SIZE, verbose=0)

    batch_size = BATCH_SIZE
    best_mu = -np.inf
    num_batches = (len(x_train_subset) - 1) // batch_size + 1
    for i in range(num_batches):
        start_i = i * batch_size
        end_i = start_i + batch_size
        ka_rms = np.sqrt(np.mean(ka_outs[start_i:end_i] ** 2))
        ka1_rms = np.sqrt(np.mean(ka1_outs[start_i:end_i] ** 2))
        mu = ka_rms * ka1_rms / w_rms
        if np.isfinite(mu) and mu > best_mu:
            best_mu = mu

    return best_mu if best_mu > 0 else np.inf


def _max_valid_blocks(kernel_size: int, img_size: int, n_convs_per_block: int) -> int:
    """Return the max n_blocks that keeps feature map positive with valid padding."""
    size = img_size
    blocks = 0
    while True:
        size -= (kernel_size - 1) * n_convs_per_block
        if size <= 0:
            break
        size //= 2  # max pool
        if size <= 0:
            break
        blocks += 1
    return blocks


def get_random_hyperparams(options: HpoSpace, img_size: int = 32, n_convs_per_block: int = 1) -> Config:
    """Get hyperparameters from options."""
    hyperparams: dict[str, HpoOption] = {}
    for key, value in options.items():
        if value is None:
            hyperparams[key] = None
        elif isinstance(value, list):
            hyperparams[key] = random.choice(value)
        elif isinstance(value[0], int):
            if key == "n_blocks" and hyperparams.get("padding") == "valid":
                k = int(hyperparams.get("kernel_size", 3))
                max_blocks = _max_valid_blocks(k, img_size, n_convs_per_block)
                lo, hi = value[0], min(value[1], max_blocks)
                if hi < lo:
                    # No valid n_blocks exists for this kernel+padding combo; fall back
                    hyperparams["padding"] = "same"
                    hyperparams[key] = random.randint(value[0], value[1])
                else:
                    hyperparams[key] = random.randint(lo, hi)
            else:
                hyperparams[key] = random.randint(value[0], value[1])
        else:
            lo, hi = value[0], value[1]
            hyperparams[key] = math.exp(random.uniform(math.log(lo), math.log(hi)))

    return Config(**hyperparams)


def get_many_random_hyperparams(options: HpoSpace, n: int) -> list[Config]:
    """Get n hyperparameters from options."""
    return [get_random_hyperparams(options) for _ in range(n)]


def get_model(
    data: Dataset, config: Config, n_classes: int = 10, dataset: str = "mnist", decay_steps: int | None = None
) -> Sequential | None:
    # Check if architecture is valid for image size
    # Precise simulation of spatial resolution
    curr_h = data.x_train.shape[1]
    curr_w = data.x_train.shape[2]

    k = config["kernel_size"]
    pad = config["padding"]

    for _ in range(config["n_blocks"]):
        if pad == "valid":
            # For CIFAR/SVHN, we have TWO convs per block (in svhn_model)
            # or ONE per block (in cifar10/mnist).
            # To be safe, let's assume the most aggressive reduction.
            if dataset == "svhn":
                curr_h -= (k - 1) * 2
                curr_w -= (k - 1) * 2
            else:
                curr_h -= k - 1
                curr_w -= k - 1

        if curr_h <= 0 or curr_w <= 0:
            return None

        # MaxPooling
        curr_h //= 2
        curr_w //= 2

        if curr_h <= 0 or curr_w <= 0:
            return None

    if dataset == "mnist":
        return get_mnist_model(config, n_classes, decay_steps=decay_steps)
    if dataset == "svhn":
        return get_svhn_model(config, n_classes, decay_steps=decay_steps)
    if dataset == "cifar10":
        return get_cifar10_model(config, n_classes, decay_steps=decay_steps)
    return None


def get_svhn_model(config: Config, n_classes: int = 10, decay_steps: int | None = None) -> Sequential:
    learner = Sequential()
    learner.add(RandomFlip("horizontal"))
    learner.add(RandomTranslation(0.1, 0.1))
    learner.add(RandomRotation(0.05))

    for i in range(config["n_blocks"]):
        n_block_filters = config["n_filters"] * (2**i)
        learner.add(
            Conv2D(
                n_block_filters,
                (config["kernel_size"], config["kernel_size"]),
                padding=config["padding"],
                activation="relu",
            )
        )
        learner.add(BatchNormalization())
        learner.add(
            Conv2D(
                n_block_filters,
                (config["kernel_size"], config["kernel_size"]),
                padding=config["padding"],
                activation="relu",
            )
        )
        learner.add(MaxPooling2D((2, 2)))
        learner.add(Dropout(config["dropout_rate"]))

    learner.add(Flatten())
    learner.add(Dense(config["n_units"], activation="relu"))
    learner.add(Dropout(config["final_dropout_rate"]))
    learner.add(Dense(n_classes, activation="softmax"))

    lr = CosineDecay(config["learning_rate"], decay_steps) if decay_steps else config["learning_rate"]
    optimizer = Adam(learning_rate=lr, weight_decay=config["weight_decay"])
    learner.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return learner


def get_mnist_model(config: Config, n_class: int = 10, decay_steps: int | None = None) -> Sequential:
    """Run one experiment, given a Data instance.

    :param {Data} data - The dataset to run on, NOT preprocessed.
    :param {dict} config - The config to use. Must be one in the format used in `process_configs`.
    :param {int} n_class - The number of classes in the dataset.
    """
    learner = Sequential()

    for _i in range(config["n_blocks"]):
        learner.add(
            Conv2D(
                config["n_filters"],
                (config["kernel_size"], config["kernel_size"]),
                padding=config["padding"],
                kernel_initializer="he_uniform",
                activation="relu",
            )
        )
        learner.add(MaxPooling2D(pool_size=(2, 2)))

    learner.add(Flatten())
    learner.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
    learner.add(Dense(n_class, activation="softmax"))

    lr = CosineDecay(config["learning_rate"], decay_steps) if decay_steps else config["learning_rate"]
    optimizer = Adam(learning_rate=lr, weight_decay=config["weight_decay"])
    learner.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return learner


def get_cifar10_model(config: Config, n_class: int = 10, decay_steps: int | None = None) -> Sequential:
    """Run one experiment given a Data insance."""
    learner = Sequential()
    learner.add(RandomTranslation(height_factor=4 / 32, width_factor=4 / 32, fill_mode="reflect"))
    learner.add(RandomFlip("horizontal"))

    for i in range(config["n_blocks"]):
        n_block_filters = config["n_filters"] * (2**i)
        learner.add(
            Conv2D(
                n_block_filters,
                (config["kernel_size"], config["kernel_size"]),
                padding=config["padding"],
            )
        )
        learner.add(BatchNormalization())
        learner.add(ReLU())
        learner.add(MaxPooling2D(pool_size=(2, 2)))

    learner.add(Flatten())
    learner.add(Dense(config["n_units"], activation="relu"))
    learner.add(Dense(n_class, activation="softmax"))

    lr = CosineDecay(config["learning_rate"], decay_steps) if decay_steps else config["learning_rate"]
    optimizer = Adam(learning_rate=lr, weight_decay=config["weight_decay"])
    learner.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return learner
