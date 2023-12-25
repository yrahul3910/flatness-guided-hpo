import random
from typing import List, Tuple

import numpy as np
import scipy
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.data import Dataset
from src.config import Config


BATCH_SIZE=64


def get_mnist():
    img_rows, img_cols = 28, 28
    return load_data()


def get_svhn():
    train_data = scipy.io.loadmat('train_32x32.mat')
    x_train = train_data['X']
    y_train = train_data['y']
    test_data = scipy.io.loadmat('test_32x32.mat')
    x_test = test_data['X']
    y_test = test_data['y']

    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)

    return (x_train, y_train), (x_test, y_test)


def get_data(dataset: str='svhn'):
    data_loaders = {
        'mnist': (get_mnist, (28, 28, 1)),
        'svhn': (get_svhn, (32, 32, 3))
    }

    if dataset not in data_loaders:
        raise ValueError('Invalid dataset name.')

    data_loader, (img_rows, img_cols, img_channels) = data_loaders[dataset]
    (x_train, y_train), (x_test, y_test) = data_loader()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
        input_shape = (img_channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
        input_shape = (img_rows, img_cols, img_channels)

    """
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    """
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)
    return Dataset(x_train, y_train, x_test, y_test)


def run_experiment(data: Dataset, config: Config, n_class: int = 10, dataset: str = 'mnist') -> float:
    print('[run_experiment] Getting model')
    model = get_model(data, config, n_class, dataset)

    print('[run_experiment] Got model')
    model.fit(data.x_train, data.y_train, epochs=50, verbose=1, batch_size=BATCH_SIZE)
    print('[run_experiment] Fit model')

    y_pred = np.argmax(model.predict(data.x_test), axis=-1)
    
    if len(data.y_test.shape) > 1:
        data.y_test = np.argmax(data.y_test, axis=1)

    return accuracy_score(data.y_test, y_pred)


def get_convexity(data: Dataset, config: Config, n_class: int = 10, dataset: str = 'mnist') -> float:
    model = get_model(data, config, n_class, dataset)

    if n_class > 2 and len(data.y_train.shape) == 1:
        data.y_train = to_categorical(data.y_train, n_class)
        data.y_test = to_categorical(data.y_test, n_class)

    # Fit for one epoch before computing smoothness
    model.fit(data.x_train, data.y_train, batch_size=BATCH_SIZE, epochs=1),

    Ka_func = K.function([model.layers[0].input], [model.layers[-2].output])

    batch_size = BATCH_SIZE
    best_mu = np.inf
    for i in range((len(data.x_train) - 1) // batch_size + 1):
        start_i = i * batch_size
        end_i = start_i + batch_size
        xb = data.x_train[start_i:end_i]

        mu = np.linalg.norm(Ka_func([xb])) / np.linalg.norm(model.layers[-1].weights[0])
        if mu < best_mu and mu != np.inf:
            best_mu = mu

    return best_mu


def get_random_hyperparams(options: dict) -> Config:
    """
    Get hyperparameters from options.
    """
    hyperparams = {}
    for key, value in options.items():
        if isinstance(value, list):
            hyperparams[key] = random.choice(value)
        elif isinstance(value, tuple):
            if isinstance(value[0], int):
                hyperparams[key] = random.randint(value[0], value[1])
            else:
                hyperparams[key] = random.uniform(value[0], value[1])
    return Config(**hyperparams)


def get_many_random_hyperparams(options: dict, n: int) -> list:
    """
    Get n hyperparameters from options.
    """
    hyperparams = []
    for _ in range(n):
        hyperparams.append(get_random_hyperparams(options))
    return hyperparams


def get_model(data: Dataset, config: Config, n_classes: int = 10, dataset: str = 'mnist') -> Sequential:
    if dataset == 'mnist':
            return get_mnist_model(data, config, n_classes)
    elif dataset == 'svhn':
            return get_svhn_model(data, config, n_classes)


def get_svhn_model(data: Dataset, config: Config, n_classes: int = 10) -> Sequential:
    learner = Sequential()

    for i in range(config.n_blocks):
        n_block_filters = config.n_filters * (2 ** i)
        learner.add(Conv2D(n_block_filters, (config.kernel_size, config.kernel_size), padding=config.padding, activation='relu'))
        learner.add(BatchNormalization())
        learner.add(Conv2D(n_block_filters, (config.kernel_size, config.kernel_size), padding=config.padding, activation='relu'))
        learner.add(MaxPooling2D((2, 2)))
        learner.add(Dropout(config.dropout_rate))

    learner.add(Flatten())
    learner.add(Dense(config.n_units, activation='relu'))
    learner.add(Dropout(config.final_dropout_rate))
    learner.add(Dense(n_classes, activation='softmax'))
    learner.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return learner


def get_mnist_model(data: Dataset, config: Config, n_class: int = 10) -> Sequential:
    """
    Runs one experiment, given a Data instance.

    :param {Data} data - The dataset to run on, NOT preprocessed.
    :param {dict} config - The config to use. Must be one in the format used in `process_configs`.
    :param {int} n_class - The number of classes in the dataset.
    """
    learner = Sequential()

    for i in range(config.n_blocks):
        learner.add(Conv2D(config.n_filters, (config.kernel_size, config.kernel_size), padding=config.padding, kernel_initializer='he_uniform', activation='relu'))
        learner.add(MaxPooling2D(pool_size=(2, 2)))

    learner.add(Flatten())
    learner.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    learner.add(Dense(n_class, activation='softmax'))
    
    learner.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return learner
