from util.config import Config
from util.data import Dataset
from util.model import get_model

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import numpy as np


def get_convexity(data: Dataset, config: Config, n_class: int = 10, dataset: str = 'mnist') -> float:
    BATCH_SIZE = config.batch_size
    model = get_model(config, data)

    if n_class > 2 and len(data.y_train.shape) == 1:
        data.y_train = to_categorical(data.y_train, n_class)
        data.y_test = to_categorical(data.y_test, n_class)

    # Fit for one epoch before computing smoothness
    model.fit(data.x_train, data.y_train, batch_size=BATCH_SIZE, epochs=1, verbose=0),

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