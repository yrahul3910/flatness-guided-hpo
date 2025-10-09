import gc

import numpy as np
from keras.models import Model

from util.config import Config
from util.data import Dataset
from util.model import get_model


def get_convexity(data: Dataset, config: Config) -> float:
    BATCH_SIZE = config.batch_size
    model = get_model(config, data)

    # Fit for one epoch before computing smoothness
    model.fit(data.x_train, data.y_train, batch_size=BATCH_SIZE, epochs=1),

    def Ka_func(xb):
        _model = Model(inputs=[model.layers[0].input], outputs=[model.layers[-2].output])
        result = _model(xb)
        del _model
        return result

    def Ka1_func(xb):
        _model = Model(inputs=[model.layers[0].input], outputs=[model.layers[-1].output])
        result = _model(xb)
        del _model
        return result

    best_mu = -np.inf
    for i in range((len(data.x_train) - 1) // BATCH_SIZE + 1):
        start_i = i * BATCH_SIZE
        end_i = start_i + BATCH_SIZE
        xb = data.x_train[start_i:end_i]

        mu = np.linalg.norm(Ka_func([xb])) * np.linalg.norm(Ka1_func([xb])) / np.linalg.norm(model.layers[-1].weights[0])
        if mu > best_mu and mu != np.inf:
            best_mu = mu

    del model
    gc.collect()

    return best_mu
