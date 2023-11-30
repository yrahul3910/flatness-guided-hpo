from src.config import Config
from src.util import get_data, get_model


hpo_space = {
    'n_filters': (2, 6),
    'kernel_size': (2, 6),
    'padding': ['valid', 'same'],
    'n_blocks': (1, 3)
}

def eval(config, *args, **kwargs):
    config = Config(**config)
    data = get_data()
    model = get_model(data, config, 10)
    model.fit(data.x_train, data.y_train, batch_size=128, epochs=50)
    scores = model.evaluate(data.x_test, data.y_test, verbose=0)[-1]

    return scores