import gc

from keras.callbacks import EarlyStopping

from util.config import Config
from util.data import Dataset
from util.model import get_model


def eval(config: Config, data: Dataset, *args, **kwargs) -> list[float]:
    if isinstance(config, dict):
        config = Config(**config)

    model = get_model(config, data)

    try:
        early_stop = EarlyStopping(monitor="loss", patience=5, min_delta=1e-3)
        model.fit(data.x_train, data.y_train, batch_size=int(config.batch_size), epochs=200, callbacks=[early_stop])
        score = model.evaluate(data.x_test, data.y_test, verbose=0)


        gc.collect()
    except ValueError:
        return [0., 0., 0.]

    return score
